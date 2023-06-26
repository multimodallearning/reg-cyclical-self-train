import torch
import torch.nn.functional as F


# correlation layer in 3D similar to PWC-Net
def correlate(mind_fix, mind_mov, disp_hw):
    B, C_mind, H, W, D = mind_fix.shape
    ssd = torch.zeros(B, (disp_hw * 2 + 1) ** 3, H, W, D, dtype=mind_fix.dtype,
                      device=mind_fix.device)
    for b in range(B):
        mind_unfold = F.unfold(
            F.pad(mind_mov[b:b + 1].squeeze(0), (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)).squeeze(0),
            disp_hw * 2 + 1)
        mind_unfold = mind_unfold.view(C_mind, -1, (disp_hw * 2 + 1) ** 2, W, D)

        for i in range(disp_hw * 2 + 1):
            mind_sum = (mind_fix[b:b + 1].permute(1, 2, 0, 3, 4) - mind_unfold[:, i:i + H]).abs().sum(0, keepdim=True)

            ssd[b, i::(disp_hw * 2 + 1)] = F.avg_pool3d(mind_sum.transpose(2, 1), 3, stride=1, padding=1).squeeze(1)
    ssd = ssd.view(B, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1, H, W, D).transpose(2, 1).reshape(B, (
                disp_hw * 2 + 1) ** 3, H, W, D)

    return ssd


# solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex_optim(ssd, disp_mesh_t, delta=20):
    B, C, H_, W_, D_ = ssd.shape  # [-3:]
    disp_soft = torch.zeros(B, 3, H_, W_, D_).to(ssd.device)
    for i in range(H_):
        output = (disp_mesh_t.view(1, 3, C, 1) * torch.softmax(-delta * ssd[:, :, i].view(B, 1, C, -1), 2)).sum(2)

        disp_soft[:, :, i] = output.view(B, 3, W_, D_)

    coeffs = torch.tensor([0.003, 0.03, 0.3])
    for j in range(len(coeffs)):

        for i in range(H_):
            coupled = ssd[:, :, i, :, :] + coeffs[j] * (
                        disp_mesh_t.unsqueeze(0) - disp_soft[:, :, i].view(B, 3, 1, -1)).pow(2).sum(1).view(B, -1, W_, D_)
            output = (disp_mesh_t.view(1, 3, C, 1) * torch.softmax(-delta * coupled.view(B, 1, C, -1), 2)).sum(2)

            disp_soft[:, :, i] = output.view(B, 3, W_, D_)
        disp_soft = F.avg_pool3d(disp_soft.reshape(B, 3, H_, W_, D_), 3, padding=1, stride=1)
    return disp_soft


# enforce inverse consistency of forward and backward transform
def inverse_consistency(disp_field1s, disp_field2s, iter=20):
    B,C,H,W,D = disp_field1s.size()
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3).to(disp_field1s.device).to(disp_field1s.dtype)
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i,disp_field2i


# the optimizer h in our framework
def coupled_convex(feat_fix, feat_mov, use_ice, img_shape):
    disp_hw = 2
    ssd = correlate(5 * (feat_fix), 5 * (feat_mov), disp_hw)
    disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().unsqueeze(0),
                                (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
                                align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    H2, W2, D2 = ssd.shape[-3:]
    disp_soft0 = coupled_convex_optim(ssd, disp_mesh_t)

    if use_ice:
        ssd_ = correlate(5 * (feat_mov), 5 * (feat_fix), disp_hw)
        disp_soft0_ = coupled_convex_optim(ssd_, disp_mesh_t)
        scale = torch.tensor([H2 - 1, W2 - 1, D2 - 1]).view(1, 3, 1, 1, 1).cuda() / 2
        disp_soft0 = inverse_consistency((disp_soft0 / scale).flip(1), (disp_soft0_ / scale).flip(1), iter=15)[0].flip(1) * scale

    scale = torch.tensor([H2 - 1, W2 - 1, D2 - 1]).view(1, 3, 1, 1, 1).cuda() / 2
    disp_torch = F.avg_pool3d(
        F.avg_pool3d(F.interpolate((disp_soft0 / scale).flip(1), scale_factor=2, mode='trilinear'), 3,
                     padding=1, stride=1), 3, padding=1, stride=1)

    disp_soft1 = F.interpolate(disp_torch, scale_factor=2, mode='trilinear', align_corners=False,
                               recompute_scale_factor=False)
    disp_soft2 = F.avg_pool3d(F.avg_pool3d(disp_soft1, 3, stride=1, padding=1), 3, stride=1, padding=1)
    disp_soft3 = F.interpolate(disp_soft2, size=img_shape, mode='trilinear')

    return disp_soft3
