import torch
import torch.nn as nn
import torch.nn.functional as F


# Instance optimization with Adam: minimize joint cost function of feature similarity and regularization
def AdamReg(mind_fix, mind_mov, dense_flow, reg_fac=1.):
    if (dense_flow.shape[-1] == 3):
        dense_flow = dense_flow.permute(0, 4, 1, 2, 3)

    H, W, D = dense_flow[0, 0].shape

    disp_hr = dense_flow.cuda().flip(1) * torch.tensor([H - 1, W - 1, D - 1]).cuda().view(1, 3, 1, 1, 1) / 2
    with torch.enable_grad():
        grid_sp = 2

        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                align_corners=False)
        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H // grid_sp, W // grid_sp, D // grid_sp),
                              align_corners=False)
        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = .65
        for iter in range(50):
            optimizer.zero_grad()
            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1, padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
            scale = torch.tensor(
                [(H // grid_sp - 1) / 2, (W // grid_sp - 1) / 2, (D // grid_sp - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(mind_mov.cuda().float(),
                                              grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).cuda() , align_corners=False, mode='bilinear')
            sampled_cost = (patch_mov_sampled - mind_fix.cuda()).pow(2).mean(1) * 12
            loss = sampled_cost.mean()

            (loss + reg_fac * reg_loss).backward()
            optimizer.step()
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(fitted_grid * grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)

    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1), 3,
                               padding=1, stride=1)

    disp_hr = torch.flip(disp_smooth / torch.tensor([H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1).cuda() * 2, [1])
    return disp_hr