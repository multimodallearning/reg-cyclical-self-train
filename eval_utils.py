import torch
import torch.nn as nn


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label - 1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def jacobian_determinant(disp):
    device = disp.device

    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(device)

    jacobian = torch.cat((gradz(disp), grady(disp), gradx(disp)), 0) + torch.eye(3, 3, device=device).view(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet.unsqueeze(0).unsqueeze(0)