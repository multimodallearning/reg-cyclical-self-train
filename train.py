import sys
import time
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from data_utils import prepare_data, augment_affine_nl
from registration_pipeline import update_fields
from coupled_convex import coupled_convex


def train(args):
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_warps = args.num_warps
    reg_fac = args.reg_fac
    use_ice = True if args.ice == 'true' else False
    use_adam = True if args.adam == 'true' else False
    do_sampling = True if args.sampling == 'true' else False
    do_augment = True if args.augment == 'true' else False

    # Loading training data (segmentations only used for validation after each stage)
    data = prepare_data(data_split='train')

    # initialize feature net
    feature_net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                           nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                           nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 16, 1)).cuda()
    print()

    N, _, H, W, D = data['images'].shape

    # generate initial pseudo labels with random features
    if use_adam:
        # w/o Adam finetuning
        all_fields_noadam, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False, num_warps=num_warps,
                                                                 ice=use_ice, reg_fac=reg_fac)
        # w/ Adam finetuning
        all_fields, _, _, d_all_adam, _ = update_fields(data, feature_net, use_adam=True, num_warps=num_warps, ice=use_ice,
                                                        reg_fac=reg_fac)
        # compute difference between finetuned and non-finetuned fields for difficulty sampling --> the larger the difference, the more difficult the sample
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tre_adam = ((all_fields_noadam[:, :, 8:-8, 8:-8, 8:-8].cuda() - all_fields[:, :, 8:-8, 8:-8,
                                                                              8:-8].cuda()) * torch.tensor(
                    [D / 2, W / 2, H / 2]).cuda().view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                tre_adam1 = (tre_adam.mean(-1).mean(-1).mean(-1))
        print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean(), '>', d_all_adam[:3].mean())

    else:
        # w/o Adam finetuning
        all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False, num_warps=num_warps,
                                                            ice=use_ice, reg_fac=reg_fac)
        print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

    # reinitialize feature net with novel random weights
    feature_net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                           nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                           nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 16, 1)).cuda()

    # perform overall 8 (2x4) cycle of self-training
    for repeat in range(2):
        stage = 0 + repeat * 4

        feature_net.cuda()
        feature_net.train()
        print()

        optimizer = torch.optim.Adam(feature_net.parameters(), lr=0.001)
        eta_min = 0.00001
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500 * 2, 1, eta_min=eta_min)
        run_lr = torch.zeros(2000 * 2)
        half_iterations = 2000 * 2
        run_loss = torch.zeros(half_iterations)
        scaler = torch.cuda.amp.GradScaler()

        # placeholders for input images, pseudo labels, and affine augmentation matrices
        img0 = torch.zeros(2, 1, H, W, D).cuda()
        img1 = torch.zeros(2, 1, H, W, D).cuda()
        target = torch.zeros(2, 3, H // 2, W // 2, D // 2).cuda()
        affine1 = torch.zeros(2,  H, W, D, 3).cuda()
        affine2 = torch.zeros(2,  H, W, D, 3).cuda()

        t0 = time.time()
        with tqdm(total=half_iterations, file=sys.stdout, colour="red") as pbar:
            for i in range(half_iterations):
                optimizer.zero_grad()
                # difficulty weighting
                if use_adam and do_sampling:
                    q = torch.zeros(len(data['pairs']))
                    q[torch.argsort(tre_adam1)] = torch.sigmoid(torch.linspace(5, -5, len(data['pairs'])))
                else:
                    q = torch.ones(len(data['pairs']))
                idx = torch.tensor(list(WeightedRandomSampler(q, 2, replacement=True))).long()

                with torch.cuda.amp.autocast():
                    # image selection and augmentation
                    img0_ = data['images'][data['pairs'][idx, 0]].cuda()
                    img1_ = data['images'][data['pairs'][idx, 1]].cuda()
                    if do_augment:
                        with torch.no_grad():
                            for j in range(len(idx)):
                                disp_field = all_fields[idx[j]:idx[j] + 1].cuda()
                                disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(disp_field)
                                img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                                img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                                target[j:j + 1] = disp_field_aff
                    else:
                        with torch.no_grad():
                            for j in range(len(idx)):
                                input_field = all_fields[idx[j]:idx[j] + 1].cuda()
                                disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(input_field, strength=0.)
                                img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                                img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                                target[j:j + 1] = disp_field_aff
                    img0.requires_grad = True
                    img1.requires_grad = True

                    # feature extraction with feature net g
                    features_fix = feature_net(img0)
                    features_mov = feature_net(img1)

                    # differentiable optimization with optimizer h (coupled convex)
                    disp_pred = coupled_convex(features_fix, features_mov, use_ice=False, img_shape=(H//2, W//2, D//2))

                    # consistency loss between prediction and pseudo label
                    tre = ((disp_pred[:, :, 8:-8, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8, 8:-8]) * torch.tensor(
                        [D / 2, W / 2, H / 2]).cuda().view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                    loss = tre.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr1 = float(scheduler.get_last_lr()[0])
                run_lr[i] = lr1

                if ((i % 1000 == 999)):
                    # end of stage
                    stage += 1
                    torch.save(feature_net.cpu(), os.path.join(out_dir, 'stage' + str(stage) + '.pth'))
                    feature_net.cuda()
                    torch.save(run_loss, os.path.join(out_dir, 'run_loss_rep={}.pth'.format(repeat)))
                    print()

                    #  recompute pseudo-labels with current model weights
                    if use_adam:
                        # w/o Adam finetuning
                        all_fields_noadam, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False,
                                                                                 num_warps=num_warps, ice=use_ice,
                                                                                 reg_fac=reg_fac)
                        # w Adam finetuning
                        all_fields, _, _, d_all_adam, _ = update_fields(data, feature_net, use_adam=True, num_warps=num_warps,
                                                                        ice=use_ice, reg_fac=reg_fac)

                        # recompute difference between finetuned and non-finetuned fields for difficulty sampling --> the larger the difference, the more difficult the sample
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                tre_adam = ((all_fields_noadam[:, :, 8:-8, 8:-8, 8:-8].cuda() - all_fields[:, :, 8:-8,
                                                                                              8:-8,
                                                                                              8:-8].cuda()) * torch.tensor(
                                    [D / 2, W / 2, H / 2]).cuda().view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                                tre_adam1 = (tre_adam.mean(-1).mean(-1).mean(-1))

                        print('fields updated val error :', d_all0[:3].mean(), '>', d_all_net[:3].mean(), '>',
                              d_all_adam[:3].mean())

                    else:
                        # w/o Adam finetuning
                        all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False,
                                                                            num_warps=num_warps, ice=use_ice, reg_fac=reg_fac)
                        print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

                    feature_net.train()

                run_loss[i] = loss.item()

                str1 = f"iter: {i}, loss: {'%0.3f' % (run_loss[i - 34:i - 1].mean())}, runtime: {'%0.3f' % (time.time() - t0)} sec, GPU max/memory: {'%0.2f' % (torch.cuda.max_memory_allocated() * 1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
