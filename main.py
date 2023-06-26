import os
import argparse

from train import train
from test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # train or test
    parser.add_argument(
        "--phase",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="directory to write results to",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    parser.add_argument(
        "--num_warps",
        default=1,
        type=int,
    )
    # whether to use inverse consistence
    parser.add_argument(
        "--ice",
        default="false",
        type=str,
    )
    # regularization weight in Adam optimization during training
    parser.add_argument(
        "--reg_fac",
        default=1.,
        type=float,
    )
    # whether to perform difficulty-weighted data sampling during training
    parser.add_argument(
        "--sampling",
        default="true",
        type=str,
    )
    # whether to finetune pseudo labels with Adam instance optimization during training
    parser.add_argument(
        "--adam",
        default="true",
        type=str,
    )
    # whether to use affine input augmentations during training
    parser.add_argument(
        "--augment",
        default="true",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    if args.phase == 'test':
        test()

    else:
        train(args)
