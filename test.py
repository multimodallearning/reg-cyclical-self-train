import torch

from data_utils import prepare_data
from registration_pipeline import update_fields


def test():
    data = prepare_data(data_split='test')
    feature_net = torch.load('PATH/TO/MODEL').cuda()

    all_fields, d_all_net, d_all0, d_all_adam, d_all_ident = update_fields(data, feature_net, False, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10.)
    print('DSC:', d_all0.sum() / (d_all_ident > 0.1).sum(), '>', d_all_net.sum() / (d_all_ident > 0.1).sum(), '>', d_all_adam.sum() / (d_all_ident > 0.1).sum())
