# reg-cyclical-self-train
Source code for our Miccai2023 paper [Unsupervised 3D registration through optimization-guided cyclical self-training](https://arxiv.org/abs/2306.16997) [[pdf](https://arxiv.org/pdf/2306.16997.pdf)].

# Dependencies
Please first install the following dependencies
* Python3 (we use 3.9.7)
* pytorch (we use 1.10.2)
* numpy
* scipy
* nibabel

# Data Preparation
1. Download the Abdomen CT-CT dataset of the [Learn2Reg challenge](https://learn2reg.grand-challenge.org/Datasets/).
2. Modify the variable path in line 8 of `data_utils.py` such that it points to the root directory of the data.

# Training
Execute `python main.py --phase train --out_dir PATH/TO/OUTDIR --gpu GPU --num_warps 2 --ice true --reg_fac 1. --augment true --adam true --sampling true`.

# Testing
In l. 9 of `test.py`, set the path to the model weights you want to use for testing (for example our `final_model.pth`). Subsequently, execute `python main.py --phase test --gpu GPU`

## Citation
If you find our code useful for your work, please cite the following paper
```latex
@article{unsupervised,
  title={Unsupervised 3D registration through optimization-guided cyclical self-training},
  author={Bigalke, Alexander and Hansen, Lasse and Mok, Tony C. W. and Heinrich, Mattias P},
  journal={arXiv preprint arXiv:2306.16997},
  year={2023}
}
```
