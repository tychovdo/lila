# Learning Invariances with Laplace Approximations (LILA)

A convenient gradient-based method for selecting the data augmentation without validation data and during training of a deep neural network. 

## Paper

Code for "[Invariance Learning in Deep Neural Networks with Differentiable Laplace Approximations](https://arxiv.org/abs/2202.10638)" paper by Alexander Immer*, Tycho F.A. van der Ouderaa*, Vincent Fortuin, Gunnar Rätsch, Mark van der Wilk. In NeurIPS 2022.


## Setup
Python 3.8 is required.

```bash
pip install -r requirements.txt
```
Create directory for results: `mkdir results` in the root of the project.

### Install custom Laplace and ASDL
```bash
pip install git+https://github.com/kazukiosawa/asdfghjkl.git@dev-alex
pip install git+https://github.com/AlexImmer/Laplace.git@lila
```

## Example runs

### Run Illustrative Example and Plot Predictive
```bash
python classification_illustration.py --method avgfunc --approximation_structure kron --curvature_type ggn --n_epochs 500 --n_obs 200 --rotation_max 120 --sigma_noise 0.06 --n_samples_aug 100 --rotation_init 0 --optimize_aug --plot --posterior_predictive --lr_aug 0.005 --lr_aug_min 0.00001
```