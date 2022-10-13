# Learning Invariances with Laplace Approximations (LILA)

A convenient gradient-based method for selecting the data augmentation without validation data and during training of a deep neural network. 

![](https://github.com/tychovdo/lila/blob/main/figs/gif_demo_bar.gif)

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


### Example of ResNet on CIFAR-10

To run LILA:

```bash
python classification_image.py --dataset cifar10 --model resnet_8_8 --approx ggn_kron --n_epochs 200 --batch_size 250 --marglik_batch_size 125 --partial_batch_size 50 --lr 0.1 --n_epochs_burnin 10 --n_hypersteps 100 --n_hypersteps_prior 4 --lr_aug 0.05 --lr_aug_min 0.005 --use_jvp --method avgfunc --n_samples_aug 20 --optimize_aug --download
```

### Example of MLP on translated MNIST

To run LILA:

```bash
python classification_image.py --method avgfunc --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --download
```

To run Augerino:
```
python classification_image.py --method augerino --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --download
```

To run non-invariant baseline:
```
python classification_image.py --method avgfunc --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --approx ggn_kron --batch_size 1000 --seed 1 --download
```



## Reproducibility

All experiments in the paper can be replicated using runscripts in

```bash
experimentscripts/
```

If you do run into issues, please get in touch so we can provide support.
