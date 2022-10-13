#!/bin/bash
cd ..



# # , seed 1
python classification_image.py --method avgfunc --dataset translated_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r180 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r90 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset scaled_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
# , seed 2
python classification_image.py --method avgfunc --dataset translated_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r180 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r90 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset scaled_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
# , seed 3
python classification_image.py --method avgfunc --dataset translated_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r180 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist_r90 --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset scaled_fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
python classification_image.py --method avgfunc --dataset fmnist --n_epochs 1000 --device cuda --subset_size 5000 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn --n_hypersteps 5 --accum_aug_grad --marglik_batch_size 256 --n_epochs_burnin 10
