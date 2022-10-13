#!/bin/bash
cd ..


# # , seed 1 --model cnn
python classification_image.py --method augerino --dataset translated_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 1 --model cnn
python classification_image.py --method augerino --dataset mnist_r180 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 1 --model cnn
python classification_image.py --method augerino --dataset mnist_r90 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 1 --model cnn
python classification_image.py --method augerino --dataset scaled_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 1 --model cnn
python classification_image.py --method augerino --dataset mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 1 --model cnn
# , seed 2 --model cnn
python classification_image.py --method augerino --dataset mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 2 --model cnn
python classification_image.py --method augerino --dataset mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 3 --model cnn
python classification_image.py --method augerino --dataset translated_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 2 --model cnn
python classification_image.py --method augerino --dataset mnist_r180 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 2 --model cnn
python classification_image.py --method augerino --dataset mnist_r90 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 2 --model cnn
python classification_image.py --method augerino --dataset scaled_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 2 --model cnn
# , seed 3 --model cnn
python classification_image.py --method augerino --dataset translated_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 3 --model cnn
python classification_image.py --method augerino --dataset mnist_r180 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 3 --model cnn
python classification_image.py --method augerino --dataset mnist_r90 --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 3 --model cnn
python classification_image.py --method augerino --dataset scaled_mnist --n_epochs 1000 --device cuda --subset_size 312 --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 312 --seed 3 --model cnn

