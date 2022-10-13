#!/bin/bash
cd ..



# , seed 1 --model cnn
python classification_image.py --method augerino --dataset translated_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn
python classification_image.py --method augerino --dataset fmnist_r180 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn
python classification_image.py --method augerino --dataset fmnist_r90 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn
python classification_image.py --method augerino --dataset scaled_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn
python classification_image.py --method augerino --dataset fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 1 --model cnn
# , seed 2 --model cnn
python classification_image.py --method augerino --dataset translated_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn
python classification_image.py --method augerino --dataset fmnist_r180 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn
python classification_image.py --method augerino --dataset fmnist_r90 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn
python classification_image.py --method augerino --dataset scaled_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn
python classification_image.py --method augerino --dataset fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 2 --model cnn
# , seed 3 --model cnn
python classification_image.py --method augerino --dataset translated_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn
python classification_image.py --method augerino --dataset fmnist_r180 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn
python classification_image.py --method augerino --dataset fmnist_r90 --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn
python classification_image.py --method augerino --dataset scaled_fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn
python classification_image.py --method augerino --dataset fmnist --n_epochs 300 --device cuda --n_samples_aug 31 --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed 3 --model cnn
