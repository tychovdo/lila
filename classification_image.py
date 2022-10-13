import numpy as np
import torch
import pickle
from copy import deepcopy
from absl import app, flags, logging
from pathlib import Path
from torchvision import transforms
from torch.nn.utils.convert_parameters import parameters_to_vector

from laplace.curvature.augmented_asdl import AugAsdlGGN, AugAsdlEF

from bayinv.marglik import marglik_opt_jvecprod, marglik_optimization
from bayinv.datasets import RotatedMNIST, TranslatedMNIST, ScaledMNIST, ScaledMNIST2
from bayinv.datasets import RotatedFashionMNIST, TranslatedFashionMNIST, ScaledFashionMNIST
from bayinv.datasets import RotatedCIFAR10, TranslatedCIFAR10, ScaledCIFAR10
from bayinv.utils import TensorDataLoader, dataset_to_tensors, get_laplace_approximation, set_seed
from bayinv.layers import AffineLayer2d
from bayinv.augerino import augerino
from bayinv.models import MLP, LeNet, ResNet, WideResNet


FLAGS = flags.FLAGS
np.set_printoptions(precision=3)

flags.DEFINE_integer('seed', 137, 'Random seed for data generation and model initialization.')
flags.DEFINE_enum(
    'method', 'avgfunc', ['avgfunc', 'augerino'],
    'Available methods: `avgfunc` is averaging functions, `augerino` is by Benton et al.')
flags.DEFINE_float('augerino_reg', 1e-2, 'Augerino regularization strength (default from paper).')
flags.DEFINE_enum(
    'dataset', 'mnist',
    ['mnist', 'mnist_r90', 'mnist_r180', 'translated_mnist', 'scaled_mnist', 'scaled_mnist2',
     'fmnist', 'fmnist_r90', 'fmnist_r180', 'translated_fmnist', 'scaled_fmnist', 'scaled_fmnist2',
     'cifar10', 'cifar10_r90', 'cifar10_r180', 'translated_cifar10', 'scaled_cifar10'],
    'Available methods: `mnist` is plain MNIST data, `mnist_r90` is partially-rotated ±90° MNIST, `mnist_r180` is fully-rotated ±180° MNIST')
flags.DEFINE_enum('model', 'mlp', ['mlp', 'cnn', 'resnet_8_16', 'resnet_8_8', 'wrn'], help='model architecture')
flags.DEFINE_enum(
    'approx', 'ggn_kron',
    ['ggn_diag', 'ggn_kron', 'ggn_full', 'ef_diag', 'ef_kron', 'ef_full'],
    'Laplace and Hessian approximation type'
)
flags.DEFINE_bool('use_jvp', False, 'whether to use JVP instead of sub-add trick')
flags.DEFINE_integer('n_epochs', 2000, 'Number of epochs')
flags.DEFINE_integer('batch_size', -1, 'Batch size for stochastic estimates. If set to -1 (default), use full batch.')
flags.DEFINE_integer('marglik_batch_size', -1, 'Batch size for marginla likelihood estimation, -1 set to train_loader')
flags.DEFINE_integer('partial_batch_size', -1, 'Batch size for partial marginla likelihood estimation, -1 set to marglik_loader')
flags.DEFINE_integer('subset_size', -1, 'Size of random subset, subset_size <= 0 means full data set')
flags.DEFINE_integer('n_samples_aug', 11, 'number of augmentation samples if applicable')
flags.DEFINE_bool('optimize_aug', False, 'Whether to differentiably optimize augmenter')
flags.DEFINE_bool('softplus', False, 'Whether to use softplus on the rot factor parameters')
flags.DEFINE_float('init_aug', 0.0, 'Initial value for the transformation parameters (before softplus if its used)')
flags.DEFINE_bool('save', True, 'Whether to save the experiment outcome as pickle')
flags.DEFINE_enum('device', 'cuda', ['cpu', 'cuda'], 'Torch device')
flags.DEFINE_bool('download', False, 'whether to (re-)download data set')
flags.DEFINE_string('data_root', 'tmp', 'root of the data directory')

flags.DEFINE_float('lr', 0.005, 'lr')
flags.DEFINE_float('lr_min', 1e-6, 'decay target of the learning rate')
flags.DEFINE_float('lr_hyp', 0.05, 'lr hyper')
flags.DEFINE_float('lr_aug', 0.05, 'lr_augmentation')
flags.DEFINE_float('lr_aug_min', 0.05, 'lr_augmentation decayed to')
flags.DEFINE_float('lr_augerino', 0.005, 'lr')
flags.DEFINE_float('prior_prec_init', 1.0, 'prior precision init')
flags.DEFINE_integer('n_epochs_burnin', 10, 'number of epochs without marglik opt')
flags.DEFINE_integer('marglik_frequency', 1, 'frequency of computing marglik in terms of epochs')
flags.DEFINE_integer(
    'n_hypersteps', 1,
    'number of steps on every marginal likelihood estimate (repeated fit, partial fit (default))'
)
flags.DEFINE_integer('n_hypersteps_prior', 1, 'hyper steps for prior')


def main(argv):
    # dataset-specific static transforms (preprocessing)
    if 'mnist' in FLAGS.dataset:
        transform = transforms.ToTensor()
    elif 'cifar' in FLAGS.dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        raise NotImplementedError(f'Transform for {FLAGS.dataset} unavailable.')

    # Load data
    if FLAGS.dataset == 'mnist':
        train_dataset = RotatedMNIST(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'mnist_r90':
        train_dataset = RotatedMNIST(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'mnist_r180':
        train_dataset = RotatedMNIST(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_mnist':
        train_dataset = TranslatedMNIST(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedMNIST(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_mnist':
        train_dataset = ScaledMNIST(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledMNIST(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_mnist2': # independently scales x- and y-axis
        train_dataset = ScaledMNIST2(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledMNIST2(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist_r90':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist_r180':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_fmnist':
        train_dataset = TranslatedFashionMNIST(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedFashionMNIST(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_fmnist':
        train_dataset = ScaledFashionMNIST(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledFashionMNIST(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10_r90':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10_r180':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_cifar10':
        train_dataset = TranslatedCIFAR10(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedCIFAR10(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_cifar10':
        train_dataset = ScaledCIFAR10(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledCIFAR10(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    else:
        raise NotImplementedError(f'Unknown dataset: {FLAGS.dataset}')

    set_seed(FLAGS.seed)

    # Subset the data if subset_size is given.
    subset_size = len(train_dataset) if FLAGS.subset_size <= 0 else FLAGS.subset_size
    if subset_size < len(train_dataset):
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    else:
        subset_indices = None
    X_train, y_train = dataset_to_tensors(train_dataset, subset_indices, FLAGS.device)
    X_test, y_test = dataset_to_tensors(test_dataset, None, FLAGS.device)

    augmenter = AffineLayer2d(n_samples=FLAGS.n_samples_aug, init_value=FLAGS.init_aug,
                              softplus=FLAGS.softplus).to(FLAGS.device)
    augmenter.rot_factor.requires_grad = FLAGS.optimize_aug

    if FLAGS.batch_size <= 0:  # full batch
        batch_size = subset_size
    else:
        batch_size = min(FLAGS.batch_size, subset_size)

    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=batch_size, shuffle=True, detach=True)
    valid_loader = TensorDataLoader(X_test, y_test, transform=augmenter, batch_size=batch_size, detach=True)
    if FLAGS.marglik_batch_size == batch_size or FLAGS.marglik_batch_size <= 0:
        marglik_loader = deepcopy(train_loader).attach()
    else:
        ml_batch_size = min(FLAGS.marglik_batch_size, subset_size)
        marglik_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=ml_batch_size,
                                          shuffle=True, detach=False)

    if FLAGS.partial_batch_size == FLAGS.marglik_batch_size or FLAGS.partial_batch_size <= 0:
        # use marglik loader
        partial_loader = marglik_loader
    else:
        pl_batch_size = min(FLAGS.partial_batch_size, subset_size)
        partial_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=pl_batch_size, shuffle=True)

    # model
    if 'mnist' in FLAGS.dataset:
        optimizer = 'Adam'
        prior_structure = 'scalar'
        if FLAGS.model == 'mlp':
            model = MLP(28*28, width=1000, depth=1, output_size=10, fixup=False, activation='tanh')
        elif FLAGS.model == 'cnn':
            model = LeNet(in_channels=1, n_out=10, activation='tanh', n_pixels=28)
        else:
            raise ValueError('Unavailable model')
    elif 'cifar10' in FLAGS.dataset:
        optimizer = 'SGD'
        prior_structure = 'layerwise'  # for fixup params
        if FLAGS.model == 'cnn':
            model = LeNet(in_channels=3, activation='relu', n_pixels=32)
        elif FLAGS.model == 'resnet_8_8':
            model = ResNet(depth=8, num_classes=10, in_planes=8, in_channels=3)
        elif FLAGS.model == 'resnet_8_16':
            model = ResNet(depth=8, num_classes=10, in_planes=16, in_channels=3)
        elif FLAGS.model == 'resnet_14_16':
            model = ResNet(depth=14, num_classes=10, in_planes=16, in_channels=3)
        elif FLAGS.model == 'wrn':
            model = WideResNet()
        else:
            raise ValueError('Unavailable model')

    model.to(FLAGS.device)

    result = dict()
    if FLAGS.method != 'augerino':  # LA marglik methods
        # Resolve backend and LA type
        hess_approx, la_structure = FLAGS.approx.split('_')
        laplace = get_laplace_approximation(la_structure)
        backend = AugAsdlGGN if hess_approx == 'ggn' else AugAsdlEF

        if FLAGS.use_jvp:
            la, model, margliks, valid_perfs, aug_history = marglik_opt_jvecprod(
                model, train_loader, marglik_loader, valid_loader, partial_loader, likelihood='classification',
                lr=FLAGS.lr, lr_hyp=FLAGS.lr_hyp, lr_aug=FLAGS.lr_aug, n_epochs=FLAGS.n_epochs,
                n_hypersteps=FLAGS.n_hypersteps, marglik_frequency=FLAGS.marglik_frequency, laplace=laplace,
                prior_structure=prior_structure, backend=backend, n_epochs_burnin=FLAGS.n_epochs_burnin,
                method=FLAGS.method, augmenter=augmenter, lr_min=FLAGS.lr_min, scheduler='cos',
                optimizer=optimizer, n_hypersteps_prior=FLAGS.n_hypersteps_prior,
                lr_aug_min=FLAGS.lr_aug_min, prior_prec_init=FLAGS.prior_prec_init
            )
        else:
            la, model, margliks, valid_perfs, aug_history = marglik_optimization(
                model, train_loader, marglik_loader, valid_loader, partial_loader, likelihood='classification',
                lr=FLAGS.lr, lr_hyp=FLAGS.lr_hyp, lr_aug=FLAGS.lr_aug, n_epochs=FLAGS.n_epochs,
                n_hypersteps=FLAGS.n_hypersteps, marglik_frequency=FLAGS.marglik_frequency, laplace=laplace,
                prior_structure=prior_structure, backend=backend, n_epochs_burnin=FLAGS.n_epochs_burnin,
                method=FLAGS.method, augmenter=augmenter, lr_min=FLAGS.lr_min, scheduler='cos',
                n_hypersteps_prior=FLAGS.n_hypersteps_prior, lr_aug_min=FLAGS.lr_aug_min, 
                prior_prec_init=FLAGS.prior_prec_init, optimizer=optimizer
            )

        prior_prec = la.prior_precision.mean().item()
        aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
        opt_marglik = np.min(margliks)
        logging.info(f'prior prec: {prior_prec:.2f}, aug params: {aug_params}, margLik: {opt_marglik:.2f}.')

        result['marglik'] = np.min(margliks)
        result['margliks'] = margliks

    elif FLAGS.method == 'augerino':
        train_loader.attach()
        model, losses, valid_perfs, aug_history = augerino(
            model, train_loader, valid_loader, n_epochs=FLAGS.n_epochs, lr=FLAGS.lr,
            augmenter=augmenter, aug_reg=FLAGS.augerino_reg, lr_aug=FLAGS.lr_augerino,
            lr_min=FLAGS.lr_min, scheduler='cos', optimizer=optimizer
        )
        aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
        logging.info(f'aug params: {aug_params}.')
        result['losses'] = losses

    else:
        raise ValueError(f'Invalid method {FLAGS.method}')

    if FLAGS.save:
        result_path = Path(f'results/{FLAGS.dataset}/{FLAGS.model}/')
        result['flags'] = FLAGS.flag_values_dict()
        if FLAGS.optimize_aug:
            result['aug_optimum'] = aug_params
            result['aug_history'] = aug_history
        result['valid_perfs'] = valid_perfs
        choice_type = 'optaug' if FLAGS.optimize_aug else 'fixedaug'
        if FLAGS.softplus:
            if np.abs(FLAGS.init_aug - np.log(2)) > 0.01: # init parameter to non-zero (for softplus)
                choice_type += 'softplus=' + str(FLAGS.init_aug)
            else:
                choice_type += 'softplus'
        elif np.abs(FLAGS.init_aug) > 0.01: # init parameter non-zero
            choice_type += '=' + str(FLAGS.init_aug)
        if 'kernel' in FLAGS.approx:
            approx_type = f'_approx={FLAGS.approx}-{FLAGS.marglik_batch_size}'
        else:
            approx_type = '' if FLAGS.method == 'augerino' else f'_approx={FLAGS.approx}'
        if FLAGS.method == 'augerino' and FLAGS.augerino_reg != 1e-2:  # not default
            approx_type += 'lam=' + str(FLAGS.augerino_reg)
        file_name = f'{FLAGS.method}{approx_type}_{choice_type}_E={FLAGS.n_epochs}_N={FLAGS.subset_size}_S={FLAGS.n_samples_aug}_seed={FLAGS.seed}.pkl'

        result_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / file_name, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    app.run(main)
