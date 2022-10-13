from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from absl import app
from absl import flags
import torch
import pickle
import logging

from laplace.curvature.asdl import AsdlEF, AsdlGGN
from laplace.curvature.augmented_asdl import AugAsdlGGN, AugAsdlEF
from laplace.baselaplace import FullLaplace
from torch.nn.utils.convert_parameters import parameters_to_vector

from lila.augerino import augerino
from lila.marglik import marglik_optimization, marglik_opt_jvecprod
from lila.datasets import get_circle_data
from lila.utils import TensorDataLoader, setup_center_grid, get_laplace_approximation
from lila.layers import RotationLayer


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 711, 'Random seed for data generation and model initialization.')
flags.DEFINE_enum('method', 'baseline', ['baseline', 'hardcoded', 'avgloglik', 'avgfunc', 'augmentation', 'augerino'],
                  'Available methods: `baseline` is no augmentation, `hardcoded` is fixed invariance,'
                  + '`avgloglik` is averaging log liks, `avgfunc` is averaging functions')
flags.DEFINE_bool('use_jvp', False, 'whether to use JVP instead of sub-add trick')
flags.DEFINE_float('augerino_reg', 1e-2, 'l2 regularization of augerino')
flags.DEFINE_enum('approximation_structure', 'full', ['full', 'kron', 'diag'],
                  'Structure of the laplace approximation.')
flags.DEFINE_integer('batch_size', -1, 'Batch size of the train loader (-1 is full batch)')
flags.DEFINE_integer('marglik_batch_size', -1, 'Batch size of the marglik data loader (-1 is full batch)')
flags.DEFINE_enum('curvature_type', 'ggn', ['ggn', 'ef'], 'Type of curvature estimate.')
flags.DEFINE_integer('n_epochs', 2000, 'Number of epochs')
flags.DEFINE_integer('n_obs', 100, 'Factor for `n_obs`, resulting `n_obs` is `n_obs_factor*6')
flags.DEFINE_integer('rotation_max', 120, 'Maximum rotation in polar coordinates',
                     lower_bound=0, upper_bound=360)
flags.DEFINE_float('sigma_noise', 0.07, 'noise on circles')
flags.DEFINE_integer('n_samples_aug', 11, 'number of augmentation samples if applicable')
flags.DEFINE_integer('rotation_init', 90, 'Initial rotation for the augmenter',
                     lower_bound=0, upper_bound=180)
flags.DEFINE_bool('stochastic_aug', True, 'Whether to sample for augmenter')
flags.DEFINE_bool('optimize_aug', False, 'Whether to differentiably optimize augmenter (for avgfunc)')
flags.DEFINE_bool('plot', True, 'Whether to produce plot')
flags.DEFINE_bool('save', False, 'Whether to save the experiment outcome as pickle')
flags.DEFINE_enum('device', 'cpu', ['cpu', 'cuda'], 'Device to run on')
flags.DEFINE_bool('posterior_predictive', True, 'Whether to use posterior predictive, otherwise MAP.')
flags.DEFINE_float('lr', 0.1, 'parameter learning rate')
flags.DEFINE_float('lr_min', 0.1, 'parameter learning rate')
flags.DEFINE_float('lr_hyp', 0.1, 'hyper parameter learning rate')
flags.DEFINE_float('lr_aug', 0.1, 'aug parameter learning rate')
flags.DEFINE_float('lr_aug_min', 0.1, 'aug parameter learning rate')


def main(argv):
    torch.manual_seed(FLAGS.seed)
    X_train_raw, y_train, _ = get_circle_data(n_obs=FLAGS.n_obs, degree=FLAGS.rotation_max, seed=FLAGS.seed,
                                              sigma_noise=FLAGS.sigma_noise)
    radius_max = torch.sqrt(X_train_raw.square().sum(dim=-1)).max()
    # TODO: resolution was 300 for HQ plots
    X_test, xx, yy = setup_center_grid(radius_max * 1.2, resolution=100)
    X_test = X_test.to(X_train_raw.dtype)
    X_train_raw = X_train_raw.to(FLAGS.device)
    y_train = y_train.to(FLAGS.device)
    X_test = X_test.to(FLAGS.device)

    # depending on the method, the data are augmented in different ways except for `baseline`
    temperature = 1
    data_factor = 1
    backend = AsdlGGN if FLAGS.curvature_type == 'ggn' else AsdlEF
    X_train = X_train_raw


    augmenter = None
    transform_y = None
    if FLAGS.method == 'hardcoded':
        # convert to polar coordinates
        X_train = torch.stack([torch.sqrt(X_train_raw.square().sum(dim=-1)),
                               torch.atan2(X_train[:, 1], X_train[:, 0])], dim=1)
        X_test = torch.stack([torch.sqrt(X_test.square().sum(dim=-1)),
                              torch.atan2(X_test[:, 1], X_test[:, 0])], dim=1)
    elif FLAGS.method == 'avgloglik':
        temperature = data_factor = FLAGS.n_samples_aug
        rotation = RotationLayer(n_samples=FLAGS.n_samples_aug, rot_factor=FLAGS.rotation_init/180,
                                 deterministic=not FLAGS.stochastic_aug, independent=FLAGS.stochastic_aug)
        augmenter = torch.nn.Sequential(rotation, torch.nn.Flatten(start_dim=0, end_dim=1))
        transform_y = lambda y: y.repeat_interleave(FLAGS.n_samples_aug)
        rotation.rot_factor.requires_grad = FLAGS.optimize_aug
    elif FLAGS.method == 'augmentation':
        rotation = RotationLayer(n_samples=1, rot_factor=FLAGS.rotation_init/180, deterministic=False,
                                 independent=True)
        augmenter = torch.nn.Sequential(rotation, torch.nn.Flatten(start_dim=0, end_dim=1))
        rotation.rot_factor.requires_grad = FLAGS.optimize_aug
    elif FLAGS.method == 'avgfunc' or FLAGS.method == 'augerino':
        augmenter = RotationLayer(n_samples=FLAGS.n_samples_aug, rot_factor=FLAGS.rotation_init/180,
                                  deterministic=not FLAGS.stochastic_aug, independent=FLAGS.stochastic_aug)
        augmenter.rot_factor.requires_grad = FLAGS.optimize_aug
        backend = AugAsdlGGN if FLAGS.curvature_type == 'ggn' else AugAsdlEF

    if augmenter is not None:
        augmenter = augmenter.to(FLAGS.device)
    batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else len(X_train)
    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, transform_y=transform_y,
                                    data_factor=data_factor, batch_size=batch_size, shuffle=True,
                                    detach=True)
    marglik_batch_size = FLAGS.marglik_batch_size if FLAGS.marglik_batch_size > 0 else len(X_train)
    marglik_loader = TensorDataLoader(X_train, y_train, transform=augmenter, transform_y=transform_y,
                                      data_factor=data_factor, batch_size=marglik_batch_size, shuffle=True,
                                      detach=False)

    # model
    model = torch.nn.Sequential(torch.nn.Linear(2, 50),
                                torch.nn.Tanh(),
                                torch.nn.Linear(50, 2)).to(FLAGS.device)

    start = timer()

    if FLAGS.method != 'augerino':
        laplace = get_laplace_approximation(FLAGS.approximation_structure)
        # marglik based methods
        if FLAGS.use_jvp:
            la, model, margliks, _, aug_history = marglik_opt_jvecprod(
                model, train_loader, marglik_loader, likelihood='classification', 
                prior_structure='scalar', n_epochs=FLAGS.n_epochs, lr=FLAGS.lr, 
                n_hypersteps=1, marglik_frequency=1, lr_min=FLAGS.lr_min,
                lr_hyp=FLAGS.lr_hyp, lr_aug=FLAGS.lr_aug, laplace=laplace, backend=backend, temperature=temperature, 
                method=FLAGS.method, augmenter=augmenter, lr_aug_min=FLAGS.lr_aug_min
            )
        else:
            la, model, margliks, _, aug_history = marglik_optimization(
                model, train_loader, marglik_loader, likelihood='classification', 
                prior_structure='scalar', n_epochs=FLAGS.n_epochs, lr=FLAGS.lr, 
                n_hypersteps=1, marglik_frequency=1, lr_min=FLAGS.lr_min,
                lr_hyp=FLAGS.lr_hyp, lr_aug=FLAGS.lr_aug, laplace=laplace, backend=backend, temperature=temperature, 
                method=FLAGS.method, augmenter=augmenter, lr_aug_min=FLAGS.lr_aug_min
            )
        end = timer()
        logging.info(f'Final LML={la.log_marginal_likelihood()} after {end - start} sec.')
        if augmenter is not None:
            aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
            prec_mean = la.prior_precision.cpu().detach().numpy().mean()
            logging.info(f'Final aug params: {aug_params*180}, prior prec mean: {prec_mean}.')

        # for comparison purposes, add full marglik
        la_full = FullLaplace(model, 'classification', prior_precision=la.prior_precision,
                              backend=backend)
        la_full.fit(train_loader)
        margliks.append(-la_full.log_marginal_likelihood().item())
    
    elif FLAGS.method == 'augerino':
        model, losses, valid_perfs, aug_history = augerino(
            model, train_loader.attach(), n_epochs=FLAGS.n_epochs, lr=FLAGS.lr, 
            augmenter=augmenter, aug_reg=FLAGS.augerino_reg, lr_aug=FLAGS.lr_aug
        )
        la = FullLaplace(model, 'classification', prior_precision=1e-4*len(X_train), backend=AugAsdlGGN)
        la.fit(train_loader)
        end = timer()
        logging.info(f'Augerino finished after {end - start} sec.')
        aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
        logging.info(f'Final aug params: {aug_params*180}.')
        plt.plot(losses)
        plt.show()

    if FLAGS.method == 'avgfunc' or FLAGS.method == 'augerino':
        augmenter.independent = False  # smooth predictive
        if FLAGS.posterior_predictive:
            pred_test = la(augmenter(X_test).detach(), link_approx='mcparam', n_samples=1000)[:, 1].detach().cpu().numpy()
        else:
            pred_test = torch.softmax(model(augmenter(X_test)).mean(dim=1), dim=-1)[:, 1].detach().cpu().numpy()
    else:
        if FLAGS.posterior_predictive:
            pred_test = la(X_test, link_approx='mcparam', n_samples=1000)[:, 1].detach().cpu().numpy()
        else:
            pred_test = torch.softmax(model(X_test), dim=-1)[:, 1].detach().cpu().numpy()
    zz = pred_test.reshape(xx.shape)

    x_train, y_train = X_train_raw.cpu().numpy(), y_train.cpu().numpy()
    if FLAGS.save:
        result = dict(xx=xx, yy=yy, zz=zz, x_train=x_train, y_train=y_train, marglik=np.min(margliks))
        if FLAGS.optimize_aug:
            if FLAGS.method != 'augerino':
                result['margliks'] = margliks
            result['aug_history'] = aug_history
            result['aug_optimum'] = parameters_to_vector(augmenter.parameters()).squeeze().cpu().detach().item()
        choice_type = 'differentiate' if FLAGS.optimize_aug else 'discrete'
        if FLAGS.stochastic_aug:
            choice_type += '_stoch'
        method = FLAGS.method + ('jvp' if FLAGS.use_jvp else '')
        file_name = f'results/toy_classification/{method}_{choice_type}_truerot={FLAGS.rotation_max}_initrot{FLAGS.rotation_init}_N={FLAGS.n_obs}_seed={FLAGS.seed}_{FLAGS.approximation_structure}_{FLAGS.curvature_type}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(result, f)

    if not FLAGS.plot:
        return

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.contourf(xx, yy, zz, alpha=0.7, cmap='RdBu', levels=21, antialiased=True)#, vmin=0, vmax=1)
    ax.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='tab:red', s=75, lw=1, edgecolors='black', marker='o')
    ax.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='tab:blue', s=75, lw=1, edgecolors='black', marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Save plot
    Path('plots').mkdir(exist_ok=True)
    plt.show()


if __name__ == '__main__':
    app.run(main)
