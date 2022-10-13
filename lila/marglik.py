# Taken from https://github.com/AlexImmer/Laplace/blob/main/laplace/marglik_training.py
# and modified for differentiable learning of invariances/augmentation strategies.
from copy import deepcopy
import logging
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector

from laplace import KronLaplace
from laplace.curvature import AsdlGGN


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec


def valid_performance(model, test_loader, likelihood, method, device):
    N = len(test_loader.dataset)
    perf = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        if method in ['avgfunc', 'augerino']:
            f = model(X).mean(dim=1)
        else:
            f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        else:
            perf += (f - y).square().sum() / N
    return perf.item()


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])


def marglik_optimization(model,
                         train_loader,
                         marglik_loader=None,
                         valid_loader=None,
                         partial_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='exp',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         n_hypersteps_prior=1,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_aug=1e-2,
                         lr_aug_min=1e-2,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         differentiable=True,
                         method='baseline',
                         augmenter=None):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    method : augmentation strategy, one of ['baseline', 'hardcoded'] -> no change
        or ['avgfunc', 'avgloglik'] -> change in protocol.
    augmenter : torch.nn.Module with differentiable parameter


    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    if marglik_loader is None:
        marglik_loader = train_loader
    if partial_loader is None:
        partial_loader = marglik_loader
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    backend_kwargs = dict(differentiable=differentiable)

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1.
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    optimize_aug = augmenter is not None and parameters_to_vector(augmenter.parameters()).requires_grad
    if optimize_aug:
        logging.info('MARGLIK: optimize augmentation.')
        n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * (1 if optimize_aug else n_hypersteps)
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug)
        aug_scheduler = CosineAnnealingLR(aug_optimizer, n_steps, eta_min=lr_aug_min)
        aug_history = [parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()]

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    losses = list()
    valid_perfs = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector(model.parameters())
            delta = expand_prior_precision(prior_prec, model)
            if method == 'avgfunc':
                f = model(X).mean(dim=1)
            else:
                f = model(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()

            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        logging.info(f'MARGLIK[epoch={epoch}]: training performance {epoch_perf*100:.2f}%.')
        gb_factor = 1024 ** 3
        logging.info('Max memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/gb_factor) + ' Gb.')
        optimizer.zero_grad(set_to_none=True)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method, device)
                valid_perfs.append(valid_perf)
                logging.info(f'MARGLIK[epoch={epoch}]: validation performance {valid_perf*100:.2f}%.')

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        torch.cuda.empty_cache()
        sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                      temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)

        if optimize_aug:
            # fit without grad and make data iterator to draw from for partial fits
            lap.fit(marglik_loader, keep_factors=True)
            torch.cuda.empty_cache()
            partial_iterator = iter(partial_loader)
            X, y = next(partial_iterator)
            lap.fit_partial(X, y)
        else:
            lap.fit(marglik_loader)
            torch.cuda.empty_cache()

        if optimize_aug:
            aug_grad = torch.zeros_like(parameters_to_vector(augmenter.parameters()))

        # 2. differentiate wrt. hyperparameters for n_hypersteps
        for i in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if optimize_aug:
                aug_optimizer.zero_grad()
            if likelihood == 'classification':
                sigma_noise = None
            elif likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise)
            torch.cuda.empty_cache()
            marglik.backward()
            torch.cuda.empty_cache()

            if optimize_aug and torch.any(torch.isnan(gradient_to_vector(augmenter.parameters()))):
                for param in augmenter.parameters():
                    param.grad = torch.nan_to_num(param.grad)

            if i <= (n_hypersteps_prior - 1) or not optimize_aug:
                # when optimizing aug, the hypersteps are only for it except for the first
                hyper_optimizer.step()

            if optimize_aug:
                aug_grad = (aug_grad + gradient_to_vector(augmenter.parameters()).data.clone())
                torch.cuda.empty_cache()
                if (i < n_hypersteps - 1):
                    try:
                        X, y = next(partial_iterator)
                    except StopIteration:
                        break
                    lap.fit_partial(X, y)

        if optimize_aug:  # take step after n_hypersteps accumulation
            for p in augmenter.parameters():
                p.grad.data = aug_grad
            aug_optimizer.step()
            aug_scheduler.step()
            aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())
            logging.info(f'Augmentation params epoch {epoch}: {aug_history[-1]}')

        margliks.append(marglik.item())
        del lap

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            model.cpu()
            best_model_dict = deepcopy(model.state_dict())
            model.to(device)
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]
            if optimize_aug:
                best_augmenter = deepcopy(augmenter.state_dict())
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}, prec: {best_precision.mean().item():.2f}. '
                         + 'Saving new best model.')
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}, prec: {prior_prec.mean().item():.2f}. '
                         + f'No improvement over {best_marglik:.2f}')

    logging.info('MARGLIK: finished training. Recover best model and fit Lapras.')
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
        if optimize_aug:
            augmenter.load_state_dict(best_augmenter)
    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
    lap.fit(marglik_loader)
    if optimize_aug:
        return lap, model, margliks, valid_perfs, aug_history
    return lap, model, margliks, valid_perfs, None


def marglik_opt_jvecprod(model,
                         train_loader,
                         marglik_loader=None,
                         valid_loader=None,
                         partial_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='exp',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         n_hypersteps_prior=1,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_aug=1e-2,
                         lr_aug_min=1e-2,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         method='baseline',
                         augmenter=None):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    method : augmentation strategy, one of ['baseline', 'hardcoded'] -> no change
        or ['avgfunc', 'avgloglik'] -> change in protocol.
    augmenter : torch.nn.Module with differentiable parameter


    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    if marglik_loader is None:
        marglik_loader = train_loader
    if partial_loader is None:
        partial_loader = marglik_loader
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    backend_kwargs = dict(differentiable=False)  # don't need differentiable laplace, only backend

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1.
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    optimize_aug = augmenter is not None and parameters_to_vector(augmenter.parameters()).requires_grad
    if optimize_aug:
        logging.info('MARGLIK: optimize augmentation.')
        n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency)
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug)
        aug_scheduler = CosineAnnealingLR(aug_optimizer, n_steps, eta_min=lr_aug_min)
        aug_history = [parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()]

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    losses = list()
    valid_perfs = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector(model.parameters())
            delta = expand_prior_precision(prior_prec, model)
            if method == 'avgfunc':
                f = model(X).mean(dim=1)
            else:
                f = model(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()

            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        logging.info(f'MARGLIK[epoch={epoch}]: training performance {epoch_perf*100:.2f}%.')
        gb_factor = 1024 ** 3
        logging.info('MAP memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/gb_factor) + ' Gb.')
        optimizer.zero_grad(set_to_none=True)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method, device)
                valid_perfs.append(valid_perf)
                logging.info(f'MARGLIK[epoch={epoch}]: validation performance {valid_perf*100:.2f}%.')

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        torch.cuda.empty_cache()
        sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                      temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
        marglik_loader.detach()
        lap.fit(marglik_loader, keep_factors=False)
        logging.info('LA fit memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/gb_factor) + ' Gb.')
        # first optimize prior precision
        for i in range(n_hypersteps_prior):
            hyper_optimizer.zero_grad()
            if likelihood == 'classification':
                sigma_noise = None
            elif likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()

        torch.cuda.empty_cache()
        # then differentiate and aggregate aug_grad
        if optimize_aug:
            partial_loader.attach()
            aug_grad = torch.zeros_like(parameters_to_vector(augmenter.parameters()))
            lap.backend.differentiable = True
            N = len(train_loader.dataset)
            if isinstance(lap, KronLaplace):
                # does the inversion internally
                hess_inv = lap.posterior_precision.jvp_logdet()
            else:
                hess_inv = lap.posterior_covariance.flatten()
            for i, (X, y) in zip(range(n_hypersteps), partial_loader):
                lap.loss, H_batch = lap._curv_closure(X, y, N)
                # curv closure creates gradient already, need to zero
                aug_optimizer.zero_grad()
                # compute grad wrt. neg. log-lik
                (- lap.log_likelihood).backward(inputs=list(augmenter.parameters()), retain_graph=True)
                # compute grad wrt. log det = 0.5 vec(P_inv) @ (grad-vec H)
                (0.5 * H_batch.flatten()).backward(gradient=hess_inv, inputs=list(augmenter.parameters()))
                aug_grad = (aug_grad + gradient_to_vector(augmenter.parameters()).data.clone())

            lap.backend.differentiable = False

            for p in augmenter.parameters():
                p.grad.data = aug_grad
            aug_optimizer.step()
            aug_scheduler.step()
            aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())
            logging.info(f'Augmentation params epoch {epoch}: {aug_history[-1]}')

        logging.info('JVP memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/gb_factor) + ' Gb.')

        margliks.append(marglik.item())
        del lap

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            model.cpu()
            best_model_dict = deepcopy(model.state_dict())
            model.to(device)
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]
            if optimize_aug:
                best_augmenter = deepcopy(augmenter.state_dict())
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}, prec: {best_precision.mean().item():.2f}. '
                         + 'Saving new best model.')
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}, prec: {prior_prec.mean().item():.2f}. '
                         + f'No improvement over {best_marglik:.2f}')

    logging.info('MARGLIK: finished training. Recover best model and fit Lapras.')
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
        if optimize_aug:
            augmenter.load_state_dict(best_augmenter)
    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
    lap.fit(marglik_loader)
    if optimize_aug:
        return lap, model, margliks, valid_perfs, aug_history
    return lap, model, margliks, valid_perfs, None
