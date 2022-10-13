import torch
from absl import logging
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim.adam import Adam
import torch.nn.functional as F

from lila.marglik import get_model_optimizer, get_scheduler, valid_performance


def augerino(model,
             train_loader,
             valid_loader=None,
             likelihood='classification',
             weight_decay=1e-4,
             aug_reg=0.01,
             n_epochs=500,
             lr=1e-3,
             lr_min=None,
             lr_aug=1e-3,
             optimizer='Adam',
             scheduler='exp',
             augmenter=None):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    optimize_aug = (parameters_to_vector(augmenter.parameters()).requires_grad) and (aug_reg != 0.0)
    # set up augmentation optimizer
    if optimize_aug:
        logging.info('AUGERINO: optimize augmentation')
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug, weight_decay=0)
        aug_history = list()

    if likelihood == 'classification':
        criterion = CrossEntropyLoss()
    elif likelihood == 'regression':
        criterion = MSELoss()
    else:
        raise ValueError(f'Invalid likelihood: {likelihood}')

    losses = list()
    valid_perfs = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        for X, y in train_loader:
            optimizer.zero_grad()
            if optimize_aug:
                aug_optimizer.zero_grad()

            f = model(X).mean(dim=1)
            loss = criterion(f, y)
            if optimize_aug:
                if hasattr(augmenter, 'softplus') and augmenter.softplus:
                    loss -= aug_reg * F.softplus(parameters_to_vector(augmenter.parameters())).norm()
                else:
                    loss -= aug_reg * parameters_to_vector(augmenter.parameters()).norm()
            loss.backward()

            optimizer.step()
            if optimize_aug:
                aug_optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        if optimize_aug:
            aug_history.append([parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()])
            logging.info(f'AUGERINO[epoch={epoch}]: augmentation params {aug_history[-1]}.%')
        losses.append(epoch_loss)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method='avgfunc', device=device)
                valid_perfs.append(valid_perf)
                logging.info(f'AUGERINO[epoch={epoch}]: validation performance {valid_perf*100:.2f}.%')

    if optimize_aug:
        return model, losses, valid_perfs, aug_history
    return model, losses, valid_perfs, None
