import os
from pathlib import Path
import warnings
import numpy as np
import torch
from torch import nn
from math import ceil
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from laplace import FullLaplace, KronLaplace, DiagLaplace


def setup_center_grid(radius, resolution):
    """Sets up a grid [-radius, radius] x [-radius, radius] 
    with specified resolution (how many points in x and y axes).
    """
    xx, yy = np.meshgrid(np.linspace(-radius, radius, resolution), 
                         np.linspace(-radius, radius, resolution))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])
    return grid, xx, yy

    
class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500, 
                 data_factor=1, shuffle=False, detach=False):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.data_factor = data_factor
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity
        self._detach = detach

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration
        
        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        if self._detach:
            x = self.transform(self.x[start:end]).detach()
        else:
            x = self.transform(self.x[start:end])
        y = self.transform_y(self.y[start:end])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    def attach(self):
        self._detach = False
        return self
        
    def detach(self):
        self._detach = True
        return self

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)

        
class DatasetDummy:
    def __init__(self, N):
        self.N = N
        
    def __len__(self):
        return self.N


class MaxPool2dAug(nn.MaxPool2d):
    # MaxPool2d with augmentation dimension
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


def get_laplace_approximation(structure):
    if structure == 'full':
        return FullLaplace
    elif structure == 'kron':
        return KronLaplace
    elif structure == 'diag':
        return DiagLaplace
    else:

        raise ValueError('Invalid Laplace structure', structure)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = True


def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y
    

def get_data_loaders(dataset_name, subset_size, data_root, batch_size, marglik_batch_size, device):
    if dataset_name == 'MNIST':
        tforms = transforms.ToTensor()
        DS = MNIST
    elif dataset_name == 'FMNIST':
        tforms = transforms.ToTensor()
        DS = FashionMNIST
    elif dataset_name == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        DS = CIFAR10

    train_set = DS(data_root, train=True, transform=tforms)
    test_set = DS(data_root, train=False, transform=tforms)
    if subset_size != -1:  # select random subset indices
        if subset_size > len(train_set):
            warnings.warn(f'Subset size larger than data size. M={subset_size}, N={len(train_set)}')
        subset_indices = torch.randperm(len(train_set))[:subset_size]
    else:
        subset_indices = None
    x, y = dataset_to_tensors(train_set, subset_indices)
    train_loader = TensorDataLoader(x.to(device), y.to(device), batch_size=batch_size, shuffle=True)
    if marglik_batch_size > 0 and marglik_batch_size != batch_size:  # need separate marglik_loader
        marglik_loader = TensorDataLoader(x.to(device), y.to(device), batch_size=marglik_batch_size, shuffle=True)
    else:
        marglik_loader = train_loader
    x, y = dataset_to_tensors(test_set)
    test_loader = TensorDataLoader(x.to(device), y.to(device), batch_size=batch_size, shuffle=False)
    return train_loader, marglik_loader, test_loader


def save_results(dataset, model, filename, metrics):
    result_path = f'./results/{dataset}/{model}/{filename}.npy'
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(result_path, metrics)
