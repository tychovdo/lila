import os
from typing import Union, Callable 
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchvision import datasets


def get_cosine_data(n_obs=100, sigma_noise=0.3, seed=711):
    torch.manual_seed(seed)

    # create toy data set (cosine from -8 to 8)
    X_train = ((torch.rand(n_obs) - 0.5) * np.pi * 2.5 * 2).unsqueeze(-1)
    y_train = torch.cos(X_train) + torch.randn_like(X_train) * sigma_noise

    # test on the data region and further away from it
    X_test = torch.linspace(-np.pi * 3.5, np.pi * 3.5, 500).unsqueeze(-1)  # one pi extra
    return X_train, y_train, X_test


def get_circle_data(n_obs=36, degree=90, sigma_noise=0.05, seed=1):
    """ Generate dummy circle data.

    Args:
        n_obs: Total number of observations
        degree: Angle in degrees
        sigma_noise: Amount of observation noise
        seed: Integer used as random seed

    """

    torch.manual_seed(seed)

    rads = np.deg2rad(degree)

    # Divide total number of observations evenly over three rings
    n_obs_x0 = int(n_obs // 2)
    n_obs_x1 = n_obs - n_obs_x0
    n_obs_x1_outer = int(n_obs_x1 // (3 / 2))
    n_obs_x1_inner = n_obs_x1 - n_obs_x1_outer

    # Generate data points and transform into rings
    X0_t = torch.rand(n_obs_x0) * rads
    X0 = torch.stack([torch.cos(X0_t), torch.sin(X0_t)]).T * (0.5 + torch.randn(n_obs_x0, 2) * sigma_noise)

    X1_inner = torch.randn(n_obs_x1_inner, 2) * sigma_noise
    X1_outer_t = torch.rand(n_obs_x1_outer) * rads
    X1_outer = torch.stack([torch.cos(X1_outer_t), torch.sin(X1_outer_t)]).T * (1.0 + torch.randn(n_obs_x1_outer, 2) * sigma_noise)
    X1 = torch.cat((X1_inner, X1_outer))

    # Generate labels
    X = torch.cat((X0, X1))
    y = (torch.arange(0, n_obs) >= n_obs_x0).long()

    return X, y, None


def get_coscircle_data(n_obs=100, radius_max=7, rot_max_degrees=360, 
                       sample_uniform_radius=False, seed=711):
    torch.manual_seed(seed)
    rot_max_radians = rot_max_degrees / 180 * np.pi
    # first generate data in polar coordinates (r, phi)
    if sample_uniform_radius:
        # sample uniform from [0, radius_max]
        rs = torch.rand(n_obs) * radius_max
    else:
        # sample uniformly from the area using rejection sampling
        # proposal density fy is Uniform[0, radius] and we want to sample from
        # fx which is a density proportional to the perimeter for each radius r
        n_samples = n_obs * 1000  # enough w.h.p.
        proposal_samples = torch.rand(n_samples) * radius_max
        acceptance_threshold = torch.rand(n_samples)
        fy = 2 * proposal_samples / (radius_max ** 2)  # perimeter-proportional
        fx = 1 / radius_max  # uniform
        accepted = (acceptance_threshold < (fy / fx)) 
        rs = proposal_samples[accepted][:n_obs]
    phis = torch.rand(n_obs) * rot_max_radians
    # labels depend only on the radius with y ~ Bern(0.5 * (cos(r) + 1))
    y_dist = Bernoulli(probs=0.5 * (torch.cos(rs) + 1))
    y_train = y_dist.sample()
    X_train = torch.stack([rs * torch.cos(phis), rs * torch.sin(phis)], dim=1)
    return X_train, y_train.long()


class RotatedMNIST(datasets.MNIST):
    """ Rotated MNIST class.
        Wraps regular pytorch MNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedMNIST(datasets.MNIST):
    """ MNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledMNIST(datasets.MNIST):
    """ MNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated FashionMNIST class.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 rotated by fixed amount using random seed """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class TranslatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class ScaledCIFAR10(datasets.CIFAR10):
    """ CIFAR10 scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)

