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
