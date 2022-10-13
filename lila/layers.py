import torch
import torch.nn.functional as F
import numpy as np
from math import pi


class AffineLayer2d(torch.nn.Module):
    """ Layer that creates affine transformed versions of 2d image inputs sampled between U(-parameter, +parameter),
        with single scalar parameter.

        Args:
            rot_factor: Initial rotation factor (0 = no invariance, 1 = full circle ±180 degrees)
            deterministic: If true, we replace the stochastic reparameterization trick by scaling fixed linspaced samples
            n_samples: Amount of samples to use.
            independent: Independently sample datapoints within batch (otherwise the n_samples samples are joinedly sampled within the batch for speed-up)

     """
    def __init__(self, n_samples=32, init_value=0.0, softplus=False, softcap_rotation=False):
        super().__init__()

        self.n_samples = n_samples
        if softplus:
            assert init_value > 0.0, 'Cannot initialize to 0 with softplus, would be -inf.'
            init_value = np.log(np.exp(init_value) - 1)
        self.rot_factor = torch.nn.Parameter(torch.full((6,), init_value))
        self.softplus = softplus
        self.softcap_rotation = softcap_rotation

        # Define generators for the Lie group
        G1 = torch.tensor([0, 0, 1,
                           0, 0, 0,
                           0, 0, 0]).reshape(1, 1, 3, 3)  # x-translation
        G2 = torch.tensor([0, 0, 0,
                           0, 0, 1,
                           0, 0, 0]).reshape(1, 1, 3, 3)  # y-translation
        G3 = torch.tensor([0, -1, 0,
                           1, 0, 0,
                           0, 0, 0]).reshape(1, 1, 3, 3) # rotation
        G4 = torch.tensor([1, 0, 0,
                           0, 0, 0,
                           0, 0, 0]).reshape(1, 1, 3, 3)  # x-scale
        G5 = torch.tensor([0, 0, 0,
                           0, 1, 0,
                           0, 0, 0]).reshape(1, 1, 3, 3) # y-scale
        G6 = torch.tensor([0, 1, 0,
                           1, 0, 0,
                           0, 0, 0]).reshape(1, 1, 3, 3)  # shearing

        self.register_buffer('G1', G1)
        self.register_buffer('G2', G2)
        self.register_buffer('G3', G3)
        self.register_buffer('G4', G4)
        self.register_buffer('G5', G5)
        self.register_buffer('G6', G6)


    def forward(self, x, align_corners=True, mode='bilinear'):
        """Connects to the next available port.

        Args:
            x: Input tensor with dimensions (B, C, H, W)
            align_corners: Uses align_corners convention.
            mode: Type of interpolation. (nearest|bilinear)

        Returns:
            Rotated input 'n_samples' times with rotations uniformly sampled between -rot_factor*pi and +rot_factor*pi rads.
            Output dimension: (B, n_samples, C, H, W)

        """

        # Obtain sample values
        device = x.device

        # Build resampling grids
        N, C, H, W = x.shape
        k = 6

        # Independently sample random points from U[-1, 1]^k
        k_samples = [torch.rand(N, self.n_samples, device=device).unsqueeze(2).unsqueeze(3) * 2 - 1 for _ in range(k)]

        rot_factor = self.rot_factor if not self.softplus else F.softplus(self.rot_factor)

        # Cap rotation. Optionally soft, always hard
        rot_id = 2
        if self.softcap_rotation:
            first_part = rot_factor[:rot_id]
            rot_part = rot_factor[rot_id:rot_id+1]
            last_part = rot_factor[rot_id+1:]
            rot_factor = torch.cat((first_part, pi * torch.tanh(rot_part), last_part))

        M = (k_samples[0] * rot_factor[0] * self.G1 + \
             k_samples[1] * rot_factor[1] * self.G2 + \
             k_samples[2] * rot_factor[2].clamp(-pi, pi) * self.G3 + \
             k_samples[3] * rot_factor[3] * self.G4 + \
             k_samples[4] * rot_factor[4] * self.G5 + \
             k_samples[5] * rot_factor[5] * self.G6)

        # Exponentiate from Lie algebra to Lie group
        matrices_batch = torch.matrix_exp(M.view(N*self.n_samples, 3, 3)).view(N, self.n_samples, 3, 3)[:, :, :2, :]


        # Evaluate corresponding vector field on pixel locations
        out_shape = (N * self.n_samples, C, H, W)

        matrices_batch = matrices_batch.view(N * self.n_samples, 2, 3)

        grids_batch = F.affine_grid(matrices_batch, out_shape, align_corners=align_corners)

        out = F.grid_sample(x.unsqueeze(1).expand(N, self.n_samples, C, H, W).contiguous().view(N*self.n_samples, C, H, W), grids_batch, align_corners=align_corners, mode=mode)
        out = out.view(N, self.n_samples, C, H, W)

        return out




class RotationLayer(torch.nn.Module):
    """ Layer that creates rotated versions of input sampled between U[-parameter, +parameter], 
        with single scalar parameter. Used for 1d data in dummy to problems. For 2d images, use AffineLayer2d.
        
        Args:
            rot_factor: Initial rotation factor (0 = no invariance, 1 = full circle ±180 degrees)
            deterministic: If true, we replace the stochastic reparameterization trick by scaling fixed linspaced samples
            n_samples: Amount of samples to use.
            independent: Independently sample datapoints within batch (otherwise the n_samples samples are joinedly sampled within the batch for speed-up)
            
     """
    def __init__(self, rot_factor=0.0, deterministic=False, n_samples=32, independent=False):
        super().__init__()
        
        self.deterministic = deterministic
        self.n_samples = n_samples
        self.independent = independent
        assert not (independent and self.deterministic), f"Independent sampling is only possible when sampling non deterministically."

        self.rot_factor = torch.nn.Parameter(torch.full((1,), rot_factor))
        
        if deterministic:
            # Distribute equally over circle in such way that endpoints don't overlap when fully invariant;
            if n_samples % 2 == 0:
                # even
                start = -(n_samples // 2 - 1) / (n_samples // 2)
                end = 1
            else:
                # uneven
                start = -(1 - 1 / n_samples)
                end = (1 - 1 / n_samples)
            self.register_buffer('_linspace', torch.linspace(start, end, n_samples), persistent=False)
            

    def forward(self, x):
        """Connects to the next available port.

        Args:
          x: Input tensor with dimensions (B, 2)

        Returns:
          Rotated input 'n_samples' times with rotations uniformly sampled between -rot_factor*pi and +rot_factor*pi rads.
          Output dimension: (B, n_samples, 2)
        
        """
        
        # Obtain sample values
        if self.deterministic:
            samples = self._linspace
        else:
            device = x.device

            if self.independent:
                batch_size = len(x)
                samples = torch.rand((batch_size, self.n_samples), device=device) * 2 - 1
            else:
                samples = torch.rand(self.n_samples, device=device) * 2 - 1

        # Build rotation matrices
        rads = samples * np.pi * self.rot_factor

        c, s = torch.cos(rads), torch.sin(rads)

        if self.independent:
            matrices = torch.stack((torch.stack((c, -s), 0), torch.stack((s, c), 0)), 0)
            matrices = matrices.permute((2, 3, 0, 1)) # (batch_size, n_samples, 2, 2)

            # Apply to input
            return torch.einsum('bcji,bi->bcj', matrices, x)
        else:
            matrices = torch.stack((torch.stack((c, -s), 0), torch.stack((s, c), 0)), 0)
            matrices = matrices.permute((2, 0, 1)) # (n_samples, 2, 2)
        
            # Apply to input
            return torch.einsum('cji,bi->bcj', matrices, x)
