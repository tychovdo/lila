import torch
import numpy as np


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
