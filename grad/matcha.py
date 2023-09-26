# https://github.com/shivammehta25/Matcha-TTS
import torch
import torch.nn.functional as F

from grad.base import BaseModule
from grad.diffusion import GradLogPEstimator2d


class FlowMatch(BaseModule):
    def __init__(self, n_feats, dec_dim, spk_emb_dim=64, pe_scale=1000):
        super().__init__()
        self.n_feats = n_feats
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = 1e-4
        self.estimator = GradLogPEstimator2d(dec_dim, n_mels=n_feats, emb_dim=spk_emb_dim, pe_scale=pe_scale)

    @torch.no_grad()
    def forward(self, spks, mu, mask, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(spks, z, t_span=t_span, mu=mu, mask=mask)

    def solve_euler(self, spks, x, t_span, mu, mask):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator(spks, x, mask, mu, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def compute_loss(self, spks, x1, mask, mu):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(spks, y, mask, mu, t.squeeze()), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y
