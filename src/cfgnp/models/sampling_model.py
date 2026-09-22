import torch
from torch import nn
import torch.nn.functional as F


class CFSamplingModel(nn.Module):
    def __init__(self, gnn_model, mc_samples=100, temperature=0.5, eps=1e-8):
        super().__init__()
        self.gnn_model = gnn_model
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.eps = eps

    def forward(self, input, deterministic=False):
        mog_output = self.gnn_model(input)
        return self.point_estimate(mog_output, self.mc_samples, self.temperature, self.eps, deterministic)

    @staticmethod
    def point_estimate(mog_output, mc_samples=100, temperature=0.5, eps=1e-8, deterministic=False):
        means = mog_output[..., 0]   # (..., K)
        vars = mog_output[..., 1]    # (..., K)
        weights = mog_output[..., 2]  # (..., K)

        # `mog_output`'s variance channel already went through a Softplus in MoGLayer
        vars = vars + eps
        weights = weights.clamp_min(eps)

        if deterministic:
            return (weights * means).sum(dim=-1)

        device = weights.device
        dtype = weights.dtype
        sample_shape = (mc_samples,) + tuple(weights.shape)

        # Differentiable categorical relaxation via Gumbel-Softmax
        u = torch.rand(
            sample_shape,
            device=device,
            dtype=torch.float32,
        ).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))

        relaxed_one_hot = F.softmax(
            (weights.unsqueeze(0).log() + gumbel) / temperature,
            dim=-1
        )

        # Reparameterized Gaussian samples for each component
        z = torch.randn(sample_shape, device=device, dtype=dtype)
        component_samples = means.unsqueeze(0) + torch.sqrt(vars.unsqueeze(0)) * z

        # Weighted mixture sample, differentiable w.r.t. weights, means, and vars
        samples = (relaxed_one_hot * component_samples).sum(dim=-1)

        # Monte Carlo average
        return samples.mean(dim=0)