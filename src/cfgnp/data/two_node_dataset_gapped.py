"""Two-node counterfactual identifiability dataset, with ``alpha`` bounded away from ``x0``.

A separate experiment from ``two_node_dataset.py`` -- not a modification of it -- because the
fix below changes which population of ``(x0, alpha)`` queries the model ever trains or is
evaluated on, and is best kept comparable/reproducible against the original by never sharing a
dataset name, checkpoint, or cache with it.

The problem this works around: sampling ``alpha`` independently and uniformly from
``x0`` occasionally lands ``alpha`` close to ``x0`` in *arctan-angle* space, i.e.
``|phi(alpha) - phi(x0)|`` near 0 in ``cross_world_gain``'s notation. That collapses the true
reference's cross-world variance ``tau2`` towards 0 -- a near-Dirac target. A finite model
essentially never matches a near-Dirac target exactly, and because ``KL(p_ref || p_model)`` is
heavily penalised whenever the model is far wider than a tight reference, these queries dominate
the upper quantiles of the reported KL divergence, even though they cover a non-trivial (~7% in
practice) fraction of queries, not a measure-zero edge case.

The fix mirrors the existing precedent for exactly this failure mode: ``DEFAULT_LAM_RANGE`` in
``two_node_dataset.py`` is already bounded away from 0 "so the true-mechanism counterfactual
never degenerates to a Dirac". This module applies the same idea to ``alpha`` sampling, which the
original dataset does not do (see ``MIN_ALPHA_ANGLE_GAP``).

Everything else -- ``TwoNodePrior``, ``TwoNodeSCM``, ``cross_world_gain``, the closed-form
``true_counterfactual``/``bma_counterfactual`` references -- is reused unchanged from
``two_node_dataset.py``: this only changes how ``alpha`` is sampled during data generation, not
the SCM family or the reference math.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

from cfgnp.data.two_node_dataset import TwoNodePrior

#: Minimum ``|phi(alpha) - phi(x0)|`` enforced between the factual ``x0`` and the intervention
#: ``alpha`` (see ``cross_world_gain``'s ``dphi`` in two_node_dataset.py). Same margin as
#: ``DEFAULT_LAM_RANGE``'s lower bound, for the same reason.
MIN_ALPHA_ANGLE_GAP = 0.15


def _sample_alpha_away_from_x0(x0: np.ndarray, alpha_range: Tuple[float, float], sigma_u: float,
                               sigma_v: float, rng: np.random.Generator,
                               min_angle_gap: float = MIN_ALPHA_ANGLE_GAP,
                               max_rounds: int = 64) -> np.ndarray:
    """Draw ``alpha ~ U(alpha_range)`` per task, rejecting draws with ``|phi(alpha)-phi(x0)|``
    below ``min_angle_gap``. Vectorised: only the still-rejected entries are redrawn each round,
    so this costs one extra pass for the ~7% of draws that land too close, not a full resample.
    """
    lo, hi = alpha_range
    phi_x0 = np.arctan(sigma_v * x0 / sigma_u)
    alpha = rng.uniform(lo, hi, size=x0.shape)
    for _ in range(max_rounds):
        phi_alpha = np.arctan(sigma_v * alpha / sigma_u)
        bad = np.abs(phi_alpha - phi_x0) <= min_angle_gap
        if not bad.any():
            break
        alpha[bad] = rng.uniform(lo, hi, size=int(bad.sum()))
    return alpha


def create_dataset_two_node_gapped(n_tasks: int, n_obs: int = 100, variant: str = "ID", seed: int = 42,
                                   prior: Optional[TwoNodePrior] = None,
                                   cache_dir: Optional[str] = None,
                                   min_angle_gap: float = MIN_ALPHA_ANGLE_GAP):
    """``create_dataset_two_node``, but ``alpha`` is rejection-sampled away from ``x0``.

    Same SCM family, same sample tuple layout, same ``scm_params`` -- see
    ``two_node_dataset.create_dataset_two_node`` for the full description. The only difference is
    the ``alpha`` draw, so this is a near-verbatim copy rather than a wrapper: keeping the
    vectorised generation self-contained here avoids threading a gap parameter through the
    original function's call sites.
    """
    prior = prior or TwoNodePrior(variant)
    cache_path = None
    if cache_dir is not None:
        cache_path = os.path.join(
            cache_dir, f"two_node_gapped_{prior.variant}_n{n_tasks}_obs{n_obs}_seed{seed}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

    rng = np.random.default_rng(seed)
    a = prior.sample_weight(n_tasks, rng)
    lam = prior.sample_lambda(n_tasks, rng)
    su, sv = prior.sigma_u, prior.sigma_v

    def loading(x, lam_):
        s = np.sqrt(su ** 2 + (sv * x) ** 2)
        d = lam_ * np.arctan(sv * x / su)
        return np.stack([s * np.cos(d), s * np.sin(d)], axis=-1)

    # Observational context: (n_tasks, n_obs). Its law does not depend on lam.
    x0_ctx = rng.standard_normal((n_tasks, n_obs))
    W_ctx = rng.standard_normal((n_tasks, n_obs, 2))
    x1_ctx = a[:, None] * x0_ctx + (loading(x0_ctx, lam[:, None]) * W_ctx).sum(-1)

    # Factual sample and its counterfactual under do(X0 = alpha), sharing the exogenous W.
    x0 = rng.standard_normal(n_tasks)
    W = rng.standard_normal((n_tasks, 2))
    x1 = a * x0 + (loading(x0, lam) * W).sum(-1)
    alpha = _sample_alpha_away_from_x0(x0, prior.alpha_range, su, sv, rng, min_angle_gap)
    y1 = a * alpha + (loading(alpha, lam) * W).sum(-1)

    context = np.stack([x0_ctx, x1_ctx], axis=-1).astype(np.float32)     # (T, n_obs, 2)
    factual = np.stack([x0, x1], axis=-1).astype(np.float32)             # (T, 2)
    target = np.stack([alpha, y1], axis=-1).astype(np.float32)           # (T, 2)
    intervention = np.stack([alpha, np.zeros_like(alpha)], axis=-1).astype(np.float32)

    int_index = torch.zeros((1, 1, 1), dtype=torch.int64)
    samples = [
        (
            (
                torch.from_numpy(intervention[i]).reshape(1, 2, 1),
                int_index,
                torch.from_numpy(factual[i]).reshape(1, 2, 1),
                torch.from_numpy(context[i]).reshape(n_obs, 2, 1),
            ),
            torch.from_numpy(target[i]).reshape(1, 2, 1),
        )
        for i in range(n_tasks)
    ]
    scm_params = np.stack([a, lam], axis=-1).astype(np.float64)

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save((samples, scm_params), cache_path)
    return samples, scm_params
