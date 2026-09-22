"""Two-node SCMs (X0 -> X1) with a *prior over mechanisms*, for the counterfactual
identifiability experiment.

This is the counterfactual analogue of the bivariate identifiability study in the MACE-TNP
paper: a single-edge linear-Gaussian model, a Bayesian prior over its mechanism, and two
regimes -- one where the query is identified from observational data and one where it is not.

Family
------
Fix known noise scales ``sigma_u, sigma_v > 0`` and write

    s(x)   = sqrt(sigma_u^2 + sigma_v^2 x^2),      phi(x) = arctan(sigma_v x / sigma_u).

For a *cross-world angle* ``lam in [0, 1]`` define the two-dimensional noise loading

    g_lam(x) = s(x) * ( cos(lam phi(x)), sin(lam phi(x)) ),        ||g_lam(x)|| = s(x),

and the SCM

    X0 := U0,                       U0 ~ N(0, 1)
    X1 := a X0 + <g_lam(X0), W>,    W  ~ N(0, I_2)   independent of U0.

Because ``||g_lam(x)|| = s(x)`` for every ``lam``, the observational law

    X0 ~ N(0, 1),   X1 | X0 = x ~ N(a x, s(x)^2)

and therefore also every interventional law is **completely independent of ``lam``**. Only the
counterfactuals see it. ``lam = 0`` recovers the point-identified variant of the original
experiment (a single monotone exogenous variable) and ``lam = 1`` recovers
``g(x) = (sigma_u, sigma_v x)``, the two-noise variant. The derivation is in
docs/two_node_identifiability.tex.

Why ``W`` is two-dimensional
----------------------------
``W`` is the exogenous variable attached to ``X1`` -- what an SCM would normally call ``U1`` -- but
it deliberately is *not* a scalar. Setting ``u = <ghat_lam(X0), W>`` with ``ghat_lam = g_lam/s``
gives a standard normal independent of ``X0`` satisfying ``X1 = a X0 + s(X0) u`` exactly, so
observationally this *is* a scalar additive-noise model. That reduction is invalid across worlds:
``u`` is a function of ``X0``, and under ``do(X0 = alpha)`` the loading rotates by ``lam*dphi``,

    <ghat_lam(alpha), W> = cos(lam dphi) u + sin(lam dphi) u_perp,

pulling in an orthogonal direction the factual sample never exposed. Abduction pins down exactly
one linear functional of ``W`` and leaves the other at its prior -- which is the entire source of
counterfactual uncertainty here. With a scalar noise, abduction is exact and every counterfactual
is a point mass -- see the "cannot be scalar" lemma in docs/two_node_identifiability.tex -- so
there would be no experiment.

What is unknown
---------------
Following the paper's "identifiable when the variances are known" setup, ``sigma_u`` and
``sigma_v`` are *known constants*; the unknowns drawn per task are

* the edge weight ``a ~ N(0, sigma_a^2)`` -- identified by the observational context ``D_obs``
  at the usual 1/M rate, and
* the cross-world angle ``lam``, which is invisible to *any* amount of observational or
  interventional data.

The two experiment variants differ only in whether ``lam`` is known:

``ID``    ``lam`` is fixed to ``lam_known`` and shared by every task, so the counterfactual query
          is identified in the limit of infinite observational context.
``UNID``  ``lam ~ U(lam_range)`` per task, so a residual, purely cross-world ambiguity survives
          no matter how large ``D_obs`` gets.

Everything the experiment needs as a reference is available in closed form -- see
``TwoNodePrior.bma_counterfactual`` (the Bayesian counterfactual model average) and
``TwoNodePrior.true_counterfactual`` (the counterfactual under the true mechanism).

Only node 0 is ever intervened on: node 1 is a leaf, so intervening on it has no descendants
to predict.
"""

import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

VARIANTS = ("ID", "UNID")

#: Known noise scales. A large ``sigma_v/sigma_u`` ratio makes ``phi`` sweep a wide angle, which
#: is what gives ``lam`` enough leverage for the non-identifiability to be visible.
DEFAULT_SIGMA_U = 0.4
DEFAULT_SIGMA_V = 1.4
#: Prior std of the edge weight ``a``. Wide enough that ``D_obs`` genuinely has to inform it.
DEFAULT_SIGMA_A = 1.0
#: ``UNID``: support of the uniform prior over ``lam``. Bounded away from 0 so the
#: true-mechanism counterfactual never degenerates to a Dirac (which would make its KL infinite).
DEFAULT_LAM_RANGE = (0.15, 1.0)
#: ``ID``: the single, known cross-world angle -- the midpoint of the ``UNID`` prior, so the two
#: panels have comparable target entropy and differ only in *knowledge* of ``lam``.
DEFAULT_LAM_KNOWN = 0.5 * (DEFAULT_LAM_RANGE[0] + DEFAULT_LAM_RANGE[1])
#: Interventions ``do(X0 = alpha)`` are drawn uniformly from this range (X0 is standard normal).
DEFAULT_ALPHA_RANGE = (-2.5, 2.5)
#: Floor on any reference counterfactual variance, mirroring the ``eps`` clamp inside ``mog_nll``.
#: Only bites for the measure-zero-ish queries with ``alpha ~= x0``.
VAR_FLOOR = 1e-4


def _as_array(*values):
    return tuple(np.asarray(v, dtype=np.float64) for v in values)


class TwoNodeSCM:
    """One draw from the family: the SCM ``X0 -> X1`` with weight ``a`` and cross-world angle ``lam``."""

    def __init__(self, a: float, lam: float, sigma_u: float = DEFAULT_SIGMA_U,
                 sigma_v: float = DEFAULT_SIGMA_V):
        self.a = float(a)
        self.lam = float(lam)
        self.sigma_u = float(sigma_u)
        self.sigma_v = float(sigma_v)

        self.num_nodes = 2
        self.descendants = {0: [1], 1: []}
        self.non_leaf_nodes = [0]

    # --- mechanism -------------------------------------------------------------------

    def noise_scale(self, x0):
        """Conditional std of X1 given X0 -- the same for every ``lam``."""
        (x0,) = _as_array(x0)
        return np.sqrt(self.sigma_u ** 2 + (self.sigma_v * x0) ** 2)

    def angle(self, x0):
        (x0,) = _as_array(x0)
        return np.arctan(self.sigma_v * x0 / self.sigma_u)

    def loading(self, x0):
        """``g_lam(x0)``, shape ``(..., 2)``."""
        d = self.lam * self.angle(x0)
        s = self.noise_scale(x0)
        return np.stack([s * np.cos(d), s * np.sin(d)], axis=-1)

    def structural_x1(self, x0, W):
        """Evaluate the X1 mechanism at an arbitrary X0 while holding the exogenous ``W`` fixed."""
        (x0,) = _as_array(x0)
        return self.a * x0 + (self.loading(x0) * W).sum(axis=-1)

    # --- sampling --------------------------------------------------------------------

    def sample_joint(self, n_samples: int, rng: np.random.Generator):
        """Return ``(x0, x1, W)`` for ``n_samples`` observational draws."""
        x0 = rng.standard_normal(n_samples)
        W = rng.standard_normal((n_samples, 2))
        return x0, self.structural_x1(x0, W), W

    def generate_observational(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        x0, x1, _ = self.sample_joint(n_samples, rng)
        return np.stack([x0, x1], axis=-1)

    def counterfactual(self, alpha, W) -> np.ndarray:
        """Counterfactual ``X`` under ``do(X0 = alpha)`` reusing the exogenous state ``W``."""
        (alpha,) = _as_array(alpha)
        return np.stack([alpha, self.structural_x1(alpha, W)], axis=-1)


def cross_world_gain(x0, alpha, lam, sigma_u: float, sigma_v: float) -> Tuple[np.ndarray, np.ndarray]:
    """The two numbers Theorem 1 reduces the counterfactual to, for ``do(X0 = alpha)``.

    Returns ``(k, tau2)`` such that, given the mechanism ``(a, lam)`` and the factual
    ``(x0, x1)``,

        X1 | do(X0 = alpha), x0, x1  ~  N( a alpha + k (x1 - a x0),  tau2 ).

    ``k = s(alpha)/s(x0) cos(lam dphi)`` is how much of the abducted residual carries over, and
    ``tau2 = s(alpha)^2 sin^2(lam dphi)`` is the irreducible cross-world variance. All arguments
    broadcast against each other.
    """
    x0, alpha, lam = _as_array(x0, alpha, lam)
    s0 = np.sqrt(sigma_u ** 2 + (sigma_v * x0) ** 2)
    sa = np.sqrt(sigma_u ** 2 + (sigma_v * alpha) ** 2)
    dphi = lam * (np.arctan(sigma_v * alpha / sigma_u) - np.arctan(sigma_v * x0 / sigma_u))
    k = sa / s0 * np.cos(dphi)
    tau2 = (sa * np.sin(dphi)) ** 2
    return k, np.maximum(tau2, VAR_FLOOR)


class TwoNodePrior:
    """The prior over two-node SCMs, plus the exact Bayesian references the experiment scores against."""

    def __init__(self, variant: str = "ID", sigma_u: float = DEFAULT_SIGMA_U,
                 sigma_v: float = DEFAULT_SIGMA_V, sigma_a: float = DEFAULT_SIGMA_A,
                 lam_known: float = DEFAULT_LAM_KNOWN,
                 lam_range: Sequence[float] = DEFAULT_LAM_RANGE,
                 alpha_range: Sequence[float] = DEFAULT_ALPHA_RANGE):
        variant = variant.upper()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
        self.variant = variant
        self.sigma_u = float(sigma_u)
        self.sigma_v = float(sigma_v)
        self.sigma_a = float(sigma_a)
        self.lam_known = float(lam_known)
        self.lam_range = (float(lam_range[0]), float(lam_range[1]))
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.num_nodes = 2

    # --- prior -----------------------------------------------------------------------

    def sample_weight(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma_a, n)

    def sample_lambda(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.variant == "ID":
            return np.full(n, self.lam_known)
        lo, hi = self.lam_range
        return rng.uniform(lo, hi, n)

    def lambda_quadrature(self, n_nodes: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Midpoint quadrature for the ``lam`` prior: ``(nodes, weights)``, weights summing to 1.

        ``ID`` collapses to the single known value. The integrand
        ``lam -> N(y; mean(lam), var(lam))`` is smooth, so the midpoint rule with a few dozen
        nodes is far more accurate than the Monte-Carlo error of the KL estimate that uses it.
        """
        if self.variant == "ID":
            return np.array([self.lam_known]), np.array([1.0])
        lo, hi = self.lam_range
        edges = np.linspace(lo, hi, n_nodes + 1)
        nodes = 0.5 * (edges[:-1] + edges[1:])
        return nodes, np.full(n_nodes, 1.0 / n_nodes)

    # --- posterior over the edge weight ----------------------------------------------

    def weight_posterior(self, x0, x1) -> Tuple[np.ndarray, np.ndarray]:
        """Exact Gaussian posterior ``p(a | evidence)`` from observed ``(x0, x1)`` pairs.

        ``x0``/``x1`` have shape ``(..., n_evidence)``; the returned mean/variance have shape
        ``(...)``. The likelihood is heteroscedastic-Gaussian with *known* scale ``s(x0)``, so
        with the ``N(0, sigma_a^2)`` prior this is ordinary conjugate weighted least squares --
        no quadrature and no sampling anywhere in the reference.
        """
        x0, x1 = _as_array(x0, x1)
        w = 1.0 / (self.sigma_u ** 2 + (self.sigma_v * x0) ** 2)
        precision = 1.0 / self.sigma_a ** 2 + (w * x0 ** 2).sum(axis=-1)
        var = 1.0 / precision
        return var * (w * x0 * x1).sum(axis=-1), var

    # --- the two reference counterfactual distributions -------------------------------

    def true_counterfactual(self, x0, x1, alpha, a_star, lam_star) -> Tuple[np.ndarray, np.ndarray]:
        """``p*_BCM(y1 | x0, x1, do(X0=alpha), a*, lam*)`` -- conditioned on the true mechanism.

        The counterpart of the paper's ``p*_BCM``: a Gaussian, and the target that only an
        *identifiable* query lets a model reach.
        """
        x0, x1, alpha, a_star = _as_array(x0, x1, alpha, a_star)
        k, tau2 = cross_world_gain(x0, alpha, lam_star, self.sigma_u, self.sigma_v)
        return a_star * alpha + k * (x1 - a_star * x0), tau2

    def bma_counterfactual(self, x0, x1, alpha, post_mean, post_var,
                           lam_nodes=None, lam_weights=None):
        """``p_BCM(y1 | x0, x1, do(X0=alpha), D_obs)`` -- the Bayesian counterfactual model average.

        Returns ``(weights, means, variances)`` of an exact Gaussian mixture with one component
        per ``lam`` quadrature node; ``weights`` has shape ``(J,)`` and the other two ``(B, J)``.

        Both integrals are done in closed form. ``lam`` is independent of the evidence (the
        likelihood does not involve it), so ``p(a, lam | D_obs, x^F) = p(a | D_obs, x^F) p(lam)``,
        and for a fixed ``lam`` the counterfactual mean is *affine* in ``a``::

            mean(a) = a (alpha - k x0) + k x1,      var = tau2,

        so marginalising the Gaussian ``a`` posterior just adds ``post_var (alpha - k x0)^2``.
        """
        if lam_nodes is None:
            lam_nodes, lam_weights = self.lambda_quadrature()
        lam_nodes, lam_weights = _as_array(lam_nodes, lam_weights)
        x0, x1, alpha, post_mean, post_var = _as_array(x0, x1, alpha, post_mean, post_var)

        k, tau2 = cross_world_gain(x0[..., None], alpha[..., None], lam_nodes,
                                   self.sigma_u, self.sigma_v)
        slope = alpha[..., None] - k * x0[..., None]           # d mean / d a
        means = post_mean[..., None] * slope + k * x1[..., None]
        variances = tau2 + post_var[..., None] * slope ** 2
        return lam_weights, means, variances


def create_dataset_two_node(n_tasks: int, n_obs: int = 100, variant: str = "ID", seed: int = 42,
                            prior: Optional[TwoNodePrior] = None,
                            cache_dir: Optional[str] = None):
    """Build the CFNP sample tuples for the two-node SCM -- **one fresh SCM per task**.

    Each task draws its own ``(a, lam)`` from the prior, its own observational context set of
    size ``n_obs``, its own factual sample and its own intervention value. That is what makes
    ``x_obs`` informative and the Bayesian model average a non-trivial target; a single fixed
    SCM would leave nothing for the context to say.

    Returns ``(samples, scm_params)`` where ``samples`` follows the layout every other
    generator in this package uses::

        ((sample_int, int_indices, x_orig, x_obs), y)
         sample_int : (1, 2, 1)      zeros except the intervened node
         int_indices: (1, 1, 1)      always 0 -- node 1 is a leaf
         x_orig     : (1, 2, 1)      factual sample
         x_obs      : (n_obs, 2, 1)  observational context set from the same SCM
         y          : (1, 2, 1)      true counterfactual

    and ``scm_params`` is an ``(n_tasks, 2)`` array of the ground-truth ``(a, lam)`` per task,
    which the evaluator needs to form ``p*_BCM`` and which is *not* recoverable from the tuples.
    """
    prior = prior or TwoNodePrior(variant)
    cache_path = None
    if cache_dir is not None:
        cache_path = os.path.join(
            cache_dir, f"two_node_{prior.variant}_n{n_tasks}_obs{n_obs}_seed{seed}.pt")
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
    alpha = rng.uniform(*prior.alpha_range, n_tasks)
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


if __name__ == "__main__":
    for var in VARIANTS:
        prior = TwoNodePrior(variant=var)
        data, params = create_dataset_two_node(8, n_obs=64, variant=var, seed=0, prior=prior)
        ((x_int, int_idx, x_orig, x_obs), y) = data[0]
        a_star, lam_star = params[0]
        print(f"\nvariant {var}: {len(data)} tasks | true a {a_star:+.3f} lam {lam_star:.3f}")
        print("  x_int", tuple(x_int.shape), x_int.flatten().tolist())
        print("  x_orig", x_orig.flatten().tolist(), " x_obs", tuple(x_obs.shape))
        print("  y", y.flatten().tolist())

        ctx = x_obs[..., 0].numpy()
        x0f, x1f = x_orig[0, 0, 0].item(), x_orig[0, 1, 0].item()
        alpha = x_int[0, 0, 0].item()
        ev0 = np.concatenate([ctx[:, 0], [x0f]])
        ev1 = np.concatenate([ctx[:, 1], [x1f]])
        m, v = prior.weight_posterior(ev0, ev1)
        w, mu, sd2 = prior.bma_counterfactual(np.array([x0f]), np.array([x1f]), np.array([alpha]),
                                              np.array([m]), np.array([v]))
        tm, tv = prior.true_counterfactual(x0f, x1f, alpha, a_star, lam_star)
        bma_mean = float((w * mu[0]).sum())
        bma_var = float((w * (sd2[0] + mu[0] ** 2)).sum() - bma_mean ** 2)
        print(f"  a posterior: {m:+.3f} +- {math.sqrt(v):.3f}")
        print(f"  p_BCM : mean {bma_mean:+.4f} var {bma_var:.4f} ({len(w)} components)")
        print(f"  p*_BCM: mean {float(tm):+.4f} var {float(tv):.4f} | realised y1 {y[0, 1, 0].item():+.4f}")
