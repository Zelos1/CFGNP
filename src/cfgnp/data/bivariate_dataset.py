"""Bivariate linear-Gaussian SCMs with an unknown *causal direction*, for the second
counterfactual identifiability experiment.

Same two nodes and same kind of query as [two_node_dataset.py](two_node_dataset.py), but the
ambiguity now sits in the **graph** rather than in a cross-world nuisance parameter. This is the
Markov-equivalence question the bivariate literature is built around -- is it ``X -> Y`` or
``Y -> X``? -- asked of a counterfactual, where the two answers are not merely different
parametrisations but different predictions about what would have happened.

The mechanism is the one in ``BivariateCounterfactualExperiment``
([../experiments/bivariate_counterfactual.py](../experiments/bivariate_counterfactual.py)),
lifted from a single fixed SCM to one drawn per task:

    X := U_X,             U_X ~ N(0, var_x)
    Y := w X + U_Y,       U_Y ~ N(0, var_y)      independent of U_X
    x_cf = x + Uniform(0.5, 2.0),   the query is  do(X = x_cf), target Y.

Three things are different from the original script, each for a reason the original could get
away with because it never trained a model:

* ``(w, var_x, var_y)`` are **drawn per task** rather than fixed at ``w = 1.5``. The model is
  trained across SCMs, so the observational context ``D_obs`` is what carries the information
  about the mechanism -- with one global SCM there would be nothing for the context to say and
  the Bayesian model average would collapse to the true model in both variants.
* The **direction is drawn per task** too, ``Y -> X`` as often as ``X -> Y``. The original always
  generates the forward model and lets only the model *average* hedge; but ``p_BCM`` is the
  Bayes-optimal target only if the task prior really does put half its mass on each direction.
  Otherwise the training distribution itself breaks the symmetry and the optimal predictive is
  the forward model, so ``KL(p_BCM || p_theta)`` would not go to 0.
* ``U_Y`` carries a **two-dimensional loading** (below). With scalar noise, abduction recovers
  ``U_Y = y - w x`` exactly, so every counterfactual is a point mass and every KL is 0 or
  infinite; the original papers over this with the ``eps = 1e-2`` smoothing width.

The two-dimensional loading
---------------------------
Write ``theta(x) = lam * arctan(x / c)`` and give the effect the noise loading

    g(x) = sqrt(var_y) * ( cos theta(x), sin theta(x) ),      U_Y = <g(X), W>,   W ~ N(0, I_2).

Since ``||g(x)|| = sqrt(var_y)`` for every ``x``, ``U_Y | X`` is still ``N(0, var_y)`` and the
observational law is *exactly* the bivariate Gaussian the identifiability argument needs. But
abduction now pins only one linear functional of ``W``: under ``do(X = x_cf)`` the loading rotates
by ``dtheta = theta(x_cf) - theta(x)`` and pulls in an orthogonal direction the factual never
exposed, leaving

    Y | do(X = x_cf), x, y, (w, var_y)  ~  N( w x_cf + cos(dtheta) (y - w x),  var_y sin^2 dtheta ),

which is the original's ``y_cf = y + w (x_cf - x)`` in the ``dtheta -> 0`` limit, but with genuine
counterfactual spread away from it. See the "cannot be scalar" lemma in
docs/two_node_identifiability.tex for the same argument in the sibling experiment.

In the reverse direction there is nothing to rotate: ``Y`` is the root, so intervening on its
child leaves it exactly where it was and the counterfactual is the point mass at ``y`` -- the
original's ``y_cf_rev = y``. That is not an artefact, it is what a Markovian two-node SCM says,
and it is why the non-identifiable panel is so stark: the model average has to hedge between
"``Y`` moves" and "``Y`` does not move at all". Point masses are represented at ``VAR_FLOOR``,
the variance clamp ``mog_nll`` already puts on the model's own predictive (the original's
``eps ** 2``), so the model can reach them and every KL stays finite.

What is unknown, and the two variants
-------------------------------------
A bivariate Gaussian never identifies its direction on its own: any covariance ``Sigma``
factorises both ways. Identifiability has to come from a *restriction on the error variances*,
which is the Peters-Bühlmann (2014) equal-error-variance result -- and note this is the opposite
way round from the original script's comment, which had unequal variances as the identifiable
case:

``ID``    ``var_x = var_y = v``, with ``v`` and ``w`` unknown per task. Reading ``Sigma`` in the
          reversed direction gives error variances ``Sigma_YY`` and
          ``Sigma_XX - Sigma_XY^2 / Sigma_YY``, which are equal only when ``w = 0``, so the
          equal-variance constraint singles out one direction: ``p(d | D_obs) -> 1`` at the truth
          and the counterfactual becomes identified in the large-context limit.
``UNID``  ``Sigma`` is unrestricted, drawn from an inverse-Wishart prior whose scale matrix is a
          multiple of the identity. Both directions then parametrise the *same* observational
          laws with the *same* prior over them, so the likelihood of any context set is a
          function of ``Sigma`` alone and

              p(d | D_obs, x, y) = p(d) = 1/2   exactly, at every context size.

          No amount of data moves it and a purely structural ambiguity survives.

A *known* error variance would identify the direction in either variant -- the constraint, not
the value, is what breaks the symmetry -- which is why ``v`` is drawn per task in ``ID`` too.

Both references the experiment scores against are closed form up to a one-dimensional quadrature
over the effect's noise variance: see ``BivariatePrior.bma_counterfactual`` (the Bayesian
counterfactual model average) and ``BivariatePrior.true_counterfactual``.
"""

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import expit, logsumexp
from scipy.stats import invgamma, invwishart

VARIANTS = ("ID", "UNID")

#: Cross-world angle gain and length scale of ``theta(x) = lam * arctan(x / c)``. Together with
#: the intervention shift below these keep ``|dtheta|`` in roughly ``[0.1, 1.2]`` rad, so
#: ``sin^2 dtheta`` -- the fraction of the effect's noise variance abduction cannot reach --
#: covers most of ``[0, 1]`` without ever vanishing.
DEFAULT_LAM = 0.6
DEFAULT_ANGLE_SCALE = 1.0
#: ``x_cf = x + Uniform(*this)``, exactly as in the original script. Strictly positive, so the
#: counterfactual world is always a different one and ``dtheta`` is never 0.
DEFAULT_SHIFT_RANGE = (0.5, 2.0)
#: ``ID``: prior std of the edge weight and the uniform prior on the shared error variance. The
#: equal-variance constraint has no leverage at ``w = 0`` -- there both directions give the same
#: observational law whatever the variance -- so a wide weight prior is what keeps the fraction of
#: never-identified tasks small. 2.0 leaves ~16% of tasks with ``|w| < 0.4``, which is the tail
#: the ``ID`` panel's upper whisker tracks.
DEFAULT_SIGMA_W = 2.0
DEFAULT_VAR_RANGE = (0.1, 2.0)
#: ``UNID``: inverse-Wishart prior on ``Sigma``, scale matrix ``iw_scale * (df - 3) * I`` so that
#: ``E[Sigma] = iw_scale * I``. Being a multiple of the identity makes it invariant under swapping
#: the two nodes, which is what pins the direction posterior at exactly 1/2. ``df`` is large
#: enough to keep ``Sigma`` well conditioned; lowering it widens the implied prior over the edge
#: weight but starts producing near-degenerate tasks.
DEFAULT_IW_DF = 12.0
DEFAULT_IW_SCALE = 1.0
#: Floor on any reference counterfactual variance -- the original's ``eps ** 2``, and the same
#: clamp ``mog_nll`` applies to the model's own variances. Unlike in the sibling experiment this
#: is load-bearing: it is the width at which the reverse branch's point mass is represented.
VAR_FLOOR = 1e-4
#: Quadrature nodes for the one integral that is not closed form (over ``var_y``).
VAR_QUAD_NODES = 48


def _as_array(*values):
    return tuple(np.asarray(v, dtype=np.float64) for v in values)


def cross_world_gain(x, x_cf, lam: float, angle_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """The two numbers the forward branch of the counterfactual reduces to.

    Returns ``(k, sin2)`` with ``k = cos(dtheta)`` -- how much of the abducted residual carries
    over into the counterfactual world -- and ``sin2 = sin^2(dtheta)``, the fraction of the
    effect's noise variance abduction cannot reach. Given the mechanism ``(w, var_y)``,

        Y | do(X = x_cf), x, y  ~  N( w x_cf + k (y - w x),  var_y sin2 ).
    """
    x, x_cf = _as_array(x, x_cf)
    dtheta = lam * (np.arctan(x_cf / angle_scale) - np.arctan(x / angle_scale))
    return np.cos(dtheta), np.sin(dtheta) ** 2


class BivariatePrior:
    """The prior over bivariate SCMs with an unknown direction, plus its exact Bayes references."""

    def __init__(self, variant: str = "ID", lam: float = DEFAULT_LAM,
                 angle_scale: float = DEFAULT_ANGLE_SCALE, sigma_w: float = DEFAULT_SIGMA_W,
                 var_range: Sequence[float] = DEFAULT_VAR_RANGE, iw_df: float = DEFAULT_IW_DF,
                 iw_scale: float = DEFAULT_IW_SCALE,
                 shift_range: Sequence[float] = DEFAULT_SHIFT_RANGE):
        variant = variant.upper()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
        self.variant = variant
        self.lam = float(lam)
        self.angle_scale = float(angle_scale)
        self.sigma_w = float(sigma_w)
        self.var_range = (float(var_range[0]), float(var_range[1]))
        self.iw_df = float(iw_df)
        self.iw_scale = float(iw_scale)
        self.shift_range = (float(shift_range[0]), float(shift_range[1]))
        self.num_nodes = 2
        #: ``E[Sigma] = Psi / (df - p - 1)`` for ``p = 2``.
        self.iw_psi = self.iw_scale * (self.iw_df - 3.0) * np.eye(2)

    # --- prior -----------------------------------------------------------------------

    def sample_mechanisms(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` mechanisms; returns an ``(n, 4)`` array of ``(direction, w, var_x, var_y)``.

        ``direction`` is 0 for ``X -> Y`` and 1 for ``Y -> X``; ``w``, ``var_x`` and ``var_y`` are
        always stated in the task's *own* causal order, i.e. ``var_x`` is the cause's variance and
        ``var_y`` the effect's noise variance whichever node the cause happens to be.

        ``ID`` draws the constrained mechanism directly. ``UNID`` draws the *observational law*
        ``Sigma`` first and only then reads it in the sampled direction, so the prior over the
        thing the data can see is identical for both directions, by construction.
        """
        direction = rng.integers(0, 2, n)
        if self.variant == "ID":
            var = rng.uniform(*self.var_range, n)
            return np.stack([direction, rng.normal(0.0, self.sigma_w, n), var, var], axis=-1)

        cov = np.asarray(invwishart.rvs(df=self.iw_df, scale=self.iw_psi, size=n,
                                        random_state=rng)).reshape(n, 2, 2)
        rows = np.arange(n)
        var_x = cov[rows, direction, direction]
        var_effect = cov[rows, 1 - direction, 1 - direction]
        cross = cov[:, 0, 1]
        w = cross / var_x
        return np.stack([direction, w, var_x,
                         np.maximum(var_effect - cross * w, VAR_FLOOR)], axis=-1)

    def noise_y(self, x, var_y, weights) -> np.ndarray:
        """``U_Y = <g(x), W>`` -- the effect's noise, evaluated at an arbitrary cause value.

        Marginally ``N(0, var_y)`` for every ``x``, which is what keeps the observational law
        Gaussian; holding ``weights`` fixed while moving ``x`` is what makes it a counterfactual.
        """
        x, var_y = _as_array(x, var_y)
        theta = self.lam * np.arctan(x / self.angle_scale)
        return np.sqrt(var_y) * (np.cos(theta) * weights[..., 0] + np.sin(theta) * weights[..., 1])

    def var_nodes(self, n_nodes: int = VAR_QUAD_NODES) -> np.ndarray:
        """Midpoint grid over the ``ID`` prior's uniform support for the shared error variance."""
        edges = np.linspace(*self.var_range, n_nodes + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    # --- posterior over the mechanism -------------------------------------------------

    def _log_evidence_id(self, s_cc, s_ce, s_ee, n_evidence: int, var):
        """``ID``: ``log p(evidence | var, direction)`` with the edge weight integrated out.

        Also returns the Gaussian posterior ``p(w | var, evidence)``. The evidence enters only
        through the sums of squares of its ``n_evidence`` pairs, taken in the direction's own
        cause/effect labelling (``s_cc = sum x_cause^2`` and so on), so the reverse direction is
        the same call with ``s_cc`` and ``s_ee`` swapped. Everything broadcasts: the stats arrive
        as ``(B, 1)`` and ``var`` as ``(1, J)``.
        """
        v_w = 1.0 / (1.0 / self.sigma_w ** 2 + s_cc / var)
        m_w = (s_ce / var) * v_w
        log_evidence = (-n_evidence * np.log(2.0 * np.pi * var) - (s_cc + s_ee) / (2.0 * var)
                        + 0.5 * np.log(v_w / self.sigma_w ** 2) + 0.5 * m_w ** 2 / v_w)
        return log_evidence, m_w, v_w

    def _posterior_unid(self, sxx, sxy, syy, n_evidence: int, n_nodes: int):
        """``UNID``: the posterior of ``(w, var_y)`` *read in the forward direction* ``X -> Y``.

        The inverse-Wishart prior is conjugate -- ``Sigma | evidence ~ IW(Psi_0 + S, df + n)`` --
        and an inverse-Wishart factorises exactly into the normal-inverse-gamma of the regression
        parametrisation it induces::

            var_y = Sigma_YY.X               ~ IG(df_n / 2, Psi_YY.X / 2),
            w     = Sigma_XY / Sigma_XX | var_y ~ N(Psi_XY / Psi_XX, var_y / Psi_XX).

        Returns ``(var_y, m_w, v_w)`` with ``var_y`` a ``(B, J)`` grid of equal-probability
        quantile nodes (weight ``1/J`` each) and ``v_w`` its matching conditional variance.
        """
        psi_xx = self.iw_psi[0, 0] + sxx
        psi_xy = self.iw_psi[0, 1] + sxy
        psi_yy = self.iw_psi[1, 1] + syy
        shape = 0.5 * (self.iw_df + n_evidence)
        scale = 0.5 * np.maximum(psi_yy - psi_xy ** 2 / psi_xx, VAR_FLOOR)

        quantiles = (np.arange(n_nodes) + 0.5) / n_nodes
        var_y = invgamma.ppf(quantiles, a=shape, scale=scale[:, None])
        return var_y, (psi_xy / psi_xx)[:, None], var_y / psi_xx[:, None]

    # --- the two reference counterfactual distributions -------------------------------

    def true_counterfactual(self, x, y, x_cf, direction, w_star, var_y_star):
        """``p*_BCM(y_cf | x, y, do(X = x_cf), d*, w*, var_y*)`` -- the true mechanism's answer.

        A Gaussian in the forward direction and the (floored) point mass at ``y`` in the reverse
        one, which is the original script's ``y_cf_rev = y``. This is the target that only an
        *identified* query lets a model reach.
        """
        x, y, x_cf, direction, w_star, var_y_star = _as_array(
            x, y, x_cf, direction, w_star, var_y_star)
        k, sin2 = cross_world_gain(x, x_cf, self.lam, self.angle_scale)
        forward = direction < 0.5
        mean = np.where(forward, w_star * (x_cf - k * x) + k * y, y)
        var = np.where(forward, np.maximum(var_y_star * sin2, VAR_FLOOR), VAR_FLOOR)
        return mean, var

    def bma_counterfactual(self, x, y, x_cf, sxx, sxy, syy, n_evidence: int,
                           n_nodes: int = VAR_QUAD_NODES):
        """``p_BCM(y_cf | x, y, do(X = x_cf), D_obs)`` -- the Bayesian counterfactual model average.

        Returns ``(weights, means, variances)`` of a Gaussian mixture, each ``(B, J + 1)``: one
        component per ``var_y`` quadrature node of the ``X -> Y`` reading, sharing the posterior
        mass ``p(d = 0 | evidence)`` between them, plus the reverse branch's point mass at ``y``
        carrying the rest. It is the original's 50/50 ``eval_mixture_logprob`` with the mixing
        weight supplied by the data and the forward branch's parameter uncertainty resolved.

        The evidence is the observational context *and* the factual sample, summarised by the sums
        of squares ``sxx, sxy, syy`` over its ``n_evidence`` pairs -- a bivariate Gaussian
        likelihood sees nothing else. Everything but the ``var_y`` integral is closed form: for a
        fixed ``var_y`` the counterfactual mean is affine in ``w``,

            mean(w) = w (x_cf - k x) + k y,      var = var_y sin2,

        so marginalising the Gaussian ``w`` posterior just adds ``v_w (x_cf - k x)^2``.
        """
        x, y, x_cf, sxx, sxy, syy = _as_array(x, y, x_cf, sxx, sxy, syy)
        k, sin2 = cross_world_gain(x, x_cf, self.lam, self.angle_scale)
        slope = (x_cf - k * x)[:, None]                       # d mean / d w
        base = (k * y)[:, None]

        if self.variant == "ID":
            var = self.var_nodes(n_nodes)[None, :]
            log_ev, m_w, v_w = self._log_evidence_id(sxx[:, None], sxy[:, None], syy[:, None],
                                                     n_evidence, var)
            # The reverse direction relabels which node is the cause, i.e. swaps the two sums of
            # squares. Its mechanism never reaches Y, so only its total evidence matters.
            log_ev_rev, _, _ = self._log_evidence_id(syy[:, None], sxy[:, None], sxx[:, None],
                                                     n_evidence, var)
            log_fwd, log_rev = logsumexp(log_ev, axis=1), logsumexp(log_ev_rev, axis=1)
            p_forward = expit(log_fwd - log_rev)
            node_weights = np.exp(log_ev - log_fwd[:, None])
        else:
            # The likelihood is a function of Sigma alone and the prior over Sigma is invariant
            # under swapping the two nodes, so the direction posterior *is* the prior. No data.
            var, m_w, v_w = self._posterior_unid(sxx, sxy, syy, n_evidence, n_nodes)
            p_forward = np.full(x.shape, 0.5)
            node_weights = np.full(var.shape, 1.0 / n_nodes)

        weights = np.concatenate([p_forward[:, None] * node_weights,
                                  (1.0 - p_forward)[:, None]], axis=1)
        means = np.concatenate([np.broadcast_to(m_w * slope + base, node_weights.shape),
                                y[:, None]], axis=1)
        variances = np.concatenate([
            np.broadcast_to(np.maximum(var * sin2[:, None] + v_w * slope ** 2, VAR_FLOOR),
                            node_weights.shape),
            np.full((x.shape[0], 1), VAR_FLOOR)], axis=1)
        return weights, means, variances


def create_dataset_bivariate(n_tasks: int, n_obs: int = 100, variant: str = "ID", seed: int = 42,
                             prior: Optional[BivariatePrior] = None,
                             cache_dir: Optional[str] = None):
    """Build the CFNP sample tuples for the bivariate SCM -- **one fresh SCM per task**.

    Each task draws its own direction and mechanism, its own observational context set of size
    ``n_obs``, its own factual sample and its own intervention. Returns ``(samples, scm_params)``
    with the layout every other generator in this package uses::

        ((sample_int, int_indices, x_orig, x_obs), y)
         sample_int : (1, 2, 1)      zeros except the intervened node
         int_indices: (1, 1, 1)      always 0 -- X is the intervention target
         x_orig     : (1, 2, 1)      factual sample (x, y)
         x_obs      : (n_obs, 2, 1)  observational context set from the same SCM
         y          : (1, 2, 1)      true counterfactual (x_cf, y_cf)

    and ``scm_params`` an ``(n_tasks, 4)`` array of the ground-truth
    ``(direction, w, var_x, var_y)`` per task, which the evaluator needs to form ``p*_BCM`` and
    which is not recoverable from the tuples.
    """
    prior = prior or BivariatePrior(variant)
    cache_path = None
    if cache_dir is not None:
        cache_path = os.path.join(
            cache_dir, f"bivariate_{prior.variant}_n{n_tasks}_obs{n_obs}_seed{seed}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

    rng = np.random.default_rng(seed)
    scm_params = prior.sample_mechanisms(n_tasks, rng)
    forward = scm_params[:, 0] < 0.5
    w, var_x, var_y = scm_params[:, 1], scm_params[:, 2], scm_params[:, 3]

    # --- BivariateCounterfactualExperiment's mechanism, per task and in causal order -------
    # Cause C and effect E: the observational context set, then the factual sample. Reads as the
    # original's `X`/`U_Y`/`Y` with a leading task axis and the two-dimensional loading.
    C_obs = rng.standard_normal((n_tasks, n_obs)) * np.sqrt(var_x)[:, None]
    U_E_obs = prior.noise_y(C_obs, var_y[:, None], rng.standard_normal((n_tasks, n_obs, 2)))
    E_obs = w[:, None] * C_obs + U_E_obs

    C = rng.standard_normal(n_tasks) * np.sqrt(var_x)
    W = rng.standard_normal((n_tasks, 2))
    E = w * C + prior.noise_y(C, var_y, W)

    x_obs = np.where(forward[:, None], C_obs, E_obs)
    y_obs = np.where(forward[:, None], E_obs, C_obs)
    x = np.where(forward, C, E)
    y = np.where(forward, E, C)

    # do(X = x_cf) reruns the effect's mechanism at x_cf when X is the cause, and leaves the
    # already-abducted root untouched when it is not -- the original's `y_cf_rev = y`.
    x_cf = x + rng.uniform(*prior.shift_range, n_tasks)
    y_cf = np.where(forward, w * x_cf + prior.noise_y(x_cf, var_y, W), y)

    context = np.stack([x_obs, y_obs], axis=-1).astype(np.float32)        # (T, n_obs, 2)
    factual = np.stack([x, y], axis=-1).astype(np.float32)                # (T, 2)
    target = np.stack([x_cf, y_cf], axis=-1).astype(np.float32)           # (T, 2)
    intervention = np.stack([x_cf, np.zeros_like(x_cf)], axis=-1).astype(np.float32)

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

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save((samples, scm_params), cache_path)
    return samples, scm_params


def evidence_statistics(context: np.ndarray, factual: np.ndarray):
    """Sums of squares of the context set *and* the factual sample -- all a Gaussian sees.

    ``context`` is ``(B, n_obs, 2)`` and ``factual`` ``(B, 2)``; returns
    ``(sxx, sxy, syy, n_evidence)`` with the first three of shape ``(B,)``.
    """
    x = np.concatenate([context[:, :, 0], factual[:, :1]], axis=1)
    y = np.concatenate([context[:, :, 1], factual[:, 1:2]], axis=1)
    return (x ** 2).sum(1), (x * y).sum(1), (y ** 2).sum(1), x.shape[1]


if __name__ == "__main__":
    for var in VARIANTS:
        prior = BivariatePrior(variant=var)
        data, params = create_dataset_bivariate(8, n_obs=64, variant=var, seed=0, prior=prior)
        ((x_int, int_idx, x_orig, x_obs), y) = data[0]
        d_star, w_star, var_x_star, var_y_star = params[0]
        print(f"\nvariant {var}: {len(data)} tasks | true direction "
              f"{'X -> Y' if d_star < 0.5 else 'Y -> X'} | w {w_star:+.3f} "
              f"var_x {var_x_star:.3f} var_y {var_y_star:.3f}")
        print("  x_orig", x_orig.flatten().tolist(), " x_obs", tuple(x_obs.shape))
        print("  y", y.flatten().tolist())

        context = x_obs[..., 0].numpy()[None]
        factual = x_orig.reshape(1, 2).numpy()
        sxx, sxy, syy, n_evidence = evidence_statistics(context, factual)
        x_cf = np.array([x_int[0, 0, 0].item()])
        weights, means, variances = prior.bma_counterfactual(
            factual[:, 0], factual[:, 1], x_cf, sxx, sxy, syy, n_evidence)
        true_mean, true_var = prior.true_counterfactual(
            factual[:, 0], factual[:, 1], x_cf, [d_star], [w_star], [var_y_star])
        bma_mean = float((weights[0] * means[0]).sum())
        bma_var = float((weights[0] * (variances[0] + means[0] ** 2)).sum() - bma_mean ** 2)
        print(f"  p(X -> Y | evidence): {1.0 - float(weights[0, -1]):.4f}")
        print(f"  p_BCM : mean {bma_mean:+.4f} var {bma_var:.4f} ({weights.shape[1]} components)")
        print(f"  p*_BCM: mean {float(true_mean[0]):+.4f} var {float(true_var[0]):.4f} "
              f"| realised y_cf {y[0, 1, 0].item():+.4f}")
