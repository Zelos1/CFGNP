
import concurrent.futures
import json
import math
import multiprocessing
import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from cfgnp.data.two_node_dataset import (DEFAULT_ALPHA_RANGE, DEFAULT_LAM_KNOWN, DEFAULT_LAM_RANGE,
                                         DEFAULT_SIGMA_A, DEFAULT_SIGMA_U, DEFAULT_SIGMA_V,
                                         TwoNodePrior, create_dataset_two_node)
from cfgnp.graph_approach import train_graph as _train_graph
from cfgnp.graph_approach.train_graph import create_dataset_artificial_graph
from cfgnp.train_suite import BaseExperimentConfig
from cfgnp.training.train import train_counterfactual
from cfgnp.util.data_paths import OtherExperimentsPath
from cfgnp.util.util import DEVICE, mog_nll


DEFAULT_NUM_OBS = (5, 10, 20, 50, 100, 250)
DEFAULT_SEEDS = (42, 43, 44, 45, 46, 47)
N_KL_SAMPLES = 512
N_KL_SAMPLES_MODEL = 1000
N_BOOTSTRAP = 1000
QUANTILE_METHODS = ("pooled", "importance", "bootstrap", "query_bootstrap")
_DEFAULT_N_KL_SAMPLES = {"pooled": N_KL_SAMPLES, "importance": N_KL_SAMPLES_MODEL,
                        "bootstrap": N_BOOTSTRAP, "query_bootstrap": N_BOOTSTRAP}
LAM_QUAD_NODES = 32
MOG_VAR_EPS = 1e-4
LOG_2PI = math.log(2.0 * math.pi)


def variant_from_name(dataset_name: str) -> str:
    for part in dataset_name.split("_"):
        if part.upper() in ("ID", "UNID"):
            return part.upper()
    raise ValueError(f"Cannot infer the SCM variant from dataset_name '{dataset_name}'")


def num_obs_from_name(dataset_name: str, default: int = 100) -> int:
    for part in dataset_name.split("_"):
        if part.startswith("obs") and part[3:].isdigit():
            return int(part[3:])
    return default


class TwoNodeExperimentConfig(BaseExperimentConfig):

    def __init__(self, dataset_name: str, gnn_mode: bool = True, num_epochs: int = 400, save_code: bool = False,
                 lr: float = 1e-4, variant: Optional[str] = None, num_obs: Optional[int] = None,
                 num_samples: int = 40000, seed: int = 33, sigma_u: float = DEFAULT_SIGMA_U,
                 sigma_v: float = DEFAULT_SIGMA_V, sigma_a: float = DEFAULT_SIGMA_A,
                 lam_known: float = DEFAULT_LAM_KNOWN,
                 lam_range: Sequence[float] = DEFAULT_LAM_RANGE,
                 alpha_range: Sequence[float] = DEFAULT_ALPHA_RANGE,
                 cache_dir: Optional[str] = None):
        self.variant = (variant or variant_from_name(dataset_name)).upper()
        self.seed = seed
        self.cache_dir = cache_dir
        self.prior = TwoNodePrior(self.variant, sigma_u=sigma_u, sigma_v=sigma_v, sigma_a=sigma_a,
                                  lam_known=lam_known, lam_range=lam_range, alpha_range=alpha_range)
        self._num_obs = num_obs if num_obs is not None else num_obs_from_name(dataset_name)
        self._num_samples = num_samples
        self._scm_params = {}
        super().__init__(dataset_name, gnn_mode, num_epochs, save_code, lr)

    def configure(self):
        self.num_nodes = 2
        self.feature_dim = 1
        self.num_count = 1
        self.hidden_dim = 128
        self.num_heads_att = 6
        self.batch_size = 256
        self.num_obs = self._num_obs
        self.num_samples = self._num_samples

        self.target_indices = [1]
        self.target_indice_map = None
        self.train_fn = mog_nll
        self.loss_fn = nn.MSELoss()

        self.preloaded_split = True
        n_train = int(0.7 * self.num_samples)
        n_val = int(0.15 * self.num_samples)
        n_test = self.num_samples - n_train - n_val
        for split, size, offset in (("train", n_train, 0), ("val", n_val, 1), ("test", n_test, 2)):
            samples, params = create_dataset_two_node(
                size, n_obs=self.num_obs, variant=self.variant, seed=self.seed + 1000 * offset,
                prior=self.prior, cache_dir=self.cache_dir)
            setattr(self, f"{split}_dataset", samples)
            self._scm_params[split] = params

        if self.cache_dir is not None:
            stem = f"{self.variant.lower()}_obs{self.num_obs}_seed{self.seed}"
            for split in ("train", "val", "test"):
                setattr(self, f"{split}_cache_path",
                        os.path.join(self.cache_dir, f"graph_{stem}_{split}.pt"))

    def _to_graph_dataset(self, dataset, cache_path, split_name):
        graphs = create_dataset_artificial_graph(dataset, cache_path=cache_path)
        for graph, params in zip(graphs, self._scm_params[split_name]):
            graph.scm_params = torch.tensor(params, dtype=torch.float32).reshape(1, 2)
        return graphs

    def build(self):
        _train_graph.CUT_INDICES = False
        return super().build()




def _mixture_logpdf(y: torch.Tensor, log_w: torch.Tensor, mean: torch.Tensor,
                    var: torch.Tensor) -> torch.Tensor:
    log_gauss = -0.5 * (LOG_2PI + torch.log(var).unsqueeze(1)
                        + (y.unsqueeze(-1) - mean.unsqueeze(1)) ** 2 / var.unsqueeze(1))
    return torch.logsumexp(log_w.unsqueeze(1) + log_gauss, dim=-1)


def _mixture_sample(w: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, n: int,
                    generator: torch.Generator) -> torch.Tensor:
    idx = torch.multinomial(w, n, replacement=True, generator=generator)
    m = mean.gather(1, idx)
    s = var.gather(1, idx).sqrt()
    return m + s * torch.randn(m.shape, generator=generator, dtype=m.dtype)


def _kl_mc(p, q, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    y = _mixture_sample(*p, n_samples, generator)
    return (_mixture_logpdf(y, p[0].log(), p[1], p[2])
            - _mixture_logpdf(y, q[0].log(), q[1], q[2])).mean(dim=1)


def _kl_importance_sampled(p, q, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    y = _mixture_sample(*q, n_samples, generator)
    log_w = (_mixture_logpdf(y, p[0].log(), p[1], p[2])
             - _mixture_logpdf(y, q[0].log(), q[1], q[2]))
    return (torch.softmax(log_w, dim=1) * log_w).sum(dim=1)


def _model_mixture(preds: torch.Tensor) -> tuple:
    node_1 = preds[:, 0, 1, 0].double()                       # (B, mog_comp, 3)
    mean, var, weight = node_1[..., 0], node_1[..., 1], node_1[..., 2]
    var = var.clamp(min=MOG_VAR_EPS)
    weight = weight.clamp(min=1e-12)
    return weight / weight.sum(dim=-1, keepdim=True), mean, var


def _quantiles(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


@torch.no_grad()
def _collect_test_mixtures(model, loader, prior: TwoNodePrior, device=DEVICE) -> dict:
    model.eval()
    lam_nodes, lam_weights = prior.lambda_quadrature(LAM_QUAD_NODES)
    p_bma_cols = {"w": [], "mean": [], "var": []}
    p_true_cols = {"w": [], "mean": [], "var": []}
    p_model_cols = {"w": [], "mean": [], "var": []}
    y_true_col = []

    for batch in loader:
        batch = batch.to(device)
        n_graphs = batch.batch_size
        # The model overwrites x_int / x_orig / x_obs with embeddings -- read them first.
        x_orig = batch.x_orig.reshape(n_graphs, prior.num_nodes, -1)[:, :, 0]
        x_int = batch.x_int.reshape(n_graphs, prior.num_nodes, -1)[:, :, 0]
        x_obs = batch.x_obs.reshape(n_graphs, -1, prior.num_nodes)
        y1 = batch.y.reshape(n_graphs, prior.num_nodes, -1)[:, 1, 0]
        a_star, lam_star = batch.scm_params[:, 0], batch.scm_params[:, 1]

        x0 = x_orig[:, 0].double().cpu().numpy()
        x1 = x_orig[:, 1].double().cpu().numpy()
        alpha = x_int[:, 0].double().cpu().numpy()
        ctx = x_obs.double().cpu().numpy()
        a_star = a_star.cpu().numpy()
        lam_star = lam_star.cpu().numpy()

        # The evidence for the mechanism is the context set *and* the factual sample.
        evidence_x0 = np.concatenate([ctx[:, :, 0], x0[:, None]], axis=1)
        evidence_x1 = np.concatenate([ctx[:, :, 1], x1[:, None]], axis=1)
        post_mean, post_var = prior.weight_posterior(evidence_x0, evidence_x1)

        bma_w, bma_mean, bma_var = prior.bma_counterfactual(
            x0, x1, alpha, post_mean, post_var, lam_nodes, lam_weights)
        true_mean, true_var = prior.true_counterfactual(x0, x1, alpha, a_star, lam_star)

        as_t = lambda arr: torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float64)
        p_bma_cols["w"].append(as_t(bma_w).expand(n_graphs, -1).contiguous())
        p_bma_cols["mean"].append(as_t(bma_mean))
        p_bma_cols["var"].append(as_t(bma_var))
        p_true_cols["w"].append(torch.ones(n_graphs, 1, dtype=torch.float64))
        p_true_cols["mean"].append(as_t(true_mean).unsqueeze(-1))
        p_true_cols["var"].append(as_t(true_var).unsqueeze(-1))
        model_w, model_mean, model_var = tuple(t.cpu() for t in _model_mixture(model(batch)))
        p_model_cols["w"].append(model_w)
        p_model_cols["mean"].append(model_mean)
        p_model_cols["var"].append(model_var)
        y_true_col.append(as_t(y1.double().cpu().numpy()).unsqueeze(-1))

    cat = lambda cols: (torch.cat(cols["w"]), torch.cat(cols["mean"]), torch.cat(cols["var"]))
    return {
        "p_bma": cat(p_bma_cols),
        "p_true": cat(p_true_cols),
        "p_model": cat(p_model_cols),
        "y_true": torch.cat(y_true_col),
    }


def _point_stats(mixtures: dict) -> dict:
    y_true = mixtures["y_true"]
    stats = {}
    for key, (w, mean, var) in (("model", mixtures["p_model"]), ("bma", mixtures["p_bma"]),
                                ("true", mixtures["p_true"])):
        stats[f"{key}_nll"] = float((-_mixture_logpdf(y_true, w.log(), mean, var)[:, 0]).mean())
        point = (w * mean).sum(-1)
        stats[f"{key}_pred_std"] = float(
            ((w * (var + mean ** 2)).sum(-1) - point ** 2).clamp(min=0).sqrt().mean())
        if key == "model":
            stats["model_rmse"] = float(((point - y_true[:, 0]) ** 2).mean().sqrt())
    return stats


def _kl_bootstrap(mixtures: dict, n_bootstrap: int, generator: torch.Generator) -> tuple:
    p_bma, p_true, p_model = mixtures["p_bma"], mixtures["p_true"], mixtures["p_model"]
    kl_bma_boot, kl_true_boot = [], []
    for _ in range(n_bootstrap):
        y = _mixture_sample(*p_model, 1, generator)
        log_model = _mixture_logpdf(y, p_model[0].log(), p_model[1], p_model[2])[:, 0]
        log_bma = _mixture_logpdf(y, p_bma[0].log(), p_bma[1], p_bma[2])[:, 0]
        log_true = _mixture_logpdf(y, p_true[0].log(), p_true[1], p_true[2])[:, 0]
        kl_bma_boot.append((log_bma - log_model).mean())
        kl_true_boot.append((log_true - log_model).mean())
    return torch.stack(kl_bma_boot), torch.stack(kl_true_boot)


def _kl_query_bootstrap(per_query: torch.Tensor, n_bootstrap: int, generator: torch.Generator) -> torch.Tensor:
    n = per_query.shape[0]
    idx = torch.randint(0, n, (n_bootstrap, n), generator=generator)
    return per_query[idx].mean(dim=1)


@torch.no_grad()
def evaluate_two_node(model, loader, prior: TwoNodePrior, device=DEVICE,
                      n_kl_samples: int = N_KL_SAMPLES, seed: int = 0,
                      quantile_method: str = "pooled") -> dict:
    generator = torch.Generator().manual_seed(seed)
    mixtures = _collect_test_mixtures(model, loader, prior, device)
    p_bma, p_true, p_model = mixtures["p_bma"], mixtures["p_true"], mixtures["p_model"]

    kl_true_bma_base = _kl_mc(p_true, p_bma, N_KL_SAMPLES, generator)
    if quantile_method == "pooled":
        kl_bma_model = _kl_mc(p_bma, p_model, n_kl_samples, generator)
        kl_true_model = _kl_mc(p_true, p_model, n_kl_samples, generator)
        kl_true_bma = kl_true_bma_base
    elif quantile_method == "importance":
        kl_bma_model = _kl_importance_sampled(p_bma, p_model, n_kl_samples, generator)
        kl_true_model = _kl_importance_sampled(p_true, p_model, n_kl_samples, generator)
        kl_true_bma = kl_true_bma_base
    elif quantile_method == "bootstrap":
        kl_bma_model, kl_true_model = _kl_bootstrap(mixtures, n_kl_samples, generator)
        kl_true_bma = kl_true_bma_base
    elif quantile_method == "query_bootstrap":
        base_bma = _kl_mc(p_bma, p_model, N_KL_SAMPLES, generator)
        base_true = _kl_mc(p_true, p_model, N_KL_SAMPLES, generator)
        kl_bma_model = _kl_query_bootstrap(base_bma, n_kl_samples, generator)
        kl_true_model = _kl_query_bootstrap(base_true, n_kl_samples, generator)
        kl_true_bma = _kl_query_bootstrap(kl_true_bma_base, n_kl_samples, generator)
    else:
        raise ValueError(f"Unknown quantile_method '{quantile_method}'")

    arrays = {
        "kl_bma_model": kl_bma_model.numpy(),
        "kl_true_model": kl_true_model.numpy(),
        "kl_true_bma": kl_true_bma.numpy(),
    }
    point_stats = _point_stats(mixtures)
    summary = {
        "quantile_method": quantile_method,
        "n_test_queries": int(mixtures["y_true"].shape[0]),
        "kl_bma_model": _quantiles(arrays["kl_bma_model"]),
        "kl_true_model": _quantiles(arrays["kl_true_model"]),
        "kl_true_bma": _quantiles(arrays["kl_true_bma"]),
        "model_nll": point_stats["model_nll"],
        "bma_nll": point_stats["bma_nll"],
        "true_nll": point_stats["true_nll"],
        "model_rmse": point_stats["model_rmse"],
        "model_pred_std": point_stats["model_pred_std"],
        "bma_pred_std": point_stats["bma_pred_std"],
        "true_pred_std": point_stats["true_pred_std"],
    }
    return {"summary": summary, "per_query": arrays}


def train_two_node(variant: str, num_obs: int, num_epochs: int = 200, lr: float = 1e-3,
                   num_samples: int = 30000, seed: int = 42, patience: int = 25,
                   print_every: int = 10, model_dir: Optional[str] = None,
                   cache_dir: Optional[str] = None, use_acceleration: Optional[bool] = None,
                   n_kl_samples: Optional[int] = None, quantile_methods: Sequence[str] = QUANTILE_METHODS,
                   **config_kwargs) -> dict:
    variant = variant.upper()
    if model_dir is None:
        model_dir = OtherExperimentsPath.get_two_node_model_dir()
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_name = f"two_node_{variant.lower()}"
    config = TwoNodeExperimentConfig(dataset_name, gnn_mode=True, num_epochs=num_epochs, save_code=False, lr=lr,
                                     variant=variant, num_obs=num_obs, num_samples=num_samples, seed=seed,
                                     cache_dir=cache_dir, **config_kwargs).build()

    save_path = os.path.join(model_dir, f"{variant.lower()}_obs{num_obs}_seed{seed}.pt")
    print(f"\n=== two-node {variant} | num_obs={num_obs} | "
          f"{len(config.train_dataset)} train / {len(config.val_dataset)} val / {len(config.test_dataset)} test ===")

    results = train_counterfactual(
        config.gnn_mode, config.model, config.train_loader, config.val_loader, config.test_loader,
        config.device, config.loss_fn, config.target_indices, n_epochs=config.num_epochs, patience=100,
        save_path=save_path, print_every=print_every, train_loss_indices=config.train_loss_indices,
        target_indice_map=config.target_indice_map, lr=config.lr, train_fn=config.train_fn, restore_state=True,
        output_classified=config.output_classified, use_acceleration=use_acceleration)

    by_method, per_query = {}, {}
    for method in quantile_methods:
        n_samples = n_kl_samples if n_kl_samples is not None else _DEFAULT_N_KL_SAMPLES[method]
        evaluation = evaluate_two_node(config.model, config.test_loader, config.prior, config.device,
                                       n_kl_samples=n_samples, seed=seed, quantile_method=method)
        by_method[method] = evaluation["summary"]
        per_query[method] = evaluation["per_query"]

    run = {
        "variant": variant,
        "num_obs": num_obs,
        "num_samples": num_samples,
        "seed": seed,
        "lr": lr,
        "num_epochs": num_epochs,
        "best_epoch": results["best_epoch"],
        "best_val_nll": results["best_val"],
        "test_mse_loss": results["test_loss"],
        "save_path": results["save_path"],
        "history": results["history"],
        "by_method": by_method,
        **by_method[quantile_methods[0]],
    }
    run["_per_query"] = per_query
    print(f"--- {variant} num_obs={num_obs} ---")
    for method, s in by_method.items():
        bma, true, floor = s["kl_bma_model"], s["kl_true_model"], s["kl_true_bma"]
        print(f"    [{method:>14}] KL(p_BCM||p_theta) {bma['median']:.4f} [{bma['q10']:.4f}, {bma['q90']:.4f}] "
              f"std={bma['std']:.4f} | "
              f"KL(p*_BCM||p_theta) {true['median']:.4f} [{true['q10']:.4f}, {true['q90']:.4f}] "
              f"std={true['std']:.4f} | "
              f"irreducible {floor['median']:.4f} std={floor['std']:.4f} | RMSE {s['model_rmse']:.4f}")
    return run


def _aggregate_seed_runs(num_obs: int, seed_runs: list) -> dict:
    run = {
        "variant": seed_runs[0]["variant"],
        "num_obs": num_obs,
        "seeds": [r["seed"] for r in seed_runs],
        "n_seeds": len(seed_runs),
        "model_rmse": float(np.mean([r["model_rmse"] for r in seed_runs])),
    }
    if "bma_dir_mass" in seed_runs[0]:
        # Only the bivariate experiment reports this -- the posterior mass on the true direction.
        run["bma_dir_mass"] = float(np.mean([r["bma_dir_mass"] for r in seed_runs]))
    if "by_method" in seed_runs[0]:
        # ``train_two_node`` (two-node experiment): one summary per quantile method.
        methods = seed_runs[0]["by_method"].keys()
        run["by_method"] = {
            method: {key: _quantiles(np.array([r["by_method"][method][key]["mean"] for r in seed_runs]))
                    for key in ("kl_bma_model", "kl_true_model", "kl_true_bma")}
            for method in methods
        }
        for key in ("kl_bma_model", "kl_true_model", "kl_true_bma"):
            run[key] = run["by_method"][next(iter(methods))][key]
    else:
        # ``BivariateCounterfactualExperiment.run_experiment``: a single pooled estimate only.
        for key in ("kl_bma_model", "kl_true_model", "kl_true_bma"):
            run[key] = _quantiles(np.array([r[key]["mean"] for r in seed_runs]))
    run["_seed_runs"] = seed_runs
    return run


def run_two_node_sweep(variant: str, num_obs_values: Sequence[int] = DEFAULT_NUM_OBS, **kwargs) -> list:
    return [train_two_node(variant, num_obs, **kwargs) for num_obs in num_obs_values]


#: Runs at most this many (num_obs, seed) trainings at once in ``run_two_node_seed_sweep``.
MAX_PARALLEL_WORKERS = 1


def run_two_node_seed_sweep(variant: str, num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                            seeds: Sequence[int] = DEFAULT_SEEDS, max_workers: int = MAX_PARALLEL_WORKERS,
                            **kwargs) -> list:
    kwargs.pop("seed", None)
    jobs = [(num_obs, seed) for num_obs in num_obs_values for seed in seeds]
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
        futures = {(num_obs, seed): pool.submit(train_two_node, variant, num_obs, seed=seed, **kwargs)
                  for num_obs, seed in jobs}
        results = {key: future.result() for key, future in futures.items()}
    return [_aggregate_seed_runs(num_obs, [results[(num_obs, seed)] for seed in seeds])
           for num_obs in num_obs_values]


def _iter_leaf_runs(runs: list):
    for run in runs:
        yield from run.get("_seed_runs", [run])


CURVES = (
    ("kl_bma_model", "tab:blue", r"$\mathrm{KL}(p_T \Vert p_\theta)$"),
    ("kl_true_model", "tab:red", r"$\mathrm{KL}(p_T^* \Vert p_\theta)$"),
)
PANEL_TITLES = {"ID": "identifiable", "UNID": "non-identifiable"}
#: Fractional offset (in categorical x-slots) applied to each curve so their whiskers don't overlap.
CURVE_X_OFFSET = 0.12


def _method_stats(run: dict, method: str) -> Optional[dict]:
    by_method = run.get("by_method")
    if by_method is not None:
        return by_method.get(method)
    return run if method == "pooled" else None


def _plot_sweep(runs_by_variant: dict, out_path: str, methods: Sequence[str] = QUANTILE_METHODS):
    import matplotlib.pyplot as plt

    variants = [v for v, runs in runs_by_variant.items() if runs]
    methods = [m for m in methods
              if any(_method_stats(r, m) is not None for runs in runs_by_variant.values() for r in runs)]
    fig, axes = plt.subplots(len(variants), len(methods),
                             figsize=(5.5 * len(methods), 4.2 * len(variants)), squeeze=False)

    for row, variant in enumerate(variants):
        runs = sorted(runs_by_variant[variant], key=lambda r: r["num_obs"])
        positions = np.arange(len(runs), dtype=float)

        for col, method in enumerate(methods):
            ax = axes[row][col]
            for i, (key, color, label) in enumerate(CURVES):
                stats = [_method_stats(r, method) for r in runs]
                if any(s is None for s in stats):
                    continue
                shift = (i - (len(CURVES) - 1) / 2) * CURVE_X_OFFSET
                medians = np.array([s[key]["median"] for s in stats])
                q10s = np.array([s[key]["q10"] for s in stats])
                q90s = np.array([s[key]["q90"] for s in stats])
                ax.errorbar(positions + shift, medians, yerr=[medians - q10s, q90s - medians],
                            fmt="o-", color=color, label=label, markersize=4, linewidth=1.2,
                            elinewidth=1.6, capsize=4, capthick=1.6)

            ax.set_xticks(positions)
            ax.set_xticklabels([str(r["num_obs"]) for r in runs])
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Observational sample size")
            ax.set_ylabel("KL divergence")
            ax.set_title(f"{PANEL_TITLES.get(variant, variant)} -- {method}")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _plot_sweep_per_method(runs_by_variant: dict, out_dir: str, prefix: str,
                           methods: Sequence[str] = QUANTILE_METHODS) -> list:
    import matplotlib.pyplot as plt

    variants = [v for v, runs in runs_by_variant.items() if runs]
    methods = [m for m in methods
              if any(_method_stats(r, m) is not None for runs in runs_by_variant.values() for r in runs)]
    out_paths = []

    for method in methods:
        fig, axes = plt.subplots(1, len(variants), figsize=(5.5 * len(variants), 4.2), squeeze=False)
        axes = axes[0]

        for col, variant in enumerate(variants):
            runs = sorted(runs_by_variant[variant], key=lambda r: r["num_obs"])
            positions = np.arange(len(runs), dtype=float)
            ax = axes[col]

            for i, (key, color, label) in enumerate(CURVES):
                stats = [_method_stats(r, method) for r in runs]
                if any(s is None for s in stats):
                    continue
                shift = (i - (len(CURVES) - 1) / 2) * CURVE_X_OFFSET
                medians = np.array([s[key]["median"] for s in stats])
                q10s = np.array([s[key]["q10"] for s in stats])
                q90s = np.array([s[key]["q90"] for s in stats])
                ax.errorbar(positions + shift, medians, yerr=[medians - q10s, q90s - medians],
                            fmt="o-", color=color, label=label, markersize=4, linewidth=1.2,
                            elinewidth=1.6, capsize=4, capthick=1.6)

            ax.set_xticks(positions)
            ax.set_xticklabels([str(r["num_obs"]) for r in runs])
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Observational sample size")
            ax.set_ylabel("KL divergence")
            ax.set_title(PANEL_TITLES.get(variant, variant))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_{method}.svg")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {out_path}")
        out_paths.append(out_path)

    return out_paths


def plot_two_node_sweep_from_json(json_path: str, out_path: Optional[str] = None) -> str:
    with open(json_path) as f:
        summary = json.load(f)
    if out_path is None:
        out_path = os.path.splitext(json_path)[0] + ".svg"
    _plot_sweep(summary["runs"], out_path)
    return out_path


def run_two_node_identifiability(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                 variants: Sequence[str] = ("ID", "UNID"),
                                 results_dir: str = "./results/two_node",
                                 plot: bool = True, **kwargs) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    runs_by_variant = {variant.upper(): run_two_node_sweep(variant, num_obs_values, **kwargs)
                       for variant in variants}

    per_query_path = os.path.join(results_dir, f"two_node_kl_{timestamp}.npz")
    np.savez_compressed(per_query_path, **{
        f"{variant}_obs{run['num_obs']}_{method}_{key}": values
        for variant, runs in runs_by_variant.items()
        for run in runs
        for method, per_query in run.pop("_per_query").items()
        for key, values in per_query.items()})

    summary = {
        "timestamp": timestamp,
        "num_obs_values": list(num_obs_values),
        "quantile_methods": list(QUANTILE_METHODS),
        "per_query_path": per_query_path,
        "runs": runs_by_variant,
    }
    results_path = os.path.join(results_dir, f"two_node_sweep_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved results to {results_path}\nSaved per-query KLs to {per_query_path}")

    if plot:
        _plot_sweep(runs_by_variant, os.path.join(results_dir, f"two_node_sweep_{timestamp}.png"))

    header = ("variant  num_obs  method          KL(p_BCM||p_th) [q10,q90] (std)          "
             "KL(p*_BCM||p_th) [q10,q90] (std)         irreducible (std)   RMSE")
    print("\n" + header)
    for variant, runs in runs_by_variant.items():
        for r in runs:
            for method in QUANTILE_METHODS:
                s = _method_stats(r, method)
                if s is None:
                    continue
                bma, true, floor = s["kl_bma_model"], s["kl_true_model"], s["kl_true_bma"]
                print(f"{variant:>7}  {r['num_obs']:>7}  {method:>14}  "
                      f"{bma['median']:.4f} [{bma['q10']:.4f},{bma['q90']:.4f}] (std={bma['std']:.4f})  "
                      f"{true['median']:.4f} [{true['q10']:.4f},{true['q90']:.4f}] (std={true['std']:.4f})  "
                      f"{floor['median']:.4f} (std={floor['std']:.4f})  {s['model_rmse']:>5.3f}")

    return summary


def run_two_node_identifiability_seeds(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                       variants: Sequence[str] = ("ID", "UNID"),
                                       seeds: Sequence[int] = DEFAULT_SEEDS,
                                       results_dir: str = "./results/two_node",
                                       plot: bool = True, **kwargs) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    runs_by_variant = {variant.upper(): run_two_node_seed_sweep(variant, num_obs_values, seeds=seeds, **kwargs)
                       for variant in variants}

    # Per-seed per-query KLs are what each seed's own quantiles were built from; keep them next
    # to the summary so the figure can be redrawn (or re-quantiled) without retraining.
    per_query_path = os.path.join(results_dir, f"two_node_seed_sweep_kl_{timestamp}.npz")
    np.savez_compressed(per_query_path, **{
        f"{variant}_obs{leaf['num_obs']}_seed{leaf['seed']}_{method}_{key}": values
        for variant, runs in runs_by_variant.items()
        for leaf in _iter_leaf_runs(runs)
        for method, per_query in leaf.pop("_per_query").items()
        for key, values in per_query.items()})

    summary = {
        "timestamp": timestamp,
        "num_obs_values": list(num_obs_values),
        "seeds": list(seeds),
        "quantile_methods": list(QUANTILE_METHODS),
        "per_query_path": per_query_path,
        "runs": runs_by_variant,
    }
    results_path = os.path.join(results_dir, f"two_node_seed_sweep_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved results to {results_path}\nSaved per-query KLs to {per_query_path}")

    if plot:
        _plot_sweep(runs_by_variant, os.path.join(results_dir, f"two_node_seed_sweep_{timestamp}.png"))

    header = ("variant  num_obs  n_seeds  method          "
             "KL(p_BCM||p_th) [q10,q90] (std)          KL(p*_BCM||p_th) [q10,q90] (std)         "
             "irreducible (std)   RMSE")
    print("\n" + header)
    for variant, runs in runs_by_variant.items():
        for r in runs:
            for method in QUANTILE_METHODS:
                s = _method_stats(r, method)
                if s is None:
                    continue
                bma, true, floor = s["kl_bma_model"], s["kl_true_model"], s["kl_true_bma"]
                print(f"{variant:>7}  {r['num_obs']:>7}  {r['n_seeds']:>7}  {method:>14}  "
                      f"{bma['median']:.4f} [{bma['q10']:.4f},{bma['q90']:.4f}] (std={bma['std']:.4f})  "
                      f"{true['median']:.4f} [{true['q10']:.4f},{true['q90']:.4f}] (std={true['std']:.4f})  "
                      f"{floor['median']:.4f} (std={floor['std']:.4f})  {r['model_rmse']:>5.3f}")

    return summary


def run_two_node_identifiability_seeds_id(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                          seeds: Sequence[int] = DEFAULT_SEEDS, **kwargs) -> dict:
    return run_two_node_identifiability_seeds(num_obs_values, variants=("ID",), seeds=seeds, **kwargs)


def run_two_node_identifiability_seeds_unid(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                            seeds: Sequence[int] = DEFAULT_SEEDS, **kwargs) -> dict:
    return run_two_node_identifiability_seeds(num_obs_values, variants=("UNID",), seeds=seeds, **kwargs)


if __name__ == "__main__":
    run_two_node_identifiability()
