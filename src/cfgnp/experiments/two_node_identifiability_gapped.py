
import json
import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from cfgnp.data.two_node_dataset_gapped import MIN_ALPHA_ANGLE_GAP, create_dataset_two_node_gapped
from cfgnp.experiments.two_node_common import train_config_and_evaluate
from cfgnp.experiments.two_node_identifiability import (DEFAULT_NUM_OBS, QUANTILE_METHODS,
                                                         TwoNodeExperimentConfig, _method_stats,
                                                         _plot_sweep_per_method)
from cfgnp.util.data_paths import OtherExperimentsPath
from cfgnp.util.util import mog_nll


class TwoNodeGappedExperimentConfig(TwoNodeExperimentConfig):

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
            samples, params = create_dataset_two_node_gapped(
                size, n_obs=self.num_obs, variant=self.variant, seed=self.seed + 1000 * offset,
                prior=self.prior, cache_dir=self.cache_dir)
            setattr(self, f"{split}_dataset", samples)
            self._scm_params[split] = params

        if self.cache_dir is not None:
            stem = f"gapped_{self.variant.lower()}_obs{self.num_obs}_seed{self.seed}"
            for split in ("train", "val", "test"):
                setattr(self, f"{split}_cache_path",
                        os.path.join(self.cache_dir, f"graph_{stem}_{split}.pt"))


def train_two_node_gapped(variant: str, num_obs: int, num_epochs: int = 200, lr: float = 1e-3,
                          num_samples: int = 30000, seed: int = 42, model_dir: Optional[str] = None,
                          cache_dir: Optional[str] = None, use_acceleration: Optional[bool] = None,
                          print_every: int = 10, n_kl_samples: Optional[int] = None,
                          quantile_methods: Sequence[str] = QUANTILE_METHODS, **config_kwargs) -> dict:
    variant = variant.upper()
    if model_dir is None:
        model_dir = os.path.join(OtherExperimentsPath.get_two_node_model_dir(), "gapped")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_name = f"two_node_gapped_{variant.lower()}"
    config = TwoNodeGappedExperimentConfig(
        dataset_name, gnn_mode=True, num_epochs=num_epochs, save_code=False, lr=lr,
        variant=variant, num_obs=num_obs, num_samples=num_samples, seed=seed,
        cache_dir=cache_dir, **config_kwargs).build()

    save_path = os.path.join(model_dir, f"{variant.lower()}_obs{num_obs}_seed{seed}.pt")
    return train_config_and_evaluate(config, save_path, seed, print_every=print_every,
                                     quantile_methods=quantile_methods, n_kl_samples=n_kl_samples,
                                     use_acceleration=use_acceleration)


def run_two_node_gapped_sweep(variant: str, num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                              **kwargs) -> list:
    return [train_two_node_gapped(variant, num_obs, **kwargs) for num_obs in num_obs_values]


def run_two_node_identifiability_gapped(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                        variants: Sequence[str] = ("ID", "UNID"),
                                        results_dir: str = "./results/two_node_gapped",
                                        plot: bool = True, **kwargs) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    runs_by_variant = {variant.upper(): run_two_node_gapped_sweep(variant, num_obs_values, **kwargs)
                       for variant in variants}

    per_query_path = os.path.join(results_dir, f"two_node_gapped_kl_{timestamp}.npz")
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
        "min_alpha_angle_gap": MIN_ALPHA_ANGLE_GAP,
        "per_query_path": per_query_path,
        "runs": runs_by_variant,
    }
    results_path = os.path.join(results_dir, f"two_node_gapped_sweep_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved results to {results_path}\nSaved per-query KLs to {per_query_path}")

    if plot:
        _plot_sweep_per_method(runs_by_variant, results_dir, f"two_node_gapped_sweep_{timestamp}")

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


def run_two_node_identifiability_gapped_id(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                           **kwargs) -> dict:
    """``run_two_node_identifiability_gapped`` restricted to the identifiable (ID) variant."""
    return run_two_node_identifiability_gapped(num_obs_values, variants=("ID",), **kwargs)


def run_two_node_identifiability_gapped_unid(num_obs_values: Sequence[int] = DEFAULT_NUM_OBS,
                                             **kwargs) -> dict:
    """``run_two_node_identifiability_gapped`` restricted to the non-identifiable (UNID) variant."""
    return run_two_node_identifiability_gapped(num_obs_values, variants=("UNID",), **kwargs)


if __name__ == "__main__":
    run_two_node_identifiability_gapped()
