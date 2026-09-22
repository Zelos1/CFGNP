"""Wall-clock runtime of a forward pass over a test set, for CFGNP and the baselines.

Each timing is the average over ``n_runs`` (default 5) full passes over the test set, to smooth
out one-off scheduling noise.

Two entry points:
  - ``run_runtime_experiment``: synthetic suite (german_loan/triangle/mshape) + artificial
    node-size sweep, for cfgnp/bgm/cfm/vaca. If a model's weights were never trained for a given
    dataset, that cell is skipped (recorded as ``None``) rather than triggering a training run.
  - ``run_chexpert_runtime_experiment``: cfgnp vs. vci vs. deepbc on the chexpert test set.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from typing import Dict, Optional

import torch

from cfgnp.train_suite import ExperimentFactory
from cfgnp.util.data_paths import ChexpertPath, OtherExperimentsPath
from cfgnp.util.util import DEVICE

from benchmarking.ot_bcm.generic_causal_bgm import (
    build_loan_bgm_model, build_mshape_bgm_model, build_triangle_bgm_model, build_artificial_bgm_model,
)
from benchmarking.causal_flow_matching.generic_causal_cfm import (
    build_loan_cfm_model, build_mshape_cfm_model, build_triangle_cfm_model, build_artificial_cfm_model,
)
from benchmarking.vaca.generic_causal_vaca import (
    build_loan_vaca_model, build_mshape_vaca_model, build_triangle_vaca_model, build_artificial_vaca_model,
)
from benchmarking.benchmark_others import DeepBCChexpert
from vci.model import ModelWrapper

from cfgnp.experiments.synthetic_values import cfgnp_paths
from cfgnp.experiments.artificial_size_experiment import NUM_NODES_ARRAY

SYNTHETIC_DATASETS = ("german_loan", "triangle_LIN", "triangle_NLIN", "triangle_NADD",
                      "mshape_LIN", "mshape_NLIN", "mshape_NADD")

#: (build_fn, path_fn(variant, seed)) per family, per baseline method. ``variant`` is ignored
#: for "loan" but kept in the signature so every family can be called uniformly.
_BASELINE_BUILDERS = {
    "bgm": {
        "loan": (build_loan_bgm_model, lambda variant, seed: OtherExperimentsPath.get_loan_bgm_path(seed)),
        "triangle": (build_triangle_bgm_model, OtherExperimentsPath.get_triangle_bgm_path),
        "mshape": (build_mshape_bgm_model, OtherExperimentsPath.get_mshape_bgm_path),
    },
    "cfm": {
        "loan": (build_loan_cfm_model, lambda variant, seed: OtherExperimentsPath.get_loan_cfm_path(seed)),
        "triangle": (build_triangle_cfm_model, OtherExperimentsPath.get_triangle_cfm_path),
        "mshape": (build_mshape_cfm_model, OtherExperimentsPath.get_mshape_cfm_path),
    },
    "vaca": {
        "loan": (build_loan_vaca_model, lambda variant, seed: OtherExperimentsPath.get_loan_vaca_path(seed)),
        "triangle": (build_triangle_vaca_model, OtherExperimentsPath.get_triangle_vaca_path),
        "mshape": (build_mshape_vaca_model, OtherExperimentsPath.get_mshape_vaca_path),
    },
}

#: Hardcoded per-size CFGNP checkpoints, mirroring artificial_size_experiment.load_cfgnp_retrained.
_ARTIFICIAL_CFGNP_PATH_TEMPLATE = (
    "/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_{size}_{mechanism}_42_model.pt"
)


def _time_forward_pass(model, loader, device: torch.device, n_runs: int = 5) -> float:
    """Average wall-clock time, over ``n_runs`` full passes over ``loader``, of calling ``model(batch)``.

    ``torch.cuda.empty_cache()`` runs before each pass so a run can't reuse memory blocks the
    CUDA caching allocator kept warm from the previous run -- otherwise run 1 pays allocation
    cost the later runs don't, biasing the average.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            start = time.perf_counter()
            for batch in loader:
                batch = batch.to(device)
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def _synthetic_family_variant(dataset: str):
    if dataset == "german_loan":
        return "loan", None
    if dataset.startswith("triangle_"):
        return "triangle", dataset.split("_")[1]
    if dataset.startswith("mshape_"):
        return "mshape", dataset.split("_")[1]
    raise ValueError(f"Unrecognized synthetic dataset {dataset}")


def _load_synthetic_baseline(method: str, dataset: str, seed: int, device: torch.device):
    family, variant = _synthetic_family_variant(dataset)
    build_fn, path_fn = _BASELINE_BUILDERS[method][family]
    path = path_fn(variant, seed)
    if not os.path.exists(path):
        return None
    model = build_fn()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def _load_synthetic_cfgnp(dataset: str, seed: int, device: torch.device, model_config):
    path = cfgnp_paths.get(dataset, {}).get(seed)
    if path is None or not os.path.exists(path):
        return None
    model = model_config.model
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def _load_artificial_baseline(method: str, num_nodes: int, mechanism: str, seed: int, device: torch.device, feature_dim: int = 5):
    builders = {
        "bgm": (build_artificial_bgm_model, OtherExperimentsPath.get_artificial_bgm_path),
        "cfm": (build_artificial_cfm_model, OtherExperimentsPath.get_artificial_cfm_path),
        "vaca": (build_artificial_vaca_model, OtherExperimentsPath.get_artificial_vaca_path),
    }
    build_fn, path_fn = builders[method]
    path = path_fn(num_nodes, mechanism, seed)
    if not os.path.exists(path):
        return None
    model = build_fn(num_nodes, feature_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def _load_artificial_cfgnp(num_nodes: int, mechanism: str, device: torch.device, model_config):
    path = _ARTIFICIAL_CFGNP_PATH_TEMPLATE.format(size=num_nodes, mechanism=mechanism)
    if not os.path.exists(path):
        return None
    model = model_config.model
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def run_synthetic_runtime_experiment(seed: int = 42, n_runs: int = 5,
                                      device: Optional[torch.device] = None) -> Dict[str, Dict[str, Optional[float]]]:
    device = device or DEVICE
    ChexpertPath.load_from_json(None)

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for dataset in SYNTHETIC_DATASETS:
        ds_name = f"{dataset}_{seed}"
        torch.manual_seed(seed)
        model_config = ExperimentFactory.create(ds_name, True, 100, True, 1e-4, seed=seed).build()
        test_loader = model_config.test_loader

        results[dataset] = {}
        cfgnp_model = _load_synthetic_cfgnp(dataset, seed, device, model_config)
        results[dataset]["cfgnp"] = None if cfgnp_model is None else _time_forward_pass(cfgnp_model, test_loader, device, n_runs)

        for method in ("bgm", "cfm", "vaca"):
            model = _load_synthetic_baseline(method, dataset, seed, device)
            results[dataset][method] = None if model is None else _time_forward_pass(model, test_loader, device, n_runs)

    return results


def run_artificial_runtime_experiment(mechanism: str = "linear", seed: int = 42, sizes=NUM_NODES_ARRAY, n_runs: int = 5,
                                       device: Optional[torch.device] = None) -> Dict[str, Dict[str, Optional[float]]]:
    device = device or DEVICE
    ChexpertPath.load_from_json(None)

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for num_nodes in sizes:
        torch.manual_seed(seed)
        model_config = ExperimentFactory.create(f"artificial_{num_nodes}_{mechanism}", True, 100, True, 1e-4, seed=seed).build()
        test_loader = model_config.test_loader

        results[str(num_nodes)] = {}
        cfgnp_model = _load_artificial_cfgnp(num_nodes, mechanism, device, model_config)
        results[str(num_nodes)]["cfgnp"] = None if cfgnp_model is None else _time_forward_pass(cfgnp_model, test_loader, device, n_runs)

        for method in ("bgm", "cfm", "vaca"):
            model = _load_artificial_baseline(method, num_nodes, mechanism, seed, device)
            results[str(num_nodes)][method] = None if model is None else _time_forward_pass(model, test_loader, device, n_runs)

    return results


def _print_runtime_table(title: str, results: Dict[str, Dict[str, Optional[float]]]) -> None:
    print(f"\n=== {title} ===")
    header = f"{'dataset':<15} {'cfgnp':>12} {'bgm':>12} {'cfm':>12} {'vaca':>12}"
    print(header)
    for name, methods in results.items():
        row = [name]
        for method in ("cfgnp", "bgm", "cfm", "vaca"):
            value = methods.get(method)
            row.append("skipped" if value is None else f"{value:.4f}s")
        print(f"{row[0]:<15} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")


def run_runtime_experiment(mechanism: str = "linear", seed: int = 42, sizes=NUM_NODES_ARRAY, n_runs: int = 5,
                            results_dir: str = "./results/runtime") -> Dict[str, Dict]:
    synthetic_results = run_synthetic_runtime_experiment(seed=seed, n_runs=n_runs)
    artificial_results = run_artificial_runtime_experiment(mechanism=mechanism, seed=seed, sizes=sizes, n_runs=n_runs)

    _print_runtime_table(f"Synthetic suite: seconds per forward pass over the test set (avg of {n_runs} runs)", synthetic_results)
    _print_runtime_table(f"Artificial size sweep: seconds per forward pass over the test set (avg of {n_runs} runs)", artificial_results)

    output = {"seed": seed, "mechanism": mechanism, "n_runs": n_runs, "synthetic": synthetic_results, "artificial": artificial_results}
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    results_path = os.path.join(results_dir, f"runtime_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return output


_CHEXPERT_CFGNP_PATH = "/data/coml-intersection-joins/lina4921/artifacts//graph_model_18-08-2026_20-09.pt"


def run_chexpert_runtime_experiment(results_dir: str = "./results/runtime", n_runs: int = 5,
                                     device: Optional[torch.device] = None) -> Dict[str, Optional[float]]:
    device = device or DEVICE
    ChexpertPath.load_from_json(None)
    torch.manual_seed(42)
    config = ExperimentFactory.create("chexpert", True, 1, False, 1e-5).build()
    test_loader = config.test_loader

    models = {}

    if os.path.exists(_CHEXPERT_CFGNP_PATH):
        cfgnp_model = config.model.to(device)
        cfgnp_model.load_state_dict(torch.load(_CHEXPERT_CFGNP_PATH, map_location=device))
        models["cfgnp"] = cfgnp_model
    else:
        models["cfgnp"] = None

    try:
        models["vci"] = ModelWrapper(config.target_classifiers).push_to_device(device)
    except Exception:
        print("=== vci model unavailable, skipping ===", flush=True)
        traceback.print_exc()
        models["vci"] = None

    try:
        models["deepbc"] = DeepBCChexpert(config.target_classifiers).push_to_device(device)
    except Exception:
        print("=== deepbc model unavailable, skipping ===", flush=True)
        traceback.print_exc()
        models["deepbc"] = None

    results: Dict[str, Optional[float]] = {}
    for name, model in models.items():
        results[name] = None if model is None else _time_forward_pass(model, test_loader, device, n_runs)

    print(f"\n=== CheXpert: seconds per forward pass over the test set (avg of {n_runs} runs) ===")
    for name, value in results.items():
        print(f"{name:<10} {'skipped' if value is None else f'{value:.4f}s'}")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    results_path = os.path.join(results_dir, f"runtime_chexpert_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return results


if __name__ == "__main__":
    run_runtime_experiment()
    run_chexpert_runtime_experiment()
