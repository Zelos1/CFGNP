from cfgnp.train_suite import train_run, ExperimentFactory
from cfgnp.graph_approach.train_graph import eval_test_graph
import torch
from cfgnp.util.util import DEVICE

from cfgnp.util.data_paths import OtherExperimentsPath
from cfgnp.graph_approach.train_graph import create_dataset_artificial_graph
from torch_geometric.loader import DataLoader as GDataLoader
from cfgnp.graph_approach.eval_by_causal_graph import eval_mse_by_causal_graph, mog_mean_forward
from cfgnp.util.util import REGRESSION_TARGET_INDICES, DEVICE
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import traceback

from benchmarking.causal_flow_matching.generic_causal_cfm import artificial_cfm_model
from benchmarking.ot_bcm.generic_causal_bgm import artificial_bgm_model
from benchmarking.vaca.generic_causal_vaca import artificial_vaca_model

from benchmarking.benchmark_others import BGMWrapper, VACAWrapper

NUM_NODES_ARRAY = [10, 15, 20, 25, 30]
SEED = 42
MECHANISM = "linear"

BENCHMARK_TRAINERS = {
    "cfm": artificial_cfm_model,
    "bgm": artificial_bgm_model,
    "vaca": artificial_vaca_model,
}

def run_artificial_size_experiment():
    results = {}
    for num_nodes in NUM_NODES_ARRAY:
        results[str(num_nodes)] = train_run(f"artificial_{num_nodes}_{MECHANISM}_{SEED}", True, num_epochs=250)
    print(results)


def evaluate_different_size():
    other_sizes = [15]
    for size in other_sizes:
        # config = ExperimentFactory.create(f"artificial_{size}_{MECHANISM}_{SEED}", True, 100, True, 1e-4).build()
        model_config = ExperimentFactory.create(f"artificial_10_{MECHANISM}_{SEED}", True, 100, True, 1e-4, seed=42).build()
        model = model_config.model

        testset = create_dataset_artificial_graph([], cache_path=OtherExperimentsPath.get_graph_ds_path("test"), target_num_nodes=4)
        default_cache_path = OtherExperimentsPath.get_graph_ds_path(f"artificial_{size}", "test", SEED)
        testset = create_dataset_artificial_graph(testset, cache_path=default_cache_path)
        test_loader = GDataLoader(testset, batch_size=4, shuffle=False)

        model.load_state_dict(torch.load("/data/coml-intersection-joins/lina4921/artifacts//graph_model_19-08-2026_02-22.pt"))
        model.to(device=DEVICE)
        test_loss = eval_test_graph(test_loader, model, model_config.loss_fn, None, model_config.target_indice_map, DEVICE, False)
        print(str(size), test_loss)

def load_cfgnp_retrained(mechanism, seed, size):
    model_config = ExperimentFactory.create(f"artificial_{size}_{mechanism}", True, 100, True, 1e-4, seed=seed).build()
    model = model_config.model
    model_path = {10: f"/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_10_{mechanism}_42_model.pt",
                  15: f"/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_15_{mechanism}_42_model.pt",
                  20: f"/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_20_{mechanism}_42_model.pt",
                  25: f"/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_25_{mechanism}_42_model.pt",
                  30: f"/data/coml-intersection-joins/lina4921/artifacts/saved_models/artificial_30_{mechanism}_42_model.pt",}
    model.load_state_dict(torch.load(model_path[size]))
    model.to(device=DEVICE)
    return mog_mean_forward(model)

def load_bgm_model(size, mechanism, seed):
    bgm_raw = artificial_bgm_model(size, mechanism, seed=seed)
    wrapped_model = BGMWrapper(None, bgm_raw)
    wrapped_model = wrapped_model.push_to_device(DEVICE)
    return mog_mean_forward(wrapped_model)

def load_vaca_model(size, mechanism, seed):
    vaca_raw = artificial_vaca_model(size, mechanism, seed=seed)
    wrapped_model = VACAWrapper(vaca_raw)
    wrapped_model = wrapped_model.push_to_device(DEVICE)
    return mog_mean_forward(wrapped_model)

def evaluate_all_sizes(mechanism = "linear", seed = 42, sizes = [10, 15, 20, 25, 30]):
    results_dir = "./results/artificial_nodesize/"
    results_path = f"{results_dir}{mechanism}_{seed}.json"

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
        for size in sizes:
            forward_methods = {
                "CFGNP(ours) retrained": load_cfgnp_retrained(mechanism, seed, size),
                "CFGNP(ours) size10": load_cfgnp_retrained(mechanism, seed, 10),
                "BGM": load_bgm_model(size, mechanism, seed=seed),
                "VACA": load_vaca_model(size, mechanism, seed=seed)
            }
            all_results[f"{size}"] = {}
            ds_config = ExperimentFactory.create(f"artificial_{size}_{mechanism}", True, 100, seed=seed).build()
            for name, fn_method in forward_methods.items():
                result = eval_mse_by_causal_graph(ds_config.test_loader, fn_method, device=DEVICE, target_indices=ds_config.target_indices,
                                                  target_indice_map=ds_config.target_indice_map)
                per_graph_mse = np.array([g["mse"] for g in result["per_graph"]])
                all_results[f"{size}"][name] = {
                    "mean_mse": result["mean_mse"],
                    "std_mse": result["std_mse"],
                    "q10_mse": float(np.quantile(per_graph_mse, 0.10)),
                    "q25_mse": float(np.quantile(per_graph_mse, 0.25)),
                    "q75_mse": float(np.quantile(per_graph_mse, 0.75)),
                    "q90_mse": float(np.quantile(per_graph_mse, 0.90)),
                }

        os.makedirs(results_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f)

    for size in sizes:
        size_results = all_results.get(f"{size}", {})
        for name, stats in size_results.items():
            print(f"size={size} {name}: mean_mse={stats['mean_mse']:.4f} std_mse={stats['std_mse']:.4f} "
                  f"[q10={stats['q10_mse']:.4f}, q25={stats['q25_mse']:.4f}, "
                  f"q75={stats['q75_mse']:.4f}, q90={stats['q90_mse']:.4f}]")

    _plot_size_experiment(all_results, mechanism, seed, sizes, results_dir)
    return all_results


_CATEGORICAL_COLORS = ["#2a78d6", "#4a3aa7", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#eb6834", "#e34948"]

label_name_map = {"cfgnp_retrained": "retrained CFGNP(ours)",
                  "cfgnp_size10": "10-CFGNP (ours)",
                  "bgm": "BGM",
                  "cfm": "CFM",
                  "vaca": "VACA"}

def _plot_size_experiment(all_results, mechanism, seed, sizes, results_dir):
    methods = sorted({name for size_results in all_results.values() for name in size_results})
    colors = {method: _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)] for i, method in enumerate(methods)}
    os.makedirs(results_dir, exist_ok=True)

    present_sizes = sorted({size for size in sizes if all_results.get(f"{size}", {})})

    fig, ax = plt.subplots(figsize=(6, 6))
    for method in methods:
        xs, means, lower, upper = [], [], [], []
        for size in sizes:
            stats = all_results.get(f"{size}", {}).get(method)
            if stats is None:
                continue
            xs.append(size)
            means.append(stats["mean_mse"])
            lower.append(max(0.0, stats["mean_mse"] - stats["q10_mse"]))
            upper.append(max(0.0, stats["q90_mse"] - stats["mean_mse"]))
        ax.errorbar(xs, means, yerr=[lower, upper], marker="o", markersize=8, linewidth=2,
                    capsize=3, color=colors[method], label=label_name_map.get(method, method))

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xticks(present_sizes)
    ax.legend(loc="upper left")
    fig.savefig(f"{results_dir}{mechanism}_{seed}_quantile.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for method in methods:
        xs, means, stds = [], [], []
        for size in sizes:
            stats = all_results.get(f"{size}", {}).get(method)
            if stats is None:
                continue
            xs.append(size)
            means.append(stats["mean_mse"])
            stds.append(stats["std_mse"])
        ax.errorbar(xs, means, yerr=stds, marker="o", markersize=8, linewidth=2,
                    capsize=3, color=colors[method], label=method)

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xticks(present_sizes)
    ax.legend(loc="upper left")
    fig.savefig(f"{results_dir}{mechanism}_{seed}_std.svg")
    plt.close(fig)


def _train_benchmark_model(method, num_nodes, mechanism, seed):
    print(f"=== start {method} artificial_{num_nodes}_{mechanism}_{seed} ===", flush=True)
    try:
        BENCHMARK_TRAINERS[method](num_nodes, mechanism, seed=seed)
        print(f"=== done {method} artificial_{num_nodes}_{mechanism}_{seed} ===", flush=True)
        return True
    except Exception:
        print(f"=== failed {method} artificial_{num_nodes}_{mechanism}_{seed} ===", flush=True)
        traceback.print_exc()
        return False


def run_all_benchmarks_size_experiment(mechanism=MECHANISM, max_parallel=3, seed=SEED,
                                        sizes=NUM_NODES_ARRAY, methods=("cfm", "bgm", "vaca")):
    """Train cfm/bgm/vaca on every artificial dataset size for one mechanism, in parallel.
    """
    jobs = [(method, size) for method in methods for size in sizes]
    ctx = mp.get_context("spawn")
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as pool:
        futures = {
            pool.submit(_train_benchmark_model, method, size, mechanism, seed): (method, size)
            for method, size in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            method, size = futures[future]
            results[(method, size)] = future.result()

    print("=== benchmark sweep summary ===")
    for (method, size), success in results.items():
        print(f"{method} artificial_{size}_{mechanism}_{seed}: {'ok' if success else 'failed'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=str, default=MECHANISM, help="Mechanism variant, e.g. linear")
    parser.add_argument("--max-parallel", type=int, default=3, help="Number of parallel training processes")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_all_benchmarks_size_experiment(mechanism=args.mechanism, max_parallel=args.max_parallel, seed=args.seed)