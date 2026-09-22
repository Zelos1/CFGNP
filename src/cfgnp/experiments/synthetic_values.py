from cfgnp.graph_approach.train_graph import eval_test_graph
from cfgnp.loss import compute_per_index_metrics_loader
from cfgnp.train_suite import ExperimentFactory
from cfgnp.util.data_paths import ChexpertPath, OtherExperimentsPath
from cfgnp.util.util import IMAGE_RESIZE, REGRESSION_TARGET_INDICES, DEVICE
import numpy as np
import torch.nn as nn
import torch
import shutil
import os
import json
from datetime import datetime
from benchmarking.ot_bcm.generic_causal_bgm import build_loan_bgm_model, build_triangle_bgm_model, build_mshape_bgm_model
from benchmarking.causal_flow_matching.benchmark_cfm import CFMWrapper
from benchmarking.benchmark_others import BGMWrapper, VACAWrapper

cfgnp_paths = {
    "german_loan": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/german_loan_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/german_loan_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/german_loan_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/german_loan_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/german_loan_4_4_model.pt"
    },
    "triangle_LIN": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_LIN_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_LIN_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_LIN_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_LIN_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_LIN_4_4_model.pt",
    },
    "triangle_NLIN": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NLIN_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NLIN_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NLIN_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NLIN_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NLIN_4_4_model.pt",
    },
    "triangle_NADD": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NADD_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NADD_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NADD_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NADD_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/triangle_NADD_4_4_model.pt"
    },
    "mshape_LIN": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_LIN_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_LIN_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_LIN_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_LIN_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_LIN_4_4_model.pt",
    },
    "mshape_NLIN": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NLIN_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NLIN_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NLIN_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NLIN_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NLIN_4_4_model.pt",
    },
    "mshape_NADD": {
        42: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NADD_42_42_model.pt",
        1: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NADD_1_1_model.pt",
        2: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NADD_2_2_model.pt",
        3: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NADD_3_3_model.pt",
        4: "/data/coml-intersection-joins/lina4921/artifacts/saved_models/mshape_NADD_4_4_model.pt"
    },
}

def benchmark_synthetic_with_std(results_dir: str = "./results/synthetic"):
    seeds = [42, 1, 2, 3, 4]

    datasets = ["german_loan", "triangle_LIN", "triangle_NLIN", "triangle_NADD", "mshape_LIN", "mshape_NLIN", "mshape_NADD"]

    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for seed in seeds:
            ds_name = f"{dataset}_{seed}"
            torch.manual_seed(seed)
            config = ExperimentFactory.create(ds_name, True, 100, True, 1e-4, seed=seed).build()
            benchmarked_models = {"bgm": BGMWrapper(ds_name), "cfm": CFMWrapper(ds_name)}
            # benchmarked_models = {"bgm": BGMWrapper(dataset)}
            # benchmarked_models = {"cfm": CFMWrapper(dataset)}

            for k, model in benchmarked_models.items():
                model = model.push_to_device(device=DEVICE)
                test_loss = eval_test_graph(config.test_loader, model, config.loss_fn, None, config.target_indice_map, DEVICE, False)
                results[dataset].setdefault(k, []).append(test_loss)

            cfgnp_model = config.model.to(device=DEVICE)
            cfgnp_model.load_state_dict(torch.load(cfgnp_paths[dataset][seed]))
            test_loss = eval_test_graph(config.test_loader, cfgnp_model, config.loss_fn, None, config.target_indice_map, DEVICE, False)
            results[dataset].setdefault("cfgnp", []).append(test_loss)

    summary = {
        dataset: {
            model: {"mean": float(np.mean(losses)), "std": float(np.std(losses))}
            for model, losses in model_losses.items()
        }
        for dataset, model_losses in results.items()
    }

    header = f"{'dataset':<15} {'model':<6} {'mean':>10} {'std':>10}"
    print(header)
    for dataset, model_stats in summary.items():
        for model, stats in model_stats.items():
            print(f"{dataset:<15} {model:<6} {stats['mean']:>10.4f} {stats['std']:>10.4f}")

    output = {
        "seeds": seeds,
        "summary": summary,
        "results": results,
    }

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    results_path = os.path.join(results_dir, f"synthetic_with_std_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved results to {results_path}")

    return summary