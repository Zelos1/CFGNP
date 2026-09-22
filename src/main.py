from cfgnp.train_suite import train_run
import matplotlib.pyplot as plt
from cfgnp.classifier import train_classifier
from cfgnp.training.train_med_image import train_med_image
from cfgnp.experiments.synthetic_values import benchmark_synthetic_with_std
from cfgnp.experiments.runtime_experiment import run_runtime_experiment, run_chexpert_runtime_experiment
from cfgnp.experiments import (run_two_node_identifiability_gapped,
                               run_two_node_identifiability_gapped_id, run_two_node_identifiability_gapped_unid,
                               run_artificial_size_experiment, evaluate_different_size, evaluate_all_sizes)
from cfgnp.data import CheXpertDataset, CheXpertClassificationDataset
from cfgnp.util.data_paths import OtherExperimentsPath

from cfgnp.experiments.train_benchmark_models import train_all_benchmark_loan, train_all_benchmark_mshape, train_all_benchmark_triangle

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import traceback

# from accelerate import Accelerator

# accelerator = Accelerator()

# if accelerator.is_main_process:
#     import debugpy
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for VS Code debugger to attach...")
#     debugpy.wait_for_client()


ARTIFICIAL_VARIANTS = ("LIN", "NLIN", "NADD")
ARTIFICIAL_DATASET_PATH_GETTERS = {
    "mshape": OtherExperimentsPath.get_mshape_dataset_path,
    "triangle": OtherExperimentsPath.get_triangle_dataset_path,
}


def artificial_datasets(seed: int = 42, families=("mshape", "triangle")):
    # The seed the experiment configs pass to the generators -- together with the family/variant
    # this gives the file create_dataset_Mshape/create_dataset_triangle writes (see OtherExperimentsPath).
    return [f"{family}_{variant}_{seed}" for family in families for variant in ARTIFICIAL_VARIANTS]


def _train_artificial(dataset_name: str, num_epochs: int, lr: float):
    print(f"=== start {dataset_name} ===", flush=True)
    seed = int(dataset_name.rsplit("_", 1)[-1])
    result = train_run(dataset_name, True, num_epochs, True, lr=lr,
                        save_path=OtherExperimentsPath.get_model_save_path(dataset_name, seed), use_acceleration=False, seed=seed)
    return result["test_loss"]


def run_artificial_sweep(datasets=None, num_epochs: int = 250, lr: float = 7e-4, max_parallel: int = 3, seed: int = 42):
    """Generate a dataset and train one model per mshape/triangle variant.

    The generated dataset file is deleted once its variant has been trained, so each job
    regenerates from the fixed seed instead of reusing a cache across jobs.

    The models are small enough to share one GPU, so ``max_parallel`` of them are trained
    concurrently. Separate processes rather than threads: train_run seeds torch globally and the
    dataset generation is pure-python, so threads would both race on the seed and serialise on the
    GIL. Spawn (not fork) because the children initialise their own CUDA context.

    Each train_run writes its own json under ./results/<dataset_name>/; a failure in one variant
    is logged and the sweep continues so a single job still yields the remaining models.
    """
    if datasets is None:
        datasets = artificial_datasets(seed)
    test_losses = {name: None for name in datasets}
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as pool:
        futures = {pool.submit(_train_artificial, name, num_epochs, lr): name for name in datasets}
        for future in concurrent.futures.as_completed(futures):
            dataset_name = futures[future]
            try:
                test_losses[dataset_name] = future.result()
                print(f"{dataset_name} test loss", test_losses[dataset_name], flush=True)
            except Exception:
                print(f"=== {dataset_name} failed ===", flush=True)
                traceback.print_exc()

    print("=== sweep summary ===")
    for dataset_name, test_loss in test_losses.items():
        print(f"{dataset_name}: {'failed' if test_loss is None else test_loss}")
    return test_losses


def main(task, seed: int = 42):
    # # Example usage of build_cfnp_model
    # model = build_cfnp_model(in_features=4, mog_comp=3)
    # print("CFNP model built:", model)
    print(task)
    
    if task == "med_image":
        train_med_image()
    if task == "classifier":
        train_classifier("Sex", 0, 2)
    if task == "chexpert_count":
        path_chexbert = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/{mode}.csv"
        save_path = "/data/coml-intersection-joins/lina4921/data/chexpert_new/chexpert_count_{mode}.pt"
        train_dataset = CheXpertDataset(None, path_chexbert.format(mode="train_cheXbert"), "notenc", save_path=save_path.format(mode="train_cheXbert")).make_counterfactuals(100)
        test_dataset = CheXpertDataset(None, path_chexbert.format(mode="valid"), "notenc", save_path=save_path.format(mode="valid")).make_counterfactuals(100)
    if task == "chexpert_count_classification":
        path_chexbert = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/{mode}.csv"
        save_path = "/data/coml-intersection-joins/lina4921/data/chexpert_nnew/chexpert_count_{mode}.pt"
        train_dataset = CheXpertClassificationDataset(None, path_chexbert.format(mode="train_cheXbert"), "notenc", save_path=save_path.format(mode="train_cheXbert")).make_counterfactuals(100)
        test_dataset = CheXpertClassificationDataset(None, path_chexbert.format(mode="valid"), "notenc", save_path=save_path.format(mode="valid")).make_counterfactuals(100)
    if task == "graph_train_mshape_NLIN":
        result = train_run(f"mshape_NLIN_{seed}", True, 400, True, lr=3e-4, seed=seed)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()
    if task == "graph_train_mshape_NADD":
            result = train_run(f"mshape_NADD_{seed}", True, 400, True, lr=3e-4, seed=seed)

            print("Test loss", result["test_loss"])
            plt.plot(result["history"]["train_loss"], label="Train Loss")
            plt.plot(result["history"]["val_loss"], label="val loss")
            plt.show()
    if task == "graph_train_triangle_NLIN":
        result = train_run(f"triangle_NLIN_{seed}", True, 400, True, lr=3e-4, seed=seed)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()
    if task == "graph_train":
        # combine_scripts()
        result = train_run(f"german_loan_{seed}", True, 200, True, lr=3e-4, seed=seed)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()
    if task == "graph_train_all":
        run_artificial_sweep(seed=seed)
    if task == "graph_train_mshape":
        run_artificial_sweep(artificial_datasets(seed, families=("mshape",)), seed=seed)
    if task == "graph_train_triangle":
        run_artificial_sweep(artificial_datasets(seed, families=("triangle",)), seed=seed)
    if task == "chexpert_graph_train":
        # combine_scripts()
        result = train_run("chexpert", True, 200, True, lr=2e-4, seed=seed)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()
    if task.startswith("chexpert_ood"):
        # combine_scripts()
        result = train_run(task, True, 400, True, lr=5e-4, seed=seed)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()
    
    if task.startswith("chexpert_individual"):
        result = train_run(task, True, 400, True, lr=5e-4)

        print("Test loss", result["test_loss"])
        plt.plot(result["history"]["train_loss"], label="Train Loss")
        plt.plot(result["history"]["val_loss"], label="val loss")
        plt.show()

    if task == "two_node_gapped":
        run_two_node_identifiability_gapped()
    if task == "two_node_gapped_id":
        run_two_node_identifiability_gapped_id()
    if task == "two_node_gapped_unid":
        run_two_node_identifiability_gapped_unid()


    if task == "evaluate_artificial_linear":
        evaluate_all_sizes("linear")
    if task == "evaluate_artificial_sum_sine":
        evaluate_all_sizes("sum_sine")
    if task == "evaluate_artificial_logsumexp":
        evaluate_all_sizes("logsumexp")

    if task == "evaluate_artificial_row":
        evaluate_different_size()

    if task.startswith("artificial"):
        result = train_run(task, True, 250, True, lr=1e-4, seed=seed)
        print(result)

    if task == "seed_train_benchmark_loan":
        train_all_benchmark_loan(seed=seed)
    if task == "seed_train_benchmark_mshape":
        train_all_benchmark_mshape(seed=seed)
    if task == "seed_train_benchmark_triangle":
        train_all_benchmark_triangle(seed=seed)

    if task == "std_benchmark":
        benchmark_synthetic_with_std()

    if task == "runtime_experiment":
        run_runtime_experiment(seed=seed)
    if task == "chexpert_runtime_experiment":
        run_runtime_experiment(seed=seed)
        run_chexpert_runtime_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="runtime_experiment",
        help="Which pipeline to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the mshape/triangle/german_loan artificial experiments"
    )

    args = parser.parse_args()
    main(args.mode, args.seed)

