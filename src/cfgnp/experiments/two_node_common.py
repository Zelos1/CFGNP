import os
from typing import Optional, Sequence

from cfgnp.experiments.two_node_identifiability import (QUANTILE_METHODS, _DEFAULT_N_KL_SAMPLES,
                                                         evaluate_two_node)
from cfgnp.training.train import train_counterfactual


def train_config_and_evaluate(config, save_path: str, seed: int, print_every: int = 10,
                              quantile_methods: Sequence[str] = QUANTILE_METHODS,
                              n_kl_samples: Optional[int] = None,
                              use_acceleration: Optional[bool] = None) -> dict:

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"\n=== two-node {config.variant} | num_obs={config.num_obs} | "
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
        "variant": config.variant,
        "num_obs": config.num_obs,
        "num_samples": config.num_samples,
        "seed": seed,
        "lr": config.lr,
        "num_epochs": config.num_epochs,
        "best_epoch": results["best_epoch"],
        "best_val_nll": results["best_val"],
        "test_mse_loss": results["test_loss"],
        "save_path": results["save_path"],
        "history": results["history"],
        "by_method": by_method,
        **by_method[quantile_methods[0]],
    }
    run["_per_query"] = per_query
    print(f"--- {config.variant} num_obs={config.num_obs} ---")
    for method, s in by_method.items():
        bma, true, floor = s["kl_bma_model"], s["kl_true_model"], s["kl_true_bma"]
        print(f"    [{method:>14}] KL(p_BCM||p_theta) {bma['median']:.4f} [{bma['q10']:.4f}, {bma['q90']:.4f}] "
              f"std={bma['std']:.4f} | "
              f"KL(p*_BCM||p_theta) {true['median']:.4f} [{true['q10']:.4f}, {true['q90']:.4f}] "
              f"std={true['std']:.4f} | "
              f"irreducible {floor['median']:.4f} std={floor['std']:.4f} | RMSE {s['model_rmse']:.4f}")
    return run
