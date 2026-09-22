import torch
import random
import numpy as np
from typing import Tuple
import os

from cfgnp.util.data_paths import OtherExperimentsPath


def scms_lin(U: np.ndarray) -> np.ndarray:
    X = [0.0] * 5
    X[0] = U[0]                            # X1 = U1
    X[1] = U[1]                            # X2 = U2
    X[2] = X[0] + U[2]                     # X3 = X1 + U3
    X[3] = -X[1] + 0.5 * X[0] + U[3]       # X4 = -X2 + 0.5*X1 + U4
    X[4] = -1.5 * X[1] + U[4]              # X5 = -1.5*X2 + U5
    return np.array(X, dtype=float)


def scms_nlin(U: np.ndarray) -> np.ndarray:
    X = [0.0] * 5
    X[0] = U[0]
    X[1] = U[1]
    X[2] = X[0] + 0.5 * (X[0] ** 2) + U[2]
    X[3] = -X[1] + 0.5 * (X[0] ** 2) + U[3]
    X[4] = -1.5 * (X[1] ** 2) + U[4]
    return np.array(X, dtype=float)


def scms_nadd(U: np.ndarray) -> np.ndarray:
    X = [0.0] * 5
    X[0] = U[0]
    X[1] = U[1]
    X[2] = X[0] * U[2]
    X[3] = (-X[1] + 0.5 * (X[0] ** 2)) * U[3]
    X[4] = (-1.5 * (X[1] ** 2)) * U[4]
    return np.array(X, dtype=float)


descendants = {
    0: [2, 3],  # descendants of X1 (X3, X4)
    1: [3, 4],  # descendants of X2 (X4, X5)
    2: [],
    3: [],
    4: []
}

non_leaf_nodes = [i for i, ds in descendants.items() if len(ds) > 0]


def sample_U(n_samples: int) -> np.ndarray:
    """Draw exogenous noises U ~ N(0,1) for all nodes; returns shape (n_samples, 5)."""
    return np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 5))


def generate_observational(n_samples: int, variant: str = "LIN") -> Tuple[np.ndarray, np.ndarray]:
    """Generate observational dataset X_obs (n_samples x 5) and corresponding U_obs."""
    variant = variant.upper()
    U = sample_U(n_samples)
    X = np.zeros((n_samples, 5), dtype=float)
    for i in range(n_samples):
        if variant == "LIN":
            X[i] = scms_lin(U[i])
        elif variant == "NLIN":
            X[i] = scms_nlin(U[i])
        elif variant == "NADD":
            X[i] = scms_nadd(U[i])
        else:
            raise ValueError("variant must be one of 'LIN', 'NLIN', 'NADD'")
    return X, U


def do_intervention(alpha: float, intervene_index: int, variant: str = "LIN", n_samples: int = 1) -> np.ndarray:
    """
    Sample from the interventional distribution P^{do(X_j = alpha)} by:
      - setting X_j = alpha
      - sampling fresh U for other nodes
      - computing downstream nodes according to topological order
    Returns array shape (n_samples, 5)
    """
    variant = variant.upper()
    samples = np.zeros((n_samples, 5), dtype=float)
    for i in range(n_samples):
        U = np.random.normal(size=5)  # fresh U for interventional population
        X = [0.0] * 5
        # intervene or sample root values
        X[0] = alpha if intervene_index == 0 else U[0]
        X[1] = alpha if intervene_index == 1 else U[1]
        # compute children using the chosen variant
        if variant == "LIN":
            X[2] = X[0] + U[2]
            X[3] = -X[1] + 0.5 * X[0] + U[3]
            X[4] = -1.5 * X[1] + U[4]
        elif variant == "NLIN":
            X[2] = X[0] + 0.5 * (X[0] ** 2) + U[2]
            X[3] = -X[1] + 0.5 * (X[0] ** 2) + U[3]
            X[4] = -1.5 * (X[1] ** 2) + U[4]
        else:  # NADD
            X[2] = X[0] * U[2]
            X[3] = (-X[1] + 0.5 * (X[0] ** 2)) * U[3]
            X[4] = (-1.5 * (X[1] ** 2)) * U[4]
        samples[i] = np.array(X, dtype=float)
    return samples


def do_intervention_counterfactual(alpha: float, intervene_index: int, U: np.ndarray, variant: str = "LIN") -> np.ndarray:
    """
    Given exogenous noise vector U (shape (5,)) for one observational sample,
    compute the counterfactual x^{do(X_j=alpha)} using the same U for nondeterministic parts
    (i.e., reuse U). Useful if you want true counterfactuals.
    """
    variant = variant.upper()
    X = [0.0] * 5
    X[0] = alpha if intervene_index == 0 else U[0]
    X[1] = alpha if intervene_index == 1 else U[1]
    if variant == "LIN":
        X[2] = X[0] + U[2]
        X[3] = -X[1] + 0.5 * X[0] + U[3]
        X[4] = -1.5 * X[1] + U[4]
    elif variant == "NLIN":
        X[2] = X[0] + 0.5 * (X[0] ** 2) + U[2]
        X[3] = -X[1] + 0.5 * (X[0] ** 2) + U[3]
        X[4] = -1.5 * (X[1] ** 2) + U[4]
    else:  # NADD
        X[2] = X[0] * U[2]
        X[3] = (-X[1] + 0.5 * (X[0] ** 2)) * U[3]
        X[4] = (-1.5 * (X[1] ** 2)) * U[4]
    return np.array(X, dtype=float)


def create_dataset_Mshape(n_samples: int, n_obs: int = 100, variant: str = "LIN", counterfactuals: bool = True, seed: int = 42):
    """
    Create dataset with observational samples and interventional samples.
    Args:
      n_obs: number of observational samples to draw (this is the number of base P samples)
      variant: "LIN", "NLIN", or "NADD"
      counterfactuals: if True, generate interventional samples using the SAME U (counterfactuals).
                       if False, sample fresh U for interventional population (interventional distribution).
    Returns:
      samples: list of tuples in requested format
      X_obs: numpy array of observational samples (n_obs x 5)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    dataset_path = OtherExperimentsPath.get_mshape_dataset_path(variant, seed)
    if os.path.exists(dataset_path):
        samples = torch.load(dataset_path)
        return samples
    n_samples = n_samples // (5 * len(non_leaf_nodes))
    X_obs, U_obs = generate_observational(n_samples, variant=variant)
    stds = np.std(X_obs, axis=0, ddof=0)
    multipliers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    intervention_values = {j: multipliers * stds[j] for j in non_leaf_nodes}

    samples = []
    for i in range(n_samples):
        x_orig = X_obs[i].copy()
        u_orig = U_obs[i].copy()
        for j in non_leaf_nodes:
            for alpha in intervention_values[j]:
                if counterfactuals:
                    x_int = do_intervention_counterfactual(alpha=float(alpha), intervene_index=j, U=u_orig, variant=variant)
                else:
                    x_int = do_intervention(alpha=float(alpha), intervene_index=j, variant=variant, n_samples=1)[0]
                intervention_indices = np.array([j], dtype=int)

                sample_int = x_int[intervention_indices]
                zeros = np.zeros_like(x_int)
                zeros[intervention_indices] = sample_int
                sample_int = zeros

                obs, _ = generate_observational(n_obs, variant=variant)

                sample_tuple = (
                    (
                        torch.tensor(sample_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                        torch.tensor(intervention_indices, dtype=torch.int64).unsqueeze(0).unsqueeze(-1),
                        torch.tensor(x_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                        torch.tensor(obs, dtype=torch.float32).unsqueeze(-1),
                    ),
                    torch.tensor(x_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                )
                samples.append(sample_tuple)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save(samples, dataset_path)
    return samples



if __name__ == "__main__":
    # small example
    n_obs_small = 10
    for var in ["LIN", "NLIN", "NADD"]:
        print(f"\nGenerating variant: {var}")
        samples_var, X_obs_var = create_dataset_Mshape(100, n_obs=n_obs_small, variant=var, counterfactuals=True)
        print(f"Generated samples: {len(samples_var)} (n_obs={n_obs_small}, interventions per obs = {len(non_leaf_nodes)*5})")
        # print first example
        ((x_int_t, int_idx_t, x_orig_t, x_obs_t), dual_t) = samples_var[0]
        print("Example 0:")
        print("  intervened index:", int_idx_t.numpy())
        print("  x_int:", x_int_t.numpy())
        print("  x_orig:", x_orig_t.numpy())
        print("  sample_dual (descendant change):", dual_t.numpy())
