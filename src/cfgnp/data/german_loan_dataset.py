import torch
import random
import numpy as np
from typing import Tuple
import os
import math

from cfgnp.util.data_paths import OtherExperimentsPath

class LoanDataset():
    """
    SCM for the 'Loan' semi-synthetic dataset (7 endogenous variables):
    indices:
      0: G (gender)
      1: A (age)
      2: E (education)
      3: L (loan amount)
      4: D (loan duration)
      5: I (income)
      6: S (savings)
    """

    def __init__(self):
        # children / descendants (topology used to determine which nodes are non-leaf)
        self.descendants = {
            0: [2, 3, 4, 5],  # G -> E, L, D (via L) , I
            1: [2, 3, 4, 5],  # A -> E, L, D, I
            2: [5],           # E -> I
            3: [4],           # L -> D
            4: [],            # D is leaf
            5: [6],           # I -> S
            6: []             # S is leaf
        }
        self.non_leaf_nodes = [i for i, ds in self.descendants.items() if len(ds) > 0]
        # variable order / names for clarity
        self.var_names = ["G", "A", "E", "L", "D", "I", "S"]

    def scms(self, U: np.ndarray) -> np.ndarray:
        """
        Structural equations using U vector of length 7 (ordered as UG, UA, UE, UL, UD, UI, US).
        Returns X array length 7 in order [G,A,E,L,D,I,S].
        """
        UG, UA, UE, UL, UD, UI, US = U.tolist()
        X = [0.0] * 7

        # G (binary)
        X[0] = 1.0 if UG >= 1 else 0.0  # UG is {0,1} from Bernoulli

        # A (age, modeled as deviation from mean): A = -35 + UA
        X[1] = -35.0 + UA

        # E (education) -- interpreted from the provided expression
        # E = 0.5 + exp(1 - 0.5*G - 1/(1 + exp(-0.1*A))) - UE
        try:
            exp_term = math.exp(1.0 - 0.5 * X[0] - UE - 1.0 / (1.0 + math.exp(-0.1 * X[1])))
        except OverflowError:
            # numerical safety
            exp_term = math.exp(1.0 - 0.5 * X[0] - UE - 1.0 / (1.0 + math.exp(-0.1 * np.clip(X[1], -100, 100))))
        X[2] = - 0.5 + 1 / exp_term 

        # L (loan amount): L = 1 + 0.01(A-5)(5-A) + G + UL
        X[3] = 1.0 + 0.01 * (X[1] - 5.0) * (5.0 - X[1]) + X[0] + UL

        # D (loan duration): D = -1 + 0.1*A + 2*G + L + UD
        X[4] = -1.0 + 0.1 * X[1] + 2.0 * X[0] + X[3] + UD

        # I (income): I = -4 + 0.1*(A+35) + 2*G + G*E + UI
        X[5] = -4.0 + 0.1 * (X[1] + 35.0) + 2.0 * X[0] + (X[0] * X[2]) + UI

        # S (savings): S = -4 + 1.5 * I^2 if I>0 else -4 + 0 + US
        indicator = 1.0 if X[5] > 0.0 else 0.0
        X[6] = -4.0 + 1.5 * X[5] * indicator + US

        return np.array(X, dtype=float)

    def sample_U(self, n_samples: int) -> np.ndarray:
        """
        Sample exogenous noises U for n_samples.
        Order: UG, UA, UE, UL, UD, UI, US
        Distributions per the prompt:
          UG ~ Bernoulli(0.5)
          UA ~ Gamma(shape=10, scale=3.5)
          UE ~ N(0, 0.25)  => std = 0.5
          UL ~ N(0, 4)     => std = 2
          UD ~ N(0, 9)     => std = 3
          UI ~ N(0, 4)     => std = 2
          US ~ N(0, 25)    => std = 5
        """
        UG = np.random.binomial(1, 0.5, size=n_samples).astype(float)  # 0 or 1
        UA = np.random.gamma(shape=10.0, scale=3.5, size=n_samples)
        UE = np.random.normal(loc=0.0, scale=0.5, size=n_samples)
        UL = np.random.normal(loc=0.0, scale=2.0, size=n_samples)
        UD = np.random.normal(loc=0.0, scale=3.0, size=n_samples)
        UI = np.random.normal(loc=0.0, scale=2.0, size=n_samples)
        US = np.random.normal(loc=0.0, scale=5.0, size=n_samples)
        return np.stack([UG, UA, UE, UL, UD, UI, US], axis=-1)

    def generate_observational(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate observational dataset X_obs (n_samples x 7) and corresponding U_obs (n_samples x 7).
        """
        U = self.sample_U(n_samples)
        X = np.zeros((n_samples, 7), dtype=float)
        for i in range(n_samples):
            X[i] = self.scms(U[i])
        return X, U

    def do_intervention(self, alpha: float, intervene_index: int, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the interventional distribution P^{do(X_j = alpha)} by:
         - setting X_j = alpha
         - sampling fresh U for other nodes
         - computing downstream nodes according to topological order
        Returns array shape (n_samples, 7)
        """
        samples = np.zeros((n_samples, 7), dtype=float)
        for i in range(n_samples):
            U = self.sample_U(1)[0]  # fresh U
            # We'll compute in topological order: G,A,E,L,D,I,S (indices 0..6)
            X = [0.0] * 7

            # For root nodes G and A: either set to alpha (if intervened) or derive from U
            # UG/UA are in U at positions 0 and 1
            # G:
            if intervene_index == 0:
                X[0] = float(alpha)
            else:
                X[0] = 1.0 if U[0] >= 1 else 0.0

            # A:
            if intervene_index == 1:
                X[1] = float(alpha)
            else:
                X[1] = -35.0 + U[1]

            # E:
            if intervene_index == 2:
                X[2] = float(alpha)
            else:
                UG_, UA_, UE_, UL_, UD_, UI_, US_ = U.tolist()
                # reuse same formula as in scms
                try:
                    nonlinear_term = math.exp(1.0 - 0.5 * X[0] - 1.0 / (1.0 + math.exp(-0.1 * X[1])))
                except OverflowError:
                    nonlinear_term = math.exp(1.0 - 0.5 * X[0] - 1.0 / (1.0 + math.exp(-0.1 * np.clip(X[1], -100, 100))))
                X[2] = 0.5 + nonlinear_term - UE_

            # L:
            if intervene_index == 3:
                X[3] = float(alpha)
            else:
                X[3] = 1.0 + 0.01 * (X[1] - 5.0) * (5.0 - X[1]) + X[0] + U[3]

            # D:
            if intervene_index == 4:
                X[4] = float(alpha)
            else:
                X[4] = -1.0 + 0.1 * X[1] + 2.0 * X[0] + X[3] + U[4]

            # I:
            if intervene_index == 5:
                X[5] = float(alpha)
            else:
                X[5] = -4.0 + 0.1 * (X[1] + 35.0) + 2.0 * X[0] + (X[0] * X[2]) + U[5]

            # S:
            if intervene_index == 6:
                X[6] = float(alpha)
            else:
                indicator = 1.0 if X[5] > 0.0 else 0.0
                X[6] = -4.0 + 1.5 * (X[5] ** 2) * indicator + U[6]

            samples[i] = np.array(X, dtype=float)
        return samples

    def do_intervention_counterfactual(self, alpha: float, intervene_index: int, U: np.ndarray) -> np.ndarray:
        """
        Given a single exogenous noise vector U (shape (7,)) for an observational sample,
        compute the counterfactual x^{do(X_j=alpha)} using the same U for nondeterministic parts.
        Returns numpy array length 7.
        """
        U = U.copy()
        # compute in topological order using same U, but replace the chosen variable with alpha
        X = [0.0] * 7

        # G
        if intervene_index == 0:
            X[0] = float(alpha)
        else:
            X[0] = 1.0 if U[0] >= 1 else 0.0

        # A
        if intervene_index == 1:
            X[1] = float(alpha)
        else:
            X[1] = -35.0 + U[1]

        # E
        if intervene_index == 2:
            X[2] = float(alpha)
        else:
            try:
                nonlinear_term = math.exp(1.0 - 0.5 * X[0] - 1.0 / (1.0 + math.exp(-0.1 * X[1])))
            except OverflowError:
                nonlinear_term = math.exp(1.0 - 0.5 * X[0] - 1.0 / (1.0 + math.exp(-0.1 * np.clip(X[1], -100, 100))))
            X[2] = 0.5 + nonlinear_term - U[2]

        # L
        if intervene_index == 3:
            X[3] = float(alpha)
        else:
            X[3] = 1.0 + 0.01 * (X[1] - 5.0) * (5.0 - X[1]) + X[0] + U[3]

        # D
        if intervene_index == 4:
            X[4] = float(alpha)
        else:
            X[4] = -1.0 + 0.1 * X[1] + 2.0 * X[0] + X[3] + U[4]

        # I
        if intervene_index == 5:
            X[5] = float(alpha)
        else:
            X[5] = -4.0 + 0.1 * (X[1] + 35.0) + 2.0 * X[0] + (X[0] * X[2]) + U[5]

        # S
        if intervene_index == 6:
            X[6] = float(alpha)
        else:
            indicator = 1.0 if X[5] > 0.0 else 0.0
            X[6] = -4.0 + 1.5 * (X[5] ** 2) * indicator + U[6]

        return np.array(X, dtype=float)


def create_dataset_loan(n_samples: int, n_obs: int = 100, counterfactuals: bool = True, seed: int = 42):
    """
    Create dataset with observational samples and interventional (or counterfactual) samples.
    Args:
      n_samples: total budget for candidate pairs (will be partitioned similarly to triangle example)
      n_obs: number of observational samples to draw for 'obs' context arrays in sample tuples
      counterfactuals: if True generate interventional samples using SAME U (counterfactuals).
                       if False sample fresh U (interventional distribution).
      seed: randomness seed
    Returns:
      samples: list of tuples in format similar to triangle example
      X_obs: numpy array of observational samples (n_obs x 7)
    """
    ds = LoanDataset()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    save_path = OtherExperimentsPath.get_loan_dataset_path(seed, n_obs)
    if os.path.exists(save_path):
        samples = torch.load(save_path)
        return samples

    # Compute how many observational base samples to actually generate from the requested n_samples
    # Mimic the Triangle logic: split budget across interventions and non_leaf nodes.
    n_per_pair = 5 * len(ds.non_leaf_nodes)
    if n_per_pair == 0:
        raise RuntimeError("No non-leaf nodes found in the SCM.")
    n_base = max(1, n_samples // n_per_pair)

    # Observational base dataset
    X_obs, U_obs = ds.generate_observational(n_base)

    # compute stds across variables (use population ddof=0 like triangle)
    stds = np.std(X_obs, axis=0, ddof=0)
    multipliers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    intervention_values = {j: multipliers * stds[j] for j in ds.non_leaf_nodes}

    samples = []
    # For each base observational sample, create counterfactual/interventional variants at intervention_values
    for i in range(n_base):
        x_orig = X_obs[i].copy()
        u_orig = U_obs[i].copy()
        for j in ds.non_leaf_nodes:
            for alpha in intervention_values[j]:
                if counterfactuals:
                    x_int = ds.do_intervention_counterfactual(alpha=float(alpha), intervene_index=j, U=u_orig)
                else:
                    x_int = ds.do_intervention(alpha=float(alpha), intervene_index=j, n_samples=1)[0]

                intervention_indices = np.array([j], dtype=int)

                # create "sample_int" consisting of zeros except the intervened entry set to the intervened value
                sample_int = np.zeros_like(x_int)
                sample_int[j] = x_int[j]

                # generate observational context 'obs' (n_obs x 7)
                obs, _ = ds.generate_observational(n_obs)

                # pack tensors similar to the triangle example shapes:
                # ((sample_int, intervention_indices, x_orig, obs), x_int)
                sample_tuple = (
                    (
                        torch.tensor(sample_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),       # (1,7,1)
                        torch.tensor(intervention_indices, dtype=torch.int64).unsqueeze(0).unsqueeze(-1), # (1,1,1)
                        torch.tensor(x_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),          # (1,7,1)
                        torch.tensor(obs, dtype=torch.float32).unsqueeze(-1),                         # (n_obs,7,1)
                    ),
                    torch.tensor(x_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),              # (1,7,1) - true counterfactual
                )
                samples.append(sample_tuple)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(samples, save_path)
    return samples


if __name__ == "__main__":
    # small example / quick smoke test
    n_obs_small = 10
    print("Generating small loan dataset (counterfactuals=True)...")
    samples = create_dataset_loan(200, n_obs=n_obs_small, counterfactuals=True, seed=42)
    ((sample_int_t, int_idx_t, x_orig_t, x_obs_t), x_int_t) = samples[0]
    print("Example 0:")
    print("  intervened index:", int_idx_t.numpy())
    print("  sample_int (only intervened var preserved):", sample_int_t.numpy().reshape(-1)[:7])
    print("  x_int (full counterfactual):", x_int_t.numpy().reshape(-1)[:7])
    print("  x_orig (orig observational):", x_orig_t.numpy().reshape(-1)[:7])
    print("  obs shape (context observational samples):", x_obs_t.shape)