import numpy as np
import networkx as nx
from typing import Optional, List
from cfgnp.util.util import DAG
import torch

class ERDAGLinearSEM:
    """
    ER DAG generator + structural equation model.

    Graph:
      1. Sample random node order (topological order)
      2. Generate ER directed graph using NetworkX
      3. Keep only edges that respect the order => DAG

    SEM, depending on ``mechanism``:
      "linear":         X_i = a * sum_{j in Pa(i)} X_j + b + eps_i
      "sum_sine":       X_i = a * sum_{j in Pa(i)} sin(c * X_j) + b + eps_i
      "logsumexp":      X_i = a * log(sum_{j in Pa(i)} exp(X_j)) + b + eps_i
      "sqrt_sum":        X_i = a * |sum_{j in Pa(i)} X_j| + b + eps_i
      "sum_sqrt":       X_i = a * sum_{j in Pa(i)} sqrt(|X_j|) + b + eps_i
      "sum_tanh":       X_i = a * sum_{j in Pa(i)} tanh(c * X_j) + b + eps_i
      "geometric_mean": X_i = a * (prod_{j in Pa(i)} |X_j|)^(1 / card(Pa(i))) + b + eps_i
      eps_i ~ N(0, noise_scale^2)

    ``noise_scale`` is used by every sampling path (observational, factual and
    counterfactual). Keeping them on one scale matters: only "linear" is scale
    equivariant, so for the nonlinear mechanisms an observational context drawn at a
    different noise scale than the queries sits in a different regime of the mechanism
    (sin() in its linear regime, logsumexp collapsed onto log(#parents)) and describes a
    different function than the one that has to be predicted.
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int = 1,
        p_edge: float = 0.5,
        mechanism: str = "linear",
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        noise_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        assert num_nodes >= 1
        assert feature_dim >= 1
        assert 0.0 <= p_edge <= 1.0
        if mechanism not in ("linear", "sum_sine", "logsumexp", "sqrt_sum", "sum_sqrt", "sum_tanh", "geometric_mean"):
            raise ValueError(
                "mechanism must be 'linear', 'sum_sine', 'logsumexp', 'sqrt_sum', "
                "'sum_sqrt', 'sum_tanh' or 'geometric_mean'"
            )

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.p_edge = p_edge
        self.mechanism = mechanism
        default_a = {"linear": 0.5, "sum_sqrt": 0.5, "geometric_mean": 3.0}.get(mechanism, 1.0)
        default_b = {"geometric_mean": 0.1}.get(mechanism, 0.0)
        default_c = {"sum_tanh": 2.0}.get(mechanism, 0.5)
        self.a = a if a is not None else default_a
        self.b = b if b is not None else default_b
        self.c = c if c is not None else default_c
        self.noise_scale = noise_scale
        # Spread of the counterfactual noise around the factual noise, relative to the
        # noise scale so that the factual/counterfactual coupling is scale independent.
        self.hidden_vars = noise_scale * np.random.uniform(low=0, high=1,size=(num_nodes, feature_dim))

        self.rng = np.random.default_rng(seed)
        self.dag = DAG(num_nodes, p_edge, seed)
        self.graph = self.dag.graph
        self.topo_order = self.dag.topo_order

    def _mechanism_output(self, parent_vals: np.ndarray) -> np.ndarray:
        if self.mechanism == "sum_sine":
            return self.a * np.sin(self.c * parent_vals).sum(axis=1) + self.b
        if self.mechanism == "logsumexp":
            # shift by the per-sample max before exponentiating so that large parent
            # values cannot overflow (log-sum-exp trick)
            shift = parent_vals.max(axis=1, keepdims=True)
            lse = shift + np.log(np.exp(parent_vals - shift).sum(axis=1, keepdims=True))
            return self.a * lse.squeeze(axis=1) + self.b
        if self.mechanism == "sqrt_sum":
            return self.a * np.sqrt(np.abs(parent_vals.sum(axis=1)))+ self.b
        if self.mechanism == "sum_sqrt":
            return self.a * np.sqrt(np.abs(parent_vals)).sum(axis=1) + self.b
        if self.mechanism == "sum_tanh":
            return self.a * np.tanh(self.c * parent_vals).sum(axis=1) + self.b
        if self.mechanism == "geometric_mean":
            num_parents = parent_vals.shape[1]
            return self.a * np.abs(parent_vals).prod(axis=1) ** (1.0 / num_parents) + self.b
        return self.a * parent_vals.sum(axis=1) + self.b

    def parents_of(self, node: int) -> List[int]:
        return list(self.graph.predecessors(node))

    def adjacency_matrix(self) -> np.ndarray:
        """
        Return adjacency matrix A where A[child, parent] = 1
        """
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for u, v in self.graph.edges():
            A[v, u] = 1
        return A

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample data from the linear SEM.

        Returns
        -------
        data : np.ndarray of shape (n_samples, num_nodes, feature_dim)
        """
        data = np.zeros((n_samples, self.num_nodes, self.feature_dim))

        # evaluate nodes in topological order
        for node in self.topo_order:
            parents = self.parents_of(node)
            if len(parents) == 0:
                noise = self.rng.normal(
                    scale=self.noise_scale,
                    size=(n_samples, self.feature_dim)
                )
                data[:, node, :] = noise
            else:
                parent_vals = data[:, parents, :]
                det = self._mechanism_output(parent_vals)
                noise = self.rng.normal(
                    scale=self.noise_scale,
                    size=(n_samples, self.feature_dim)
                )
                data[:, node, :] = det + noise

        return data

    def generate_from_noise(self, noises: np.ndarray) -> np.ndarray:
        n_samples = noises.shape[0]
        data = np.zeros((n_samples, self.num_nodes, self.feature_dim))

        # evaluate nodes in topological order
        for node in self.topo_order:
            parents = self.parents_of(node)
            if len(parents) == 0:
                noise = noises[:, node, :]
                data[:, node, :] = noise
            else:
                parent_vals = data[:, parents, :]
                det = self._mechanism_output(parent_vals)
                noise = noises[:, node, :]
                data[:, node, :] = det + noise

        return data
    
    def sample_counterfactual_dmsm(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample counterfactual data from the linear SEM under interventions.

        Parameters
        ----------
        intervention_nodes : List[int]
            List of node indices to intervene on.
        intervention_values : np.ndarray of shape (len(intervention_nodes), feature_dim)
            Values to set for the intervened nodes.

        Returns
        -------
        data_cf : np.ndarray of shape (n_samples, num_nodes, feature_dim)
        """

        orig_noises = np.random.normal(
            scale=self.noise_scale,
            size=(n_samples, self.num_nodes, self.feature_dim)
        )

        data_orig = self.generate_from_noise(orig_noises)
        
        count_noises = np.random.normal(loc=orig_noises, scale=self.hidden_vars)
        data_dual = self.generate_from_noise(count_noises)
        return data_orig, data_dual

def create_dataset_artifical(feature_dim: int, num_nodes: int, num_samples: int, num_obs: int=100, num_count: int=1, batch_size: int=32, seed: int=42,
                              mechanism: str = "linear", a: Optional[float] = None, b: Optional[float] = None, c: Optional[float] = None,
                              noise_scale: float = 1.0):
    np.random.seed(seed)
    samples = []

    for i in range(num_samples):
        sample_dag = ERDAGLinearSEM(num_nodes=num_nodes, feature_dim=feature_dim, p_edge=0.5, mechanism=mechanism, a=a, b=b, c=c, noise_scale=noise_scale)
        sample_obs = sample_dag.sample(n_samples=num_obs)
        sample_orig, sample_dual = sample_dag.sample_counterfactual_dmsm(n_samples=num_count)

        intervention_indices = np.random.choice(range(num_nodes), size=int(np.random.uniform(1, num_nodes // 2)), replace=False)
        sample_int = sample_dual[:, intervention_indices]
        zeros = np.zeros_like(sample_dual)
        zeros[:, intervention_indices] = sample_int
        sample_int = zeros

        # The number of intervened nodes differs between samples, so a variable-length
        # index list cannot be collated into a batch. Encode the intervention set as a
        # fixed-size {0, 1} mask over the nodes instead, one row per counterfactual.
        intervention_mask = np.zeros((num_count, num_nodes), dtype=np.int64)
        intervention_mask[:, intervention_indices] = 1

        # Carry the ground-truth causal graph along with the sample so downstream
        # evaluation can group samples by graph (A[child, parent] == 1). The leading
        # axis keeps it aligned with the other per-sample tensors under collation.
        adjacency = torch.tensor(sample_dag.adjacency_matrix(), dtype=torch.int64).unsqueeze(0)

        samples.append(((torch.tensor(sample_int), torch.tensor(intervention_mask), torch.tensor(sample_orig), torch.tensor(sample_obs), adjacency), torch.tensor(sample_dual)))
    return samples