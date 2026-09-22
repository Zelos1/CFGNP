from collections import defaultdict
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import torch


def dag_from_adjacency(adjacency, node_labels=None) -> nx.DiGraph:
    """
    Build the causal DAG from an adjacency matrix in the convention of
    ``ERDAGLinearSEM.adjacency_matrix``: ``A[child, parent] == 1``.
    """
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square adjacency matrix, got shape {a.shape}")

    graph = nx.DiGraph()
    if node_labels is None:
        graph.add_nodes_from(range(a.shape[0]))
    else:
        graph.add_nodes_from((i, {"label": int(l)}) for i, l in enumerate(node_labels))

    children, parents = np.nonzero(a)
    graph.add_edges_from(zip(parents.tolist(), children.tolist()))
    return graph


def _label_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


class DAGIsomorphismGrouper:
    """
    Assigns DAGs to isomorphism classes, returning a stable integer id per class.
    """

    def __init__(self, wl_iterations: int = 3, exact: bool = True):
        self.wl_iterations = wl_iterations
        self.exact = exact
        self.representatives: List[np.ndarray] = []
        self.hashes: List[str] = []
        self._graphs: List[nx.DiGraph] = []
        self._buckets = defaultdict(list)
        self._seen = {}

    def __len__(self) -> int:
        return len(self.representatives)

    def class_of(self, adjacency, node_labels=None) -> int:
        a = np.asarray(adjacency, dtype=np.int8)
        labels = None if node_labels is None else np.asarray(node_labels, dtype=np.int64)

        seen_key = (a.shape, a.tobytes(), None if labels is None else labels.tobytes())
        cached = self._seen.get(seen_key)
        if cached is not None:
            return cached

        graph = dag_from_adjacency(a, labels)
        graph_hash = nx.weisfeiler_lehman_graph_hash(
            graph,
            iterations=self.wl_iterations,
            node_attr=None if labels is None else "label",
        )

        bucket = self._buckets[(a.shape[0], graph_hash)]
        class_id = None
        if self.exact:
            node_match = None if labels is None else _label_match
            for candidate in bucket:
                if nx.is_isomorphic(graph, self._graphs[candidate], node_match=node_match):
                    class_id = candidate
                    break
        elif bucket:
            class_id = bucket[0]

        if class_id is None:
            class_id = len(self.representatives)
            self.representatives.append(a)
            self.hashes.append(graph_hash)
            self._graphs.append(graph)
            bucket.append(class_id)

        self._seen[seen_key] = class_id
        return class_id


def select_target_entries(
    y_pred: torch.Tensor,
    targets: torch.Tensor,
    int_indices: torch.Tensor,
    target_indices: Optional[List[int]] = None,
    target_indice_map: Optional[torch.Tensor] = None,
):
    """
    Restrict predictions and targets to the supervised nodes.
    """
    if target_indice_map is not None:
        mapped = target_indice_map.to(y_pred.device)[int_indices.flatten()]
        mapped = mapped.reshape((int_indices.shape[0], y_pred.shape[1], -1))
        preds_selected = torch.gather(y_pred, 2, mapped.unsqueeze(-1).expand(-1, -1, -1, y_pred.shape[-1]))
        targets_selected = torch.gather(targets, 2, mapped.unsqueeze(-1).expand(-1, -1, -1, targets.shape[-1]))
        return preds_selected, targets_selected
    if target_indices is not None:
        return y_pred[:, :, target_indices], targets[:, :, target_indices]
    return y_pred, targets


def mog_mean_forward(model) -> Callable:
    """
    Wrap a CFNP model into a `forward_fn`, collapsing the mixture-of-Gaussians head to
    its (weight-averaged) mean, exactly as `eval_test_graph` does.
    """
    model.eval()

    def forward(batch):
        preds = model(batch)
        means = preds[..., 0]
        # Already a probability vector out of MoGLayer; see `eval_test_graph`.
        weights = preds[..., 2]
        return (means * weights).sum(dim=-1)

    return forward


def _batch_adjacencies(batch, batch_size: int) -> np.ndarray:
    if not hasattr(batch, "causal_adj") or batch.causal_adj is None:
        raise ValueError(
            "batch has no `causal_adj`; rebuild the graph dataset with "
            "create_dataset_artificial_graph(..., force_rebuild=True) so the "
            "ground-truth causal adjacency is carried along"
        )
    adj = batch.causal_adj
    if adj.dim() == 2:  # collated from (N, N) entries instead of (1, N, N)
        adj = adj.reshape(-1, adj.shape[-1], adj.shape[-1])
    if adj.shape[0] != batch_size:
        raise ValueError(f"expected {batch_size} adjacency matrices in the batch, got {adj.shape[0]}")
    return adj.detach().cpu().numpy()


def _intervention_labels(batch, batch_size: int, num_nodes: int) -> np.ndarray:
    """
    Per-graph {0, 1} node labels marking the intervened nodes, read off the intervention
    mask produced by `create_dataset_artifical`. Counterfactuals of the same graph are
    merged with a union, so a graph gets one label vector.
    """
    int_indices = batch.int_indices
    if int_indices.shape[-1] != num_nodes:
        raise ValueError(
            "group_by_intervention needs `int_indices` to be a {0, 1} mask over the "
            f"{num_nodes} nodes, but its last dimension is {int_indices.shape[-1]}"
        )
    masks = int_indices.reshape(batch_size, -1, num_nodes).amax(dim=1)
    return (masks != 0).long().detach().cpu().numpy()


def eval_mse_by_causal_graph(
    loader,
    forward_fn: Callable,
    device,
    target_indices: Optional[List[int]] = None,
    target_indice_map: Optional[torch.Tensor] = None,
    group_by_intervention: bool = False,
    wl_iterations: int = 3,
    exact_isomorphism: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Collect predictions over `loader` and report the MSE per causal-graph isomorphism
    class.
    """
    grouper = DAGIsomorphismGrouper(wl_iterations=wl_iterations, exact=exact_isomorphism)
    se_sums = defaultdict(float)
    counts = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_pred = forward_fn(batch)
            targets = batch.y.to(device)
            if targets.dim() != y_pred.dim():
                targets = targets.unsqueeze(1)

            preds_selected, targets_selected = select_target_entries(
                y_pred, targets, batch.int_indices, target_indices, target_indice_map
            )
            # One MSE per graph in the batch, so that class averages are plain means
            # over samples (every sample contributes the same number of entries).
            batch_size = preds_selected.shape[0]
            per_sample = ((preds_selected - targets_selected) ** 2).reshape(batch_size, -1).mean(dim=1)
            per_sample = per_sample.detach().cpu().numpy()

            adjacencies = _batch_adjacencies(batch, batch_size)
            labels = (
                _intervention_labels(batch, batch_size, adjacencies.shape[-1])
                if group_by_intervention
                else None
            )

            for i in range(batch_size):
                class_id = grouper.class_of(adjacencies[i], None if labels is None else labels[i])
                se_sums[class_id] += float(per_sample[i])
                counts[class_id] += 1

    per_graph = [
        {
            "class_id": class_id,
            "graph_hash": grouper.hashes[class_id],
            "num_samples": counts[class_id],
            "num_edges": int(grouper.representatives[class_id].sum()),
            "mse": se_sums[class_id] / counts[class_id],
            "adjacency": grouper.representatives[class_id].tolist(),
        }
        for class_id in range(len(grouper))
    ]
    per_graph.sort(key=lambda entry: entry["mse"], reverse=True)

    graph_mses = np.array([entry["mse"] for entry in per_graph], dtype=np.float64)
    num_samples = int(sum(counts.values()))
    results = {
        "num_graph_classes": len(per_graph),
        "num_samples": num_samples,
        "mean_mse": float(graph_mses.mean()) if graph_mses.size else float("nan"),
        # Spread across graphs, which is the quantity of interest here; undefined for a
        # single class, reported as 0.0 rather than nan so it stays JSON-serialisable.
        "std_mse": float(graph_mses.std(ddof=1)) if graph_mses.size > 1 else 0.0,
        "overall_mse": float(sum(se_sums.values()) / num_samples) if num_samples else float("nan"),
        "per_graph": per_graph,
    }

    if verbose:
        print(
            f"{results['num_graph_classes']} causal-graph classes over {num_samples} samples: "
            f"MSE {results['mean_mse']:.6f} +/- {results['std_mse']:.6f} "
            f"(overall {results['overall_mse']:.6f})"
        )
        for entry in per_graph[:10]:
            print(
                f"  {entry['graph_hash'][:8]}  edges={entry['num_edges']:3d}  "
                f"n={entry['num_samples']:4d}  mse={entry['mse']:.6f}"
            )

    return results
