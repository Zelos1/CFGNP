import numpy as np
from typing import Optional
import networkx as nx
import torch
import os
import re
import math

IMAGE_RESIZE = 128

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE
# DEVICE = torch.device("cpu")

CUT_INDICES = True
REGRESSION_TARGET_INDICES = [1]
NUM_CLASS_CHEXPERT = {
    "Sex": 2,
    "Frontal/Lateral": 2, #1, # Should probably not be included
    "Age": 100,  # Not used anyways
    "AP/PA": 2,
    "No Finding": 2,
}

CLASS_MAPPING = {
    "Sex": 0,
    # "Frontal/Lateral": 1,
    "Age": 1,
    "AP/PA": 2,
    "No Finding": 3,
}


def is_regression_target(target_index: int) -> bool:
    return target_index in REGRESSION_TARGET_INDICES


def map_int_indices(target_indice_map: torch.Tensor, int_indices: torch.Tensor, batch_rows: int,
                    device=None) -> torch.Tensor:
    """Rows of `target_indice_map` holding the supervised targets for each sample's intervention.

    The range check runs on the host: an out-of-range `int_indices` otherwise only shows up as
    an asynchronous `vectorized_gather_kernel` device assert, attributed to whatever CUDA call
    happens to synchronise next and naming neither tensor. The usual cause is a dataset that
    drops a target node without renumbering its intervention indices to match the map.

    Negative indices are the "no intervened node" sentinel of OOD splits whose intervened
    attribute was dropped from the graph. They select from the end of the map by the usual
    convention; maps used with them carry a single all-targets row, so which end is picked
    makes no difference.
    """
    if device is None:
        device = int_indices.device
    target_indice_map = target_indice_map.to(device=device, dtype=torch.long)
    int_indices = int_indices.to(device=device, dtype=torch.long)
    rows = target_indice_map.shape[0]

    if int_indices.numel() > 0:
        lowest, highest = int(int_indices.min()), int(int_indices.max())
        if lowest < -rows or highest >= rows:
            raise IndexError(
                f"int_indices span [{lowest}, {highest}], which does not fit a "
                f"target_indice_map of {rows} rows"
            )

    return target_indice_map[int_indices].reshape(batch_rows, -1)


def intervention_nll_mog(preds: torch.Tensor, target:torch.Tensor, target_indices: list[int], target_indice_map: torch.Tensor,
                         int_indices: torch.Tensor, eps: float=1e-6, var_pen: float=1e-2, int_pen:float=0.5):
    if target_indice_map is not None:
        target_ind_mapped_ext = map_int_indices(target_indice_map, int_indices, preds.shape[0], device=preds.device)
        preds_selected = torch.gather(preds, 1, target_ind_mapped_ext.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, preds.shape[-2], preds.shape[-1]))
        targets_selected = torch.gather(target, 1, target_ind_mapped_ext.unsqueeze(-1).expand(-1, -1, target.shape[-1]))
    else:
        preds_selected = preds[:,:, target_indices]
        targets_selected = target[:,:, target_indices]

    mog_loss = mog_nll(preds_selected, targets_selected)
    # var_term = preds_selected[..., 1].sum(dim=-1).mean() * var_pen
    if target.dim() == preds.dim() - 2:
        target = target.unsqueeze(-1)

    int_pred = (preds[int_indices][..., 0] * preds[int_indices][..., 2] - target[int_indices])
    intervention_term = int_pred.norm(dim=-1).sum(dim=-1).mean() * int_pen
    return mog_loss + intervention_term, mog_loss

    

def compute_indiced_loss(preds: torch.Tensor, targets: torch.Tensor, target_indices: list[int], target_indice_map: torch.Tensor,
                         int_indices: torch.Tensor, loss_fn, needs_indices=False, loss_extras=None) -> torch.Tensor:
    # `loss_extras` carries arguments that are not indexed by node (the decoded image of the
    # vision models). They only enter the optimised loss, so the reported targeted loss stays
    # a pure per-target number and remains comparable across configurations.
    loss_extras = loss_extras or {}
    if loss_fn == intervention_nll_mog:
        return intervention_nll_mog(preds, targets, target_indices, target_indice_map, int_indices)
    if target_indice_map is not None:
        target_ind_mapped_ext = map_int_indices(target_indice_map, int_indices, preds.shape[0], device=preds.device)
        target_ind_mapped = target_ind_mapped_ext
        if int_indices.numel() > 0 and int(int_indices.min()) < 0:
            # The intervened node is gathered alongside the mapped targets below, and gather
            # takes no negative index, so the "no intervened node" sentinel cannot be trained
            # on -- those splits are evaluated through `eval_dloader_image_loss` instead.
            raise ValueError("compute_indiced_loss cannot take the 'no intervened node' sentinel (-1)")
        target_ind_mapped_ext = torch.concat([target_ind_mapped_ext
                                              , int_indices.to(device=preds.device).reshape(preds.shape[0], -1)], dim=-1)
        preds_selected = torch.gather(preds, 1, target_ind_mapped_ext.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, preds.shape[-3], preds.shape[-2], preds.shape[-1]))
        targets_selected = torch.gather(targets, 1, target_ind_mapped_ext.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))
        if needs_indices:
            loss = loss_fn(preds_selected, targets_selected, target_indices=target_ind_mapped_ext, **loss_extras)
        else:
            loss = loss_fn(preds_selected, targets_selected, **loss_extras)

        preds_selected_t_indices = torch.gather(preds, 1, target_ind_mapped.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, preds.shape[-3], preds.shape[-2], preds.shape[-1]))
        targets_selected_t_indices = torch.gather(targets, 1, target_ind_mapped.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))
        if needs_indices:
            test_targeted_loss = loss_fn(preds_selected_t_indices, targets_selected_t_indices, target_indices=target_ind_mapped)
        else:
            test_targeted_loss = loss_fn(preds_selected_t_indices, targets_selected_t_indices)
    else:
        loss = loss_fn(preds[:,target_indices], targets[:,target_indices], **loss_extras)
        test_targeted_loss = loss
    return loss, test_targeted_loss

def mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / pred).abs().sum()

MOG_VAR_FLOOR = 1e-4


def set_mog_var_floor(value: float) -> None:
    """Raise the variance floor of `mog_nll` for this process.

    The NLL rewards shrinking a component's variance around a point it has memorised
    (`-0.5*log(var)` grows without bound), so a model can buy train loss it cannot generalise.
    The floor caps that reward at `-0.5*log(floor)` nats per point: the 1e-4 default allows
    ~4.6, a floor of 1e-2 (sigma >= 0.1) allows ~1.4.

    Process-global because `mog_nll` is reached through several call paths that carry no
    config, and each experiment runs in its own process. It applies to training and validation
    alike, which is what keeps the two comparable -- the floor is a modelling assumption about
    minimum aleatoric noise, not a training trick.
    """
    global MOG_VAR_FLOOR
    MOG_VAR_FLOOR = value


def mog_nll(pred: torch.Tensor, target: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    # unpack
    eps = MOG_VAR_FLOOR if eps is None else eps
    mean = pred[..., 0]       # shape (B, C, N, K, mog)
    var = pred[..., 1]    # shape (B, C, N, K, mog)
    weights = pred[..., 2]     # shape (B, C, N, K, mog)

    # ensure target broadcastable to (B, C, N, K)
    if target.dim() < mean.dim():
        t = target.unsqueeze(-1)  # (B, C, N, 1)
    else:
        t = target
    sq = (t - mean) ** 2

    var = var.clamp(min=eps)
    # Kept off `eps` so raising the variance floor does not also change which mixture weights
    # are treated as zero; this clamp only exists to keep `log` finite.
    weights = weights.clamp(min=1e-4)

    log_gauss = -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + sq / var)

    log_w = torch.log(weights)                # (..., K)
    comp_log = log_w + log_gauss              # (..., K)
    log_prob = torch.logsumexp(comp_log, dim=-1)

    nll = log_prob  # negative log-likelihood per (B,C,N)
    return -nll.sum()

class DAG:
    def __init__(self, num_nodes: int, p_edge: float = 0.2, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

        # --- Step 1: random topological order ---
        self.topo_order = list(range(num_nodes))
        self.rng.shuffle(self.topo_order)
        self.order_index = {n: i for i, n in enumerate(self.topo_order)}

        # --- Step 2: ER directed graph ---
        # This may have cycles
        G = nx.gnp_random_graph(
            n=num_nodes,
            p=p_edge,
            seed=seed,
            directed=True
        )

        # --- Step 3: keep only forward edges to enforce DAG ---
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(num_nodes))
        for u, v in G.edges():
            # keep edge u -> v only if u comes before v in topo_order
            if self.order_index[u] < self.order_index[v]:
                self.graph.add_edge(u, v)

        assert nx.is_directed_acyclic_graph(self.graph)


def get_py_files(directory):
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != 'merge_py_files.py' and file != "main.py":
                py_files.append(os.path.join(root, file))
    return py_files

def get_module_names(py_files):
    return {os.path.splitext(os.path.basename(f))[0] for f in py_files}

def remove_local_imports(code, local_modules):
    # Remove lines like: import mymodule or from mymodule import ...
    pattern = re.compile(r'^\s*(from|import)\s+([a-zA-Z0-9_\.]+)')
    lines = code.splitlines()
    filtered = []
    for line in lines:
        found = False
        match = pattern.match(line)
        if match:
            mod = match.group(2).split('.')
            for m_name in mod:
                if m_name in local_modules:
                    found = True
        if found:
            continue
        if str.startswith(line, "if __name__ =="):
            break
        filtered.append(line)
    return '\n'.join(filtered)

def combine_scripts():
    directory = './src'
    output_file = './merged.py'
    py_files = get_py_files(directory)
    local_modules = get_module_names(py_files)
    print(local_modules)

    merged_code = []
    for file in py_files:
        with open(file, 'r') as f:
            code = f.read()
            code = remove_local_imports(code, local_modules)
            merged_code.append(f'# --- {file} ---\n{code}\n')

    with open(output_file, 'w') as out:
        out.write('\n'.join(merged_code))

    print(f'Merged {len(py_files)} files into {output_file}')


# def get_observation_graph_structure(num_nodes, num_obs):
#     """
#     This returns the graph structure of the num_obs observations but the edges of the exogenous variables are reversed. The edges are only for one batch
#     """
#     single_node_set = np.arange(num_nodes)

#     single_graph = np.stack([single_node_set.repeat(num_nodes), single_node_set.reshape((1, -1)).repeat(num_nodes, axis=0).flatten()], axis=0)

#     obs_graph = single_graph.repeat(num_obs, axis=0).reshape(2, -1)
#     # Disconnected obs graphs. Obs graphs are fully connected
#     all_obs = obs_graph + np.arange(num_obs).repeat(single_graph.shape[1]) * num_nodes

#     # TODO how to initialize the connecting indices? or maybe leave them out
#     # The connecting indices are a copy of the observation graph that combines information of all observations. This copy does not have any edges
#     connecting_indices = all_obs.max() + 1 + np.arange(num_nodes).reshape((1, -1)).repeat(num_obs, axis=0).flatten()
#     exogenous_connection = np.stack([np.arange(all_obs.max() + 1), connecting_indices])
    

#     # graph_z = (1 + connecting_indices.max()) + single_graph
#     z_connection = np.stack([np.arange(num_nodes) + (1 + connecting_indices.max()), connecting_indices[:num_nodes]])

#     # entire_graph = np.concat([all_obs, exogenous_connection, graph_z, z_connection], axis=1)
#     entire_graph = np.concat([all_obs, exogenous_connection], axis=1)
#     return torch.from_numpy(entire_graph)