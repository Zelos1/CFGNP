from typing import Optional, List
from pathlib import Path
import torch
from torch_geometric.data import Data
from cfgnp.models import CFSamplingModel
from cfgnp.util.util import mog_nll
from cfgnp.graph_approach.eval_by_causal_graph import select_target_entries
from cfgnp.graph_approach.causal_graph import CausalGraphFactory

def eval_nll_graph(loader, model, device, target_indices: list[int]=None, target_indice_map: torch.Tensor=None):
    model.eval()
    total_nll = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            # batch is a torch_geometric.data.Batch
            batch = batch.to(device)
            # call model with batch directly if it accepts Batch
            # otherwise construct inputs tuple:
            # inputs = (batch.x_int, batch.x_orig, batch.x_obs, batch.edge_index)
            preds = model(batch)  # model should return shape (B, C, N, K, 3, 3)
            targets = batch.y.to(device) # should be of shape (B, C, N, K)
            # if CUT_INDICES:
            #     targets = targets.reshape((preds.shape[0], -1, 3))[:, :preds.shape[2], :1]
            #     targets = targets.reshape((targets.shape[0], 1, targets.shape[1], targets.shape[2]))
            targets = targets.reshape((preds.shape[0], preds.shape[1], preds.shape[2], targets.shape[-1]))
            preds = preds.reshape((targets.shape[0], targets.shape[1], targets.shape[2], preds.shape[-3], preds.shape[-2], preds.shape[-1]))
            if target_indices is None and target_indice_map is None:
                nll = mog_nll(preds, targets)
            else:
                if target_indice_map is not None:
                    target_ind_mapped = target_indice_map.to(preds.device)[batch.int_indices.flatten()]
                    target_ind_mapped = target_ind_mapped.reshape((batch.int_indices.shape[0], preds.shape[1], -1))
                    preds_selected = torch.gather(preds, 2, target_ind_mapped.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, preds.shape[-3], preds.shape[-2], preds.shape[-1]))
                    targets_selected = torch.gather(targets, 2, target_ind_mapped.unsqueeze(-1).expand(-1, -1, -1, targets.shape[-1]))
                    nll = mog_nll(preds_selected, targets_selected)
                else:
                    nll = mog_nll(preds[:, :, target_indices], targets[:, :, target_indices])
            total_nll += nll.item()
            n_samples += batch.batch_size  # number of graphs in this batch
    mean_nll = total_nll / max(1, n_samples)
    return mean_nll


def fake_mog_output(value: torch.Tensor) -> torch.Tensor:
    """Reshape a plain (B, N) point prediction into the (B, 1, N, 1, K=3, 3) layout
    `eval_test_graph` expects from a real MoGLayer, so non-mixture baselines (BGM, CFM, VACA)
    can be scored through the same pipeline. The weight channel must sum to 1 across K so that
    `(means * weights).sum(-1)` reduces back to `value` instead of `K * value**2`.
    """
    mean = value.unsqueeze(-1).expand(-1, -1, 3)
    weight = torch.full_like(mean, 1.0 / 3.0)
    return torch.stack([mean, torch.zeros_like(mean), weight], dim=-1).unsqueeze(1).unsqueeze(3)


def eval_test_graph(loader, model, loss_fn, target_indices, target_indice_map, device, mc_sampling):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mc_sampling:
                model = CFSamplingModel(model)
                y_pred = model(batch)
            else:
                preds = model(batch)
                means = preds[..., 0]
                weights = preds[..., 2]
                y_pred = (means * weights).sum(dim=-1)  # (B,C,N)
            targets = batch.y.to(device)
            if len(targets.shape) != len(y_pred.shape):
                targets = targets.unsqueeze(1)
            preds_selected, targets_selected = select_target_entries(
                y_pred, targets, batch.int_indices, target_indices, target_indice_map
            )
            loss = loss_fn(preds_selected, targets_selected)
            total_loss += loss.item()
            n_samples += batch.batch_size
    mean_loss = total_loss / max(1, n_samples)
    return mean_loss



def create_dataset_artificial_graph(
    data,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    target_num_nodes = None
) -> List[Data]:
    """
    Create graph dataset with optional save/load cache.

    Args:
        data: Source data
        cache_path: Path to save/load dataset (.pt file)
        force_rebuild: If True, rebuild even if cache exists

    Returns:
        List[Data]
    """

    # Load cached dataset
    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_rebuild:
            print(f"Loading dataset from {cache_path}")
            return torch.load(cache_path, weights_only=False)

    dataset = []
    

    for i in range(len(data)):
        inputs, sample_dual = data[i]
        sample_int, int_indices, sample_orig, sample_obs = inputs[:4]
        causal_adj = inputs[4] if len(inputs) > 4 else None

        num_nodes = sample_int.shape[-2]
        cg_full = CausalGraphFactory.get_causal_graph("full")
        dmsm_structure = cg_full.dual_graph(num_nodes)
        if target_num_nodes is None:
            target_num_nodes = num_nodes

        data_point = Data(
            x_int=sample_int.reshape(
                (-1, num_nodes, sample_int.shape[-1])
            ).to(dtype=torch.float32),

            int_indices=int_indices.reshape(
                -1, int_indices.shape[-1]
            ).to(dtype=torch.long),

            x_orig=sample_orig.reshape(
                (-1, num_nodes, sample_int.shape[-1])
            ).to(dtype=torch.float32),

            x_obs=sample_obs.reshape(
                (-1, num_nodes, sample_int.shape[-1])
            ).to(dtype=torch.float32),

            y=sample_dual.reshape(
                (-1, target_num_nodes, sample_int.shape[-1])
            ).to(dtype=torch.float32),

            num_nodes=2 * num_nodes,
            edge_index=dmsm_structure
        )
        if causal_adj is not None:
            # (1, N, N) so that collation stacks one adjacency per graph in the batch.
            data_point.causal_adj = causal_adj.reshape(1, num_nodes, num_nodes).to(dtype=torch.long)

        dataset.append(data_point)

    # Save dataset
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(dataset, cache_path)
        print(f"Dataset saved to {cache_path}")

    return dataset


def create_classified_ood_graph_dataset(
    data,
    holdout_int_idx: int,
    keep_holdout: bool,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    target_num_nodes = None
) -> List[Data]:
    """
    Build a graph dataset for classified CheXpert samples and optionally keep only
    the held-out intervention target (OOD) or exclude it (ID).
    """

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_rebuild:
            return torch.load(cache_path, weights_only=False)

    dataset: List[Data] = []
    for i in range(len(data)):
        (sample_int, int_indices, sample_orig, sample_obs), sample_y = data[i]

        idx = int(int_indices.reshape(-1)[0].item())
        is_holdout = idx == holdout_int_idx
        if keep_holdout != is_holdout:
            continue

        num_nodes = sample_int.shape[-2]
        cg_full = CausalGraphFactory.get_causal_graph("full")
        dmsm_structure = cg_full.dual_graph(num_nodes)
        if target_num_nodes is None:
            target_num_nodes = num_nodes

        dataset.append(
            Data(
                x_int=sample_int.reshape((-1, num_nodes, sample_int.shape[-1])).to(torch.float32),
                int_indices=int_indices.reshape(-1, int_indices.shape[-1]).to(torch.long),
                x_orig=sample_orig.reshape((-1, num_nodes, sample_int.shape[-1])).to(torch.float32),
                x_obs=sample_obs.reshape((-1, num_nodes, sample_int.shape[-1])).to(torch.float32),
                y=sample_y.reshape((-1, target_num_nodes, sample_int.shape[-1])).to(torch.float32),
                num_nodes=2 * num_nodes,
                edge_index=dmsm_structure,
            )
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, cache_path)

    return dataset


def _masked_node_view(source, shape, node_idx, mask_value, cache=None):
    if cache is not None:
        cached = cache.get(source.data_ptr())
        if cached is not None:
            return cached

    out = source.reshape(shape).to(torch.float32)
    if node_idx is not None:
        if out.data_ptr() == source.data_ptr():
            out = out.clone()
        out[:, node_idx, :] = mask_value

    if cache is not None:
        cache[source.data_ptr()] = out
    return out


def create_classified_ood_masked_graph_dataset(
    data,
    holdout_int_idx: int,
    keep_holdout: bool,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    target_num_nodes = None,
    mask_value: float = 0.0,
    mask_target: bool = True,
) -> List[Data]:
    """
    Same split logic as `create_classified_ood_graph_dataset`, but additionally masks
    out the node at `holdout_int_idx` so the model never sees that attribute's value.

    The mask is applied to `x_int`, `x_orig` and `x_obs` (and to `y` when
    `mask_target=True`, provided the node is still inside the truncated target).
    """

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_rebuild:
            return torch.load(cache_path, weights_only=False)

    dataset: List[Data] = []
    dmsm_structure = None  # built once and shared: identical for every sample
    feat_shape = None
    y_shape = None
    mask_node = None
    mask_y_node = None
    # x_orig/x_obs are the same tensor object across every counterfactual variant of a
    # given row
    shared_field_cache = {}

    for i in range(len(data)):
        (sample_int, int_indices, sample_orig, sample_obs), sample_y = data[i]

        # Cheapest possible test first, so filtered-out samples cost nothing else.
        if keep_holdout != (int_indices.reshape(-1)[0].item() == holdout_int_idx):
            continue

        if dmsm_structure is None:
            num_nodes = sample_int.shape[-2]
            num_feat = sample_int.shape[-1]
            if target_num_nodes is None:
                target_num_nodes = num_nodes
            cg_full = CausalGraphFactory.get_causal_graph("full")
            dmsm_structure = cg_full.dual_graph(num_nodes)
            feat_shape = (-1, num_nodes, num_feat)
            y_shape = (-1, target_num_nodes, num_feat)
            in_range = 0 <= holdout_int_idx < num_nodes
            mask_node = holdout_int_idx if in_range else None
            mask_y_node = (
                holdout_int_idx
                if mask_target and 0 <= holdout_int_idx < target_num_nodes
                else None
            )

        dataset.append(
            Data(
                x_int=_masked_node_view(sample_int, feat_shape, mask_node, mask_value),
                int_indices=int_indices.reshape(-1, int_indices.shape[-1]).to(torch.long),
                x_orig=_masked_node_view(sample_orig, feat_shape, mask_node, mask_value, cache=shared_field_cache),
                x_obs=_masked_node_view(sample_obs, feat_shape, mask_node, mask_value, cache=shared_field_cache),
                y=_masked_node_view(sample_y, y_shape, mask_y_node, mask_value),
                num_nodes=2 * num_nodes,
                edge_index=dmsm_structure,
            )
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, cache_path)

    return dataset

def _dropped_node_view(source, mask, cache=None):
    """
    Slice out the held-out node along dim 1 of `source`, memoizing by `source`'s underlying
    storage like `_masked_node_view` above: several samples (e.g. every counterfactual
    variant of one CheXpert row) are built from `x_orig.to(dtype).unsqueeze(0).unsqueeze(-1)`,
    which re-wraps the *same* storage in a fresh Tensor object each time it runs — different
    `id(source)` per variant even though `data_ptr()` is identical — so a cache keyed by
    `id()` never hits. Keying by `data_ptr()` catches the real sharing, and torch.save
    dedupes storages by identity, so slicing each one independently would multiply the
    cached file size by the number of variants per row instead of reusing the one shared copy.
    """
    if cache is not None:
        cached = cache.get(source.data_ptr())
        if cached is not None:
            return cached

    out = source[:, mask].to(torch.float32)

    if cache is not None:
        cache[source.data_ptr()] = out
    return out


def create_chexpert_reduced_ood_dataset(
    data,
    holdout_int_idx: int,
    keep_holdout: bool,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    target_num_nodes = None,
    mask_value: float = 0.0,
    mask_target: bool = True,
) -> List[Data]:
    """
    The node at
    `holdout_int_idx` is *removed* from `x_int`, `x_orig`, `x_obs` and `y`.
    """

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_rebuild:
            return torch.load(cache_path, weights_only=False)

    if len(data) == 0:
        raise ValueError(
            f"No samples to build {cache_path} from. The raw counterfactuals were skipped "
            "because the other cache files exist -- delete them too, or point this split at "
            "the cache that was built alongside them."
        )

    dataset: List[Data] = []
    dmsm_structure = None  # built once and shared: identical for every sample
    mask = None
    # x_orig/x_obs are the same tensor object across every counterfactual variant of a
    # given row
    shared_field_cache = {}

    for i in range(len(data)):
        (sample_int, int_indices, sample_orig, sample_obs), sample_y = data[i]

        int_indices = int_indices.reshape(-1, int_indices.shape[-1]).to(torch.long)
        if keep_holdout != bool((int_indices == holdout_int_idx).any()):
            continue

        if keep_holdout:
            # No node is left to mark as intervened; see the docstring.
            int_indices = torch.full_like(int_indices, -1)
        else:
            # Removing node `holdout_int_idx` renumbers every node above it, and `int_indices`
            # is what indexes `target_indice_map`'s rows downstream, so the surviving indices
            # shift down by one.
            int_indices = int_indices - (int_indices > holdout_int_idx).long()

        if dmsm_structure is None:
            num_nodes = sample_int.shape[-2]
            cg_full = CausalGraphFactory.get_causal_graph("full")
            dmsm_structure = cg_full.dual_graph(num_nodes)
            mask = torch.arange(sample_int.size(1), device=sample_int.device) != holdout_int_idx

        dataset.append(
            Data(
                x_int=sample_int[:, mask].to(torch.float32),
                int_indices=int_indices,
                x_orig=_dropped_node_view(sample_orig, mask, cache=shared_field_cache),
                x_obs=_dropped_node_view(sample_obs, mask, cache=shared_field_cache),
                y=sample_y[:, torch.arange(sample_y.size(1), device=sample_y.device) != holdout_int_idx].to(torch.float32),
                num_nodes=2 * num_nodes,
                edge_index=dmsm_structure,
            )
        )

    if not dataset:
        raise ValueError(
            f"None of the {len(data)} input samples "
            f"{'intervene on' if keep_holdout else 'intervene on anything other than'} "
            f"index {holdout_int_idx}, so this split would be empty"
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, cache_path)

    return dataset