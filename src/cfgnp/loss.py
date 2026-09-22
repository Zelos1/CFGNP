import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from cfgnp.util.util import DEVICE, map_int_indices
from cfgnp.models.sampling_model import CFSamplingModel

import lpips


def mog_sampling_mse_loss(pred: torch.Tensor, target: torch.Tensor, mc_samples: int = 100, temperature: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    y_pred = CFSamplingModel.point_estimate(pred, mc_samples=mc_samples, temperature=temperature, eps=eps, deterministic=False)
    return F.mse_loss(y_pred, target, reduction="sum")


class MultiTargetCrossEntropy(nn.Module):
    def __init__(self, class_counts, reduction="sum"):
        super().__init__()
        self.class_counts = list(class_counts)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, T, C_max]
            targets: [B, T]

        Returns:
            scalar loss
        """
        B, T, _ = logits.shape

        if T != len(self.class_counts):
            raise ValueError(
                f"Expected {len(self.class_counts)} targets, got {T}"
            )

        total_loss = 0.0

        for t, c in enumerate(self.class_counts):
            logits_t = logits[:, t, :c]   # [B, c]
            targets_t = targets[:, t]     # [B]

            total_loss += F.cross_entropy(
                logits_t,
                targets_t.long(),
                reduction="sum",
            )

        if self.reduction == "mean":
            total_loss = total_loss / B

        return total_loss


class MixedTargetLoss(nn.Module):
    def __init__(self, class_counts, regression_target_indices=None, reduction="sum"):
        """
        Args:
            class_counts: list[int] (num classes per target)
            regression_target_indices: list[int] indices that should use MSE
            reduction: 'sum' or 'mean'
        """
        super().__init__()
        self.class_counts = list(class_counts)
        self.reduction = reduction
        self.regression_target_indices = set(regression_target_indices or [])

    def forward(self, logits, targets):
        logits = logits.reshape((logits.shape[0], logits.shape[1], -1))  # [B, T, C_max]
        B, T, _ = logits.shape

        if T != len(self.class_counts):
            raise ValueError(
                f"Expected {len(self.class_counts)} targets, got {T}"
            )

        total_loss = 0.0

        for t, c in enumerate(self.class_counts):
            logits_t = logits[:, t, :c]
            targets_t = targets[:, t]
            if t in self.regression_target_indices:
                predicted = logits_t[:, 0]
                total_loss += F.mse_loss(predicted, targets_t.float(), reduction="sum")
            else:
                total_loss += F.cross_entropy(
                    logits_t,
                    targets_t.flatten().long(),
                    reduction="sum",
                )

        if self.reduction == "mean":
            total_loss = total_loss / B

        return total_loss

class ReorderedMixedTargetLoss(nn.Module):
    def __init__(self, class_counts, regression_target_indices=None, reduction="sum"):
        """
        Args:
            class_counts: list[int] (num classes per target)
            regression_target_indices: list[int] indices that should use MSE
            reduction: 'sum' or 'mean'
        """
        super().__init__()
        self.class_counts = torch.tensor(list(class_counts))
        self.reduction = reduction
        self.regression_target_indices = set(regression_target_indices or [])
        self.needs_indices = True

    def forward(self, logits, targets, target_indices=None):
        logits = logits.reshape(logits.shape[0], logits.shape[1], -1)  # [B, K, Cmax]

        if target_indices is None:
            total_loss = logits.new_tensor(0.0)
            for t, c in enumerate(self.class_counts):
                logits_t = logits[:, t, :c]
                targets_t = targets[:, t]

                if t in self.regression_target_indices:
                    total_loss = total_loss + F.mse_loss(
                        logits_t[:, 0], targets_t.float(), reduction=self.reduction
                    )
                else:
                    total_loss = total_loss + F.cross_entropy(
                        logits_t, targets_t.long().flatten(), reduction=self.reduction
                    )
        else:
            B, K, Cmax = logits.shape
            if target_indices.shape != (B, K):
                raise ValueError("target_indices must have shape [B, K]")

            # Flatten packed positions
            class_ids = target_indices.reshape(-1)          # [B*K]
            targets_flat = targets.reshape(-1)              # [B*K]
            logits_flat = logits.reshape((-1, Cmax))          # [B*K, Cmax]

            total_loss = logits.new_tensor(0.0)

            for t, c in enumerate(self.class_counts):
                mask = class_ids == t
                if not mask.any():
                    continue

                logits_t = logits_flat[mask, :c]
                targets_t = targets_flat[mask]

                if t in self.regression_target_indices:
                    total_loss = total_loss + F.mse_loss(
                        logits_t, targets_t.float().unsqueeze(-1), reduction=self.reduction
                    ) / 8
                else:
                    total_loss = total_loss + F.cross_entropy(
                        logits_t, targets_t.long(), reduction=self.reduction
                    )


        if self.reduction == "mean":
            total_loss = total_loss / class_ids.nunique()

        return total_loss


class ReconstructionRegularizedLoss(nn.Module):

    needs_images = True

    def __init__(self, base_loss: nn.Module, weight: float = 10.0, lpips_weight: float=1.0, tv_weight: float = 1.0):
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.tv_weight = tv_weight
        self.lpips_weight = lpips_weight
        self.needs_indices = bool(getattr(base_loss, "needs_indices", False))

        self.lpips_model = lpips.LPIPS(net='vgg').eval()

        # Freeze LPIPS parameters so we don't accidentally train the VGG network
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def total_variation(self, image):
        diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
        diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
        return diff_h.flatten(1).mean(dim=1) + diff_w.flatten(1).mean(dim=1)

    def reconstruction_loss(self, pred_image, target_image):
        error = F.mse_loss(pred_image, target_image.to(pred_image.dtype), reduction="none")

        if next(self.lpips_model.parameters()).device != pred_image.device:
            self.lpips_model = self.lpips_model.to(pred_image.device)
            self.lpips_model.eval()

        perceptual_error = self.lpips_model(pred_image, target_image)
        lpips_loss = perceptual_error.sum()

        tv_loss = self.total_variation(pred_image).sum()

        return error.flatten(1).mean(dim=1).sum() + lpips_loss * self.lpips_weight + self.tv_weight * tv_loss

    def forward(self, logits, targets, target_indices=None, pred_image=None, target_image=None):
        base_kwargs = {"target_indices": target_indices} if self.needs_indices else {}
        loss = self.base_loss(logits, targets, **base_kwargs)

        if pred_image is None or target_image is None:
            return loss

        return loss + self.weight * self.reconstruction_loss(pred_image, target_image)


def image_loss_kwargs(loss_fn, pred_image, target_image):
    if pred_image is None or not getattr(loss_fn, "needs_images", False):
        return {}
    return {"pred_image": pred_image, "target_image": target_image}


def compute_per_index_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_indices: torch.Tensor,
    class_counts: torch.Tensor,
    regression_target_indices: set,
):
    """Compute metrics for each original target index.

    Args:
        logits:
            Tensor of shape [N, K, Cmax].
        targets:
            Tensor of shape [N, K].
        target_indices:
            Tensor of shape [N, K], mapping each selected position to its
            original target index.
        class_counts:
            Tensor of shape [T], containing the number of classes for each
            original target.
        regression_target_indices:
            Set of original target indices that represent regression tasks.

    Returns:
        dict[int, dict[str, float | None]]:
            Metrics grouped by original target index.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [N, K, Cmax], got {tuple(logits.shape)}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"targets must have shape [N, K], got {tuple(targets.shape)}"
        )

    if target_indices.ndim != 2:
        raise ValueError(
            "target_indices must have shape [N, K], "
            f"got {tuple(target_indices.shape)}"
        )

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "The first two dimensions of logits must match targets: "
            f"{tuple(logits.shape[:2])} != {tuple(targets.shape)}"
        )

    if targets.shape != target_indices.shape:
        raise ValueError(
            "targets and target_indices must have the same shape: "
            f"{tuple(targets.shape)} != {tuple(target_indices.shape)}"
        )

    # Move everything to CPU once.
    logits_np = logits.detach().float().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    indices_np = target_indices.detach().long().cpu().numpy()
    class_counts_np = np.array(class_counts)

    metrics = {}

    for orig_idx_value in np.unique(indices_np):
        orig_idx = int(orig_idx_value)

        if orig_idx < 0 or orig_idx >= len(class_counts_np):
            raise IndexError(
                f"Target index {orig_idx} is outside class_counts, "
                f"which has length {len(class_counts_np)}"
            )

        mask = indices_np == orig_idx

        # Boolean indexing converts [N, K, Cmax] to [M, Cmax] and
        # [N, K] to [M], where M is the number of observations for this target.
        logits_orig = logits_np[mask]
        targets_orig = targets_np[mask]

        if targets_orig.size == 0:
            continue

        if orig_idx in regression_target_indices:
            if logits_orig.ndim != 2 or logits_orig.shape[1] < 1:
                raise ValueError(
                    f"Regression target {orig_idx} must have at least one "
                    "logit/output value"
                )

            y_true = targets_orig.astype(np.float64)
            y_pred = logits_orig[:, 0].astype(np.float64)

            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            mae = (
                float(mean_absolute_error(y_true, y_pred))
                if y_true.size > 0
                else float("nan")
            )

            metrics[orig_idx] = {
                "mae": mae,
                "accuracy": None,
                "roc_auc": None,
            }
            continue

        num_classes = int(class_counts_np[orig_idx])

        if num_classes < 2:
            raise ValueError(
                f"Classification target {orig_idx} must have at least "
                f"2 classes, got {num_classes}"
            )

        if logits_orig.shape[1] < num_classes:
            raise ValueError(
                f"Logits for target {orig_idx} contain "
                f"{logits_orig.shape[1]} outputs, but {num_classes} "
                "classes are required"
            )

        y_true = targets_orig.astype(np.int64)

        valid_target_mask = (
            np.isfinite(targets_orig)
            & (y_true >= 0)
            & (y_true < num_classes)
        )

        y_true = y_true[valid_target_mask]
        logits_valid = logits_orig[valid_target_mask, :num_classes]

        if y_true.size == 0:
            metrics[orig_idx] = {
                "mae": None,
                "accuracy": float("nan"),
                "roc_auc": float("nan"),
            }
            continue

        # Stable softmax.
        shifted_logits = logits_valid - np.max(
            logits_valid,
            axis=1,
            keepdims=True,
        )
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        y_pred = np.argmax(probs, axis=1)
        accuracy = float(accuracy_score(y_true, y_pred))

        present_classes = np.unique(y_true)

        if num_classes == 2:
            if present_classes.size < 2:
                roc_auc = float("nan")
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        probs[:, 1],
                    )
                )
        else:
            if present_classes.size < num_classes:
                roc_auc = float("nan")
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        probs,
                        labels=np.arange(num_classes),
                        multi_class="ovr",
                        average="macro",
                    )
                )

        metrics[orig_idx] = {
            "mae": None,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

    return metrics


def compute_per_index_metrics_loader(
    loader,
    model,
    class_counts: torch.Tensor,
    regression_target_indices: set,
    target_indice_map: torch.Tensor,
    device=None,
):
    if device is None:
        device = DEVICE

    was_training = model.training
    model.eval()

    all_logits = []
    all_targets = []
    all_target_indices = []

    target_indice_map_device = target_indice_map.to(
        device=device,
        dtype=torch.long,
    )

    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                preds = model(batch)
                if isinstance(preds, tuple):
                    # Image models return (logits, decoded_image); metrics only need logits.
                    preds = preds[0]
                targets = batch.y

                if preds.ndim < 3:
                    raise ValueError(
                        "Model predictions must have at least three dimensions "
                        "before reshaping; expected a target and class dimension, "
                        f"got shape {tuple(preds.shape)}"
                    )

                # Preserve the original reshaping logic.
                targets = targets.reshape(
                    -1,
                    preds.shape[2],
                    targets.shape[-1],
                )
                preds = preds.reshape(
                    targets.shape[0],
                    preds.shape[2],
                    -1,
                )

                mapped_indices = map_int_indices(
                    target_indice_map_device,
                    batch.int_indices,
                    preds.shape[0],
                    device=preds.device,
                )

                if mapped_indices.numel() > 0:
                    min_index = int(mapped_indices.min().item())
                    max_index = int(mapped_indices.max().item())

                    if min_index < 0 or max_index >= preds.shape[1]:
                        raise IndexError(
                            "Mapped target indices are outside the prediction "
                            f"target dimension [0, {preds.shape[1] - 1}]. "
                            f"Observed range: [{min_index}, {max_index}]"
                        )

                preds_selected = torch.gather(
                    preds,
                    dim=1,
                    index=mapped_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        preds.shape[-1],
                    ),
                )

                targets_selected = torch.gather(
                    targets,
                    dim=1,
                    index=mapped_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        targets.shape[-1],
                    ),
                )

                # compute_per_index_metrics expects scalar targets [N, K].
                if targets_selected.shape[-1] != 1:
                    raise ValueError(
                        "Each selected target must contain exactly one scalar "
                        "value. Received targets_selected with shape "
                        f"{tuple(targets_selected.shape)}"
                    )

                targets_selected = targets_selected.squeeze(-1)

                all_logits.append(preds_selected.detach().cpu())
                all_targets.append(targets_selected.detach().cpu())
                all_target_indices.append(mapped_indices.detach().cpu())

    finally:
        model.train(was_training)

    if not all_logits:
        return {}

    accumulated_logits = torch.cat(all_logits, dim=0)
    accumulated_targets = torch.cat(all_targets, dim=0)
    accumulated_target_indices = torch.cat(all_target_indices, dim=0)

    return compute_per_index_metrics(
        logits=accumulated_logits,
        targets=accumulated_targets,
        target_indices=accumulated_target_indices,
        class_counts=class_counts,
        regression_target_indices=regression_target_indices,
    )


def eval_dloader_loss(loss_fn, loader, model, target_indice_map=None, device=None):
    if device is None:
        device = DEVICE

    n_samples = 0
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        preds = model(batch)
        targets = batch.y.to(device)

        targets = targets.reshape((-1, preds.shape[2], targets.shape[-1]))  # (B * C, N, feat)
        preds = preds.reshape((targets.shape[0], preds.shape[2], -1))

        if target_indice_map is not None:
            target_ind_mapped = map_int_indices(target_indice_map, batch.int_indices, preds.shape[0], device=preds.device)

            preds_selected_t_indices = torch.gather(preds, 1, target_ind_mapped.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
            targets_selected_t_indices = torch.gather(targets, 1, target_ind_mapped.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))

            loss = loss_fn(
                preds_selected_t_indices,
                targets_selected_t_indices,
                target_indices=target_ind_mapped
            )
        else:
            loss = loss_fn(preds, targets)

        total_loss += loss.item()
        n_samples += targets.shape[0]

    return total_loss / n_samples


def eval_dloader_image_loss(loss_fn, loader, model, recon_target_fn, target_indice_map=None, device=None):
    """`eval_dloader_loss` for models that also return a decoded image.
    """
    if device is None:
        device = DEVICE

    n_samples = 0
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        preds, pred_image = model(batch)
        targets = batch.y.to(device)
        extras = image_loss_kwargs(loss_fn, pred_image, recon_target_fn(batch))

        targets = targets.reshape((-1, preds.shape[2], targets.shape[-1]))  # (B * C, N, feat)
        preds = preds.reshape((targets.shape[0], preds.shape[2], -1))

        if target_indice_map is not None:
            target_ind_mapped = map_int_indices(target_indice_map, batch.int_indices, preds.shape[0], device=preds.device)

            preds_selected_t_indices = torch.gather(preds, 1, target_ind_mapped.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
            targets_selected_t_indices = torch.gather(targets, 1, target_ind_mapped.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))

            loss = loss_fn(
                preds_selected_t_indices,
                targets_selected_t_indices,
                target_indices=target_ind_mapped,
                **extras,
            )
        else:
            loss = loss_fn(preds, targets, **extras)

        total_loss += loss.item()
        n_samples += targets.shape[0]

    return total_loss / n_samples
