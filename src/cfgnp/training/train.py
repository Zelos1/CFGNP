import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cfgnp.util.data_paths import ChexpertPath
from cfgnp.util.util import compute_indiced_loss, intervention_nll_mog, mog_nll
from cfgnp.train import eval_nll, eval_test
from cfgnp.graph_approach.train_graph import eval_nll_graph, eval_test_graph
from cfgnp.loss import eval_dloader_image_loss, eval_dloader_loss, image_loss_kwargs


@dataclass
class EpochStatistics:
    loss_sum: float
    targeted_loss_sum: float
    sample_count: int


@dataclass
class PreparedTraining:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader


def state_path_for(save_path: str) -> str:
    """Path of the resumable per-epoch state, kept next to the best-model checkpoint."""
    root, ext = os.path.splitext(save_path)
    return f"{root}_last_state{ext or '.pt'}"


class CounterfactualTrainingTemplate(ABC):
    """Template Method: shared training flow with runtime-specific hooks."""

    _reported_batch = False

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    def rank_label(self) -> str:
        return "[single device]"

    def _report_batch_once(self, sample_count: int) -> None:
        """Print the batch each process actually receives, so a broken shard is visible in the log."""
        if self._reported_batch:
            return
        self._reported_batch = True
        print(f"{self.rank_label} per-device train batch: {sample_count} samples", flush=True)

    @abstractmethod
    def prepare_training(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader) -> PreparedTraining:
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize_epoch_statistics(self, statistics: EpochStatistics) -> EpochStatistics:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, model: nn.Module, save_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def restore_checkpoint(self, model: nn.Module, save_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_training_state(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, progress: dict, state_path: str) -> None:
        """Persist the *current* (not best) model + optimizer + scheduler + progress so training can resume."""
        raise NotImplementedError

    @abstractmethod
    def load_training_state(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, state_path: str) -> Optional[dict]:
        """Load a state written by `save_training_state` in place; return its progress dict, or None if absent."""
        raise NotImplementedError

    @abstractmethod
    def report(self, message: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize_validation_values(self, val_loss: float, test_val_loss: float) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def finalize_test_value(self, test_loss: float) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        raise NotImplementedError

    def _forward_batch(self, graph_mode: bool, model: nn.Module, batch):
        if graph_mode:
            batch = batch.to(self.device)
            preds = model(batch)
            targets = batch.y
            int_indices = batch.int_indices
            sample_count = batch.batch_size
        else:
            inputs, targets = batch
            # Some datasets append extra per-sample tensors (e.g. the causal adjacency);
            # only the first four are model inputs.
            sample_int, int_indices, sample_orig, obs = inputs[:4]
            inputs = (sample_int.float().to(self.device), int_indices.to(self.device), sample_orig.float().to(self.device), obs.float().to(self.device))
            targets = targets.float().to(self.device)
            preds = model(inputs)
            sample_count = targets.shape[0]

        targets = targets.reshape(-1, preds.shape[2], targets.shape[-1])
        preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[-2], preds.shape[-1])
        return preds, targets, int_indices, sample_count

    def _run_training_epoch(self, graph_mode: bool, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, target_indices, 
                            target_indice_map, train_loss_indices: bool, train_fn, needs_indices: bool) -> EpochStatistics:
        model.train()
        loss_sum, targeted_loss_sum, sample_count = 0.0, 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds, targets, int_indices, batch_size = self._forward_batch(graph_mode, model, batch)
            self._report_batch_once(batch_size)

            if train_loss_indices:
                loss, targeted_loss = compute_indiced_loss(preds, targets, target_indices, target_indice_map, int_indices, train_fn, needs_indices=needs_indices)
            else:
                loss = mog_nll(preds, targets)
                targeted_loss = loss

            self.backward(loss)
            optimizer.step()
            loss_sum += loss.detach().item()
            targeted_loss_sum += targeted_loss.detach().item()
            sample_count += batch_size

        return EpochStatistics(loss_sum, targeted_loss_sum, sample_count)

    def _run_validation(self, graph_mode: bool, output_classified: bool, train_loss_indices: bool, val_loader: DataLoader, model: nn.Module, loss_fn, target_indices, target_indice_map, mc_sampling: bool) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            if output_classified:
                kwargs = {"target_indice_map": target_indice_map} if train_loss_indices else {}
                val_loss = eval_dloader_loss(loss_fn, val_loader, model, device=self.device, **kwargs)
                test_val_loss = val_loss
            elif graph_mode:
                val_loss = eval_nll_graph(val_loader, model, self.device, target_indices, target_indice_map) if train_loss_indices else eval_nll_graph(val_loader, model, self.device)
                test_val_loss = eval_test_graph(val_loader, model, loss_fn, target_indices, target_indice_map, self.device, mc_sampling)
            else:
                val_loss = eval_nll(val_loader, model, self.device, target_indices, target_indice_map) if train_loss_indices else eval_nll(val_loader, model, self.device)
                test_val_loss = eval_test(val_loader, model, loss_fn, target_indices, target_indice_map, self.device, mc_sampling)

        return self.finalize_validation_values(float(val_loss), float(test_val_loss))

    def _run_test(self, graph_mode: bool, output_classified: bool, test_loader: DataLoader, model: nn.Module, loss_fn, target_indices, target_indice_map, 
                  mc_sampling: bool) -> float:
        model.eval()
        with torch.no_grad():
            if output_classified:
                test_loss = eval_dloader_loss(loss_fn, test_loader, model, target_indice_map=target_indice_map, device=self.device)
            elif graph_mode:
                test_loss = eval_test_graph(test_loader, model, loss_fn, target_indices, target_indice_map, self.device, mc_sampling)
            else:
                test_loss = eval_test(test_loader, model, loss_fn, target_indices, target_indice_map, self.device, mc_sampling)
        return self.finalize_test_value(float(test_loss))

    def train_counterfactual(self, graph_mode: bool, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, 
                             loss_fn, target_indices, n_epochs: int = 100, patience: int = 50, save_path: str = "./model_artifacts/best_model_graph.pt", 
                             print_every: int = 1, train_loss_indices: bool = False, target_indice_map: Optional[torch.Tensor] = None, lr: float = 5e-5, 
                             mc_sampling: bool = False, train_fn=None, output_classified: bool = False, restore_state: bool = True,
                             state_path: Optional[str] = None) -> dict:
        train_fn = intervention_nll_mog if train_fn is None else train_fn
        needs_indices = bool(getattr(train_fn, "needs_indices", False))
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        state_path = state_path or state_path_for(save_path)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=10)
        prepared = self.prepare_training(model, optimizer, train_loader)
        model, optimizer, train_loader = prepared.model, prepared.optimizer, prepared.train_loader
        target_indice_map = target_indice_map.to(self.device) if target_indice_map is not None else None

        best_val, best_epoch, epochs_no_improve, start_epoch = float("inf"), -1, 0, 1
        history = {"train_loss": [], "target_indice_train_loss": [], "val_loss": [], "val_fntest_loss": []}
        epoch_times: list[float] = []
        self.report(f"Training on {self.device}")

        if restore_state:
            progress = self.load_training_state(model, optimizer, scheduler, state_path)
            if progress is None:
                self.report(f"No training state at {state_path}; starting from scratch.")
            else:
                best_val, best_epoch = progress["best_val"], progress["best_epoch"]
                epochs_no_improve, start_epoch = progress["epochs_no_improve"], progress["epoch"] + 1
                history = {key: list(progress["history"].get(key, [])) for key in history}
                self.report(f"Resuming from epoch {start_epoch} (best_val {best_val:.6f} at epoch {best_epoch})")

        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start = time.perf_counter()
            stats = self._run_training_epoch(graph_mode, model, optimizer, train_loader, target_indices, target_indice_map, train_loss_indices, train_fn,
                                             needs_indices)
            stats = self.finalize_epoch_statistics(stats)
            train_loss = stats.loss_sum / max(1, stats.sample_count)
            targeted_loss = stats.targeted_loss_sum / max(1, stats.sample_count)
            val_loss, test_val_loss = self._run_validation(graph_mode, output_classified, train_loss_indices, val_loader, model, loss_fn, target_indices, 
                                                           target_indice_map, mc_sampling)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["target_indice_train_loss"].append(targeted_loss)
            history["val_loss"].append(val_loss)
            history["val_fntest_loss"].append(test_val_loss)

            if val_loss < best_val - 1e-9:
                best_val, best_epoch, epochs_no_improve = val_loss, epoch, 0
                self.save_checkpoint(model, save_path)
            else:
                epochs_no_improve += 1

            self.save_training_state(model, optimizer, scheduler, {"epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                                                                   "epochs_no_improve": epochs_no_improve, "history": history}, state_path)

            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            self.report(f"[Epoch {epoch:3d}] took {epoch_time:.2f}s")

            if epoch == 1 or epoch % print_every == 0:
                self.report(f"[Epoch {epoch:3d}] train_loss {train_loss:.6f} | train_t_indices {targeted_loss:.6f} | val_loss {val_loss:.6f} | val_fntest {test_val_loss:.6f} | best_val {best_val:.6f} (epoch {best_epoch})")

            self.synchronize()
            if epochs_no_improve >= patience:
                self.report(f"Early stopping triggered (no improvement in {patience} epochs).")
                break

        if os.path.exists(save_path):
            self.report(f"Restoring best model from epoch {best_epoch} with val_loss={best_val:.6f}")
            self.restore_checkpoint(model, save_path)
        else:
            self.report("No saved checkpoint found; saving final weights.")
            self.save_checkpoint(model, save_path)

        test_loss = self._run_test(graph_mode, output_classified, test_loader, model, loss_fn, target_indices, target_indice_map, mc_sampling)
        final_dir = ChexpertPath.get_final_path() if ChexpertPath.get_final_path() else "./artifacts"
        os.makedirs(final_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

        final_path = f"{final_dir}/graph_model_{timestamp}.pt" if graph_mode else f"{final_dir}/model_{timestamp}.pt"
        self.save_checkpoint(model, final_path)

        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

        return {"history": history, "best_epoch": best_epoch, "best_val": best_val, "test_loss": test_loss, "save_path": save_path, "final_path": final_path,
                "state_path": state_path, "lr": lr, "is_main_process": self.is_main_process, "avg_epoch_time": avg_epoch_time}


class ImageCounterfactualMixin:
    """Training variant for models that return `(predictions, decoded_image)`.
    """

    def __init__(self, *args, recon_target_fn=None, **kwargs):
        if recon_target_fn is None:
            raise ValueError("The image training pipeline needs a `recon_target_fn` to build reconstruction targets.")
        self.recon_target_fn = recon_target_fn
        super().__init__(*args, **kwargs)

    def _forward_batch(self, graph_mode: bool, model: nn.Module, batch):
        if not graph_mode:
            raise ValueError("The image training pipeline only supports graph datasets.")

        batch = batch.to(self.device)
        preds, pred_image = model(batch)
        targets = batch.y
        int_indices = batch.int_indices
        sample_count = batch.batch_size

        targets = targets.reshape(-1, preds.shape[2], targets.shape[-1])
        preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[-2], preds.shape[-1])
        # The image stays as (B, C, H, W) and the target is taken from the batch, not from `y`:
        # the CheXpert targets only carry the tabular attributes.
        return preds, targets, int_indices, sample_count, pred_image, self.recon_target_fn(batch)

    def _run_training_epoch(self, graph_mode: bool, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, target_indices,
                            target_indice_map, train_loss_indices: bool, train_fn, needs_indices: bool) -> EpochStatistics:
        model.train()
        loss_sum, targeted_loss_sum, sample_count = 0.0, 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds, targets, int_indices, batch_size, pred_image, target_image = self._forward_batch(graph_mode, model, batch)
            self._report_batch_once(batch_size)

            loss_extras = image_loss_kwargs(train_fn, pred_image, target_image)
            if train_loss_indices:
                loss, targeted_loss = compute_indiced_loss(preds, targets, target_indices, target_indice_map, int_indices, train_fn,
                                                           needs_indices=needs_indices, loss_extras=loss_extras)
            else:
                loss = train_fn(preds, targets, **loss_extras)
                targeted_loss = loss

            self.backward(loss)
            optimizer.step()
            loss_sum += loss.detach().item()
            targeted_loss_sum += targeted_loss.detach().item()
            sample_count += batch_size

        return EpochStatistics(loss_sum, targeted_loss_sum, sample_count)

    def _run_validation(self, graph_mode: bool, output_classified: bool, train_loss_indices: bool, val_loader: DataLoader, model: nn.Module, loss_fn,
                        target_indices, target_indice_map, mc_sampling: bool) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            val_loss = eval_dloader_image_loss(loss_fn, val_loader, model, self.recon_target_fn,
                                               target_indice_map=target_indice_map if train_loss_indices else None, device=self.device)
        return self.finalize_validation_values(float(val_loss), float(val_loss))

    def _run_test(self, graph_mode: bool, output_classified: bool, test_loader: DataLoader, model: nn.Module, loss_fn, target_indices, target_indice_map,
                  mc_sampling: bool) -> float:
        model.eval()
        with torch.no_grad():
            test_loss = eval_dloader_image_loss(loss_fn, test_loader, model, self.recon_target_fn, target_indice_map=target_indice_map, device=self.device)
        return self.finalize_test_value(float(test_loss))


class OriginalCounterfactualTrainer(CounterfactualTrainingTemplate):
    def __init__(self, device: torch.device):
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main_process(self) -> bool:
        return True

    def prepare_training(self, model, optimizer, train_loader):
        return PreparedTraining(model.to(self.device), optimizer, train_loader)

    def backward(self, loss):
        loss.backward()

    def finalize_epoch_statistics(self, statistics):
        return statistics

    def save_checkpoint(self, model, save_path):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # Write-then-rename so a torn/interrupted write cannot leave a corrupted checkpoint behind.
        torch.save(model.state_dict(), f"{save_path}.tmp")
        os.replace(f"{save_path}.tmp", save_path)

    def restore_checkpoint(self, model, save_path):
        model.load_state_dict(torch.load(save_path, map_location=self.device))

    def save_training_state(self, model, optimizer, scheduler, progress, state_path):
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "progress": progress}
        # Write-then-rename so a job killed mid-save cannot leave a truncated state behind.
        torch.save(state, f"{state_path}.tmp")
        os.replace(f"{state_path}.tmp", state_path)

    def load_training_state(self, model, optimizer, scheduler, state_path):
        if not os.path.exists(state_path):
            return None
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        return state["progress"]

    def report(self, message):
        print(message)

    def synchronize(self):
        pass

    def finalize_validation_values(self, val_loss, test_val_loss):
        return val_loss, test_val_loss

    def finalize_test_value(self, test_loss):
        return test_loss


class AcceleratedCounterfactualTrainer(CounterfactualTrainingTemplate):
    def __init__(self, split_batches: bool = True, find_unused_parameters: bool = True):
        try:
            from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        except ImportError as exc:
            raise ImportError("Install Accelerate with: pip install accelerate") from exc
        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        )
        self._report_distribution(split_batches)

    def _report_distribution(self, split_batches: bool) -> None:
        state = self.accelerator.state
        print(f"{self.rank_label} device={self.accelerator.device} distributed_type={state.distributed_type} "
              f"split_batches={split_batches} visible_gpus={torch.cuda.device_count()}", flush=True)
        if state.num_processes == 1:
            self.report("WARNING: only one process is running, so nothing is distributed. Launch with "
                        "`accelerate launch --multi_gpu --num_processes=<num_gpus>`.")

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def rank_label(self) -> str:
        return f"[rank {self.accelerator.process_index + 1}/{self.accelerator.num_processes}]"

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    def prepare_training(self, model, optimizer, train_loader):
        model, optimizer, train_loader = self.accelerator.prepare(model, optimizer, train_loader)
        return PreparedTraining(model, optimizer, train_loader)

    def backward(self, loss):
        self.accelerator.backward(loss)

    def finalize_epoch_statistics(self, statistics):
        values = torch.tensor([statistics.loss_sum, statistics.targeted_loss_sum, statistics.sample_count], dtype=torch.float64, device=self.device)
        values = self.accelerator.reduce(values, reduction="sum")
        return EpochStatistics(values[0].item(), values[1].item(), int(values[2].item()))

    def save_checkpoint(self, model, save_path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            # Write-then-rename so a torn/interrupted write cannot leave a corrupted checkpoint behind.
            self.accelerator.save(self.accelerator.unwrap_model(model).state_dict(), f"{save_path}.tmp")
            os.replace(f"{save_path}.tmp", save_path)
        self.accelerator.wait_for_everyone()

    def restore_checkpoint(self, model, save_path):
        self.accelerator.wait_for_everyone()
        state = torch.load(save_path, map_location="cpu")
        self.accelerator.unwrap_model(model).load_state_dict(state)
        self.accelerator.wait_for_everyone()

    def save_training_state(self, model, optimizer, scheduler, progress, state_path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
            state = {"model": self.accelerator.unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "progress": progress}
            # Write-then-rename so a job killed mid-save cannot leave a truncated state behind.
            self.accelerator.save(state, f"{state_path}.tmp")
            os.replace(f"{state_path}.tmp", state_path)
        self.accelerator.wait_for_everyone()

    def load_training_state(self, model, optimizer, scheduler, state_path):
        self.accelerator.wait_for_everyone()
        if not os.path.exists(state_path):
            return None
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.accelerator.unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        self.accelerator.wait_for_everyone()
        return state["progress"]

    def report(self, message):
        self.accelerator.print(message)

    def synchronize(self):
        self.accelerator.wait_for_everyone()

    def finalize_validation_values(self, val_loss, test_val_loss):
        values = torch.tensor([val_loss, test_val_loss], dtype=torch.float64, device=self.device)
        values = self.accelerator.reduce(values, reduction="mean")
        return values[0].item(), values[1].item()

    def finalize_test_value(self, test_loss):
        value = torch.tensor(test_loss, dtype=torch.float64, device=self.device)
        return self.accelerator.reduce(value, reduction="mean").item()


class OriginalImageCounterfactualTrainer(ImageCounterfactualMixin, OriginalCounterfactualTrainer):
    """Single-device image pipeline."""


class AcceleratedImageCounterfactualTrainer(ImageCounterfactualMixin, AcceleratedCounterfactualTrainer):
    """Multi-device image pipeline."""


def create_counterfactual_trainer(device: torch.device, use_acceleration: Optional[bool] = None, recon_target_fn=None) -> CounterfactualTrainingTemplate:
    use_acceleration = torch.cuda.device_count() > 1 if use_acceleration is None else use_acceleration
    if recon_target_fn is not None:
        if use_acceleration:
            return AcceleratedImageCounterfactualTrainer(recon_target_fn=recon_target_fn)
        return OriginalImageCounterfactualTrainer(device, recon_target_fn=recon_target_fn)
    return AcceleratedCounterfactualTrainer() if use_acceleration else OriginalCounterfactualTrainer(device)


def train_counterfactual(graph_mode: bool, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, 
                         loss_fn, target_indices, n_epochs: int = 100, patience: int = 10, save_path: str = "./model_artifacts/best_model_graph.pt", 
                         print_every: int = 1, train_loss_indices: bool = False, target_indice_map: Optional[torch.Tensor] = None, lr: float = 5e-5, 
                         mc_sampling: bool = False, train_fn=None, output_classified: bool = False, use_acceleration: Optional[bool] = None,
                         restore_state: bool = True, state_path: Optional[str] = None, recon_target_fn=None) -> dict:
    # Passing `recon_target_fn` selects the image pipeline, which keeps the model's decoded
    # image out of the node-indexed reshaping and feeds it to the loss directly.
    trainer = create_counterfactual_trainer(device, use_acceleration, recon_target_fn)
    return trainer.train_counterfactual(graph_mode, model, train_loader, val_loader, test_loader, loss_fn, target_indices, n_epochs, patience, save_path,
                                        print_every, train_loss_indices, target_indice_map, lr, mc_sampling, train_fn, output_classified, restore_state,
                                        state_path)