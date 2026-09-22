import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T


class CheXpertDataset:
    def __init__(
        self,
        vae: nn.Module,
        path: str,
        mode: str,
        save_path: Optional[str] = None,
        image_resize: Optional[int] = 128,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        resume_checkpoints: bool = True,
    ):
        self.vae = vae
        self.path = Path(path)
        self.mode = mode
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.transform = T.Compose(
            [
                T.Resize((image_resize, image_resize)) if image_resize is not None else nn.Identity(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
        )

        self.cols = ["Sex", "Age", "AP/PA", "No Finding"]
        self.finding_cols = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        self.save_path = f"./data/chexpert/chexpert_{seed}.pt" if save_path is None else save_path

        # Checkpoint configuration
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(self.save_path), "checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.resume_checkpoints = resume_checkpoints

        set_name = self.path.name
        safe_path = self.path.parent / f"safe_{set_name}"

        if safe_path.exists():
            print(f"[CheXpert] Loading cached labels from {safe_path}")
            labels = pd.read_csv(safe_path, index_col="Path")
        else:
            print("[CheXpert] Building labels from scratch...")
            labels = pd.read_csv(self.path, index_col="Path")

            labels = labels.loc[labels["Frontal/Lateral"] == "Frontal"].copy()

            labels.loc[labels["Sex"] == "Unknown", "Sex"] = "Female"

            mask = labels["No Finding"].isna()
            labels.loc[mask, "No Finding"] = (
                ~labels.loc[mask, self.finding_cols].eq(1).any(axis=1).astype(int)
            )
            labels["No Finding"] = (labels["No Finding"] == 1).astype(int)

            labels.fillna(2, inplace=True)

            str_2_int = {
                "Sex": {"Male": 0, "Female": 1},
                "Frontal/Lateral": {"Frontal": 0, "Lateral": 1},
                "AP/PA": {"AP": 0, "PA": 1},
            }
            labels.replace(str_2_int, inplace=True)

            root = Path(self.path.parent)
            existing_files = {
                str(p.relative_to(root)).replace("\\", "/")
                for p in root.rglob("*")
                if p.is_file()
            }

            labels = labels.loc[labels.index.isin(existing_files)]

            valid_indices = []
            removed = 0
            for uid in labels.index:
                path_item = root / uid
                try:
                    with Image.open(path_item) as img:
                        img.convert("RGB")
                    valid_indices.append(uid)
                except Exception:
                    removed += 1

            print(f"[CheXpert] Removed {removed} corrupted images.")
            labels = labels.loc[valid_indices]

            safe_path.parent.mkdir(parents=True, exist_ok=True)
            labels.to_csv(safe_path)
            print(f"[CheXpert] Saved cleaned labels -> {safe_path}")

        self.labels = labels.apply(pd.to_numeric, errors="coerce")
        print(f"[CheXpert] Final dataset size: {len(self.labels)}")

    def _checkpoint_stem(self, suffix: str = "") -> str:
        return f"{Path(self.save_path).stem}{suffix}"

    def _checkpoint_manifest_path(self, suffix: str = "") -> Path:
        return self.checkpoint_dir / f"{self._checkpoint_stem(suffix)}.manifest.json"

    def _checkpoint_part_path(self, part_idx: int, suffix: str = "") -> Path:
        return self.checkpoint_dir / f"{self._checkpoint_stem(suffix)}.part{part_idx}.pt"

    def _load_checkpoint_parts(self, suffix: str = ""):
        part_files = sorted(self.checkpoint_dir.glob(f"{self._checkpoint_stem(suffix)}.part*.pt"))
        samples = []
        for pf in part_files:
            samples.extend(torch.load(pf, map_location="cpu"))
        return samples, part_files

    def _save_checkpoint_part(self, samples, part_idx: int, suffix: str = ""):
        if not samples:
            return None
        part_file = self._checkpoint_part_path(part_idx, suffix=suffix)
        torch.save(samples, part_file)
        return part_file

    def _load_manifest(self, suffix: str = ""):
        manifest_path = self._checkpoint_manifest_path(suffix=suffix)
        if not manifest_path.exists():
            return {"processed_keys": [], "next_part_idx": 0}
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("processed_keys", [])
        manifest.setdefault("next_part_idx", 0)
        return manifest

    def _save_manifest(self, processed_keys, next_part_idx: int, suffix: str = ""):
        manifest_path = self._checkpoint_manifest_path(suffix=suffix)
        payload = {
            "processed_keys": list(processed_keys),
            "next_part_idx": int(next_part_idx),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _clear_checkpoints(self, suffix: str = ""):
        for pf in self.checkpoint_dir.glob(f"{self._checkpoint_stem(suffix)}.part*.pt"):
            try:
                pf.unlink()
            except Exception:
                pass

        manifest_path = self._checkpoint_manifest_path(suffix=suffix)
        try:
            if manifest_path.exists():
                manifest_path.unlink()
        except Exception:
            pass

    def get_image(self, path):
        img_path = Path(path)
        if not img_path.is_absolute():
            img_path = self.path.parent / img_path

        im = Image.open(img_path).convert("RGB")
        img_tens = self.transform(im).float()

        if self.mode == "enc":
            returned = self.vae(img_tens)
            return returned.reshape((returned.shape[0], -1)).transpose(0, 1)
        else:
            return img_tens.reshape((img_tens.shape[0], -1)).transpose(0, 1)

    def combine_images(self, paths):
        return torch.stack([self.get_image(path) for path in paths], dim=0).mean(dim=0)

    def get_observational_samples(self, n_obs, rng=None):
        if rng is None:
            rng = np.random.default_rng(self.seed)

        idx = rng.integers(0, len(self.labels), size=n_obs)

        single_feats = torch.from_numpy(
            self.labels.iloc[idx][self.cols].to_numpy(dtype=np.float32)
        )
        paths = self.labels.iloc[idx].index

        imgs = torch.stack([self.get_image(path) for path in paths], dim=0)

        return torch.cat(
            [single_feats.unsqueeze(-1).expand(-1, -1, imgs.shape[-1]), imgs],
            dim=1,
        )

    def match_dims(self, target, single_feats):
        if isinstance(single_feats, (pd.Series, pd.DataFrame)):
            single_feats = single_feats.to_numpy(dtype=np.float32)

        single_feats = torch.from_numpy(single_feats).float()
        single_feats = single_feats.unsqueeze(-1).expand(-1, target.shape[-1])

        return torch.cat([single_feats, target], dim=0)

    def _build_row_counterfactuals(
        self,
        path,
        row,
        labels,
        n_obs,
        max_values_per_col,
        row_seed,
    ):
        rng = np.random.default_rng(row_seed)
        samples = []

        x_orig = self.match_dims(self.get_image(path), row[self.cols])
        obs = self.get_observational_samples(n_obs, rng=rng)

        def full_match_mask(frame, row, cols):
            if len(cols) == 0:
                return pd.Series(True, index=frame.index)
            return frame[cols].eq(row[cols]).all(axis=1)

        for i_col, col in enumerate(self.cols):
            left_cols = self.cols[:i_col]
            right_cols = self.cols[i_col + 1:]

            matching_left = full_match_mask(labels, row, left_cols)
            matching_right = full_match_mask(labels, row, right_cols)

            different = labels.loc[
                matching_left & matching_right & (labels[col] != row[col])
            ]

            unique_others = different[col].drop_duplicates().to_numpy()

            # Only allow Age interventions with distance >= 5
            if col == "Age":
                unique_others = np.array(
                    [v for v in unique_others if abs(float(v) - float(row[col])) >= 5]
                )

            if max_values_per_col is not None and len(unique_others) > max_values_per_col:
                unique_others = rng.choice(unique_others, size=max_values_per_col, replace=False)

            for val in unique_others:
                paths = different.loc[different[col] == val].index
                if len(paths) == 0:
                    continue

                chosen_path = rng.choice(paths)
                target = self.get_image(chosen_path)

                cf_row = row.copy()
                cf_row[col] = val
                x_int = self.match_dims(target, cf_row[self.cols])

                val_tensor = torch.zeros_like(x_int)
                val_tensor[i_col] = torch.ones(x_int.shape[-1]) * val

                sample_tuple = (
                    (
                        val_tensor.unsqueeze(0).unsqueeze(-1),
                        torch.tensor(i_col, dtype=torch.int64).unsqueeze(0).unsqueeze(-1),
                        x_orig.to(dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                        obs.to(dtype=torch.float32).unsqueeze(-1),
                    ),
                    x_int.to(dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                )
                samples.append(sample_tuple)

        return samples

    def make_counterfactuals(
        self,
        n_obs: int = 50,
        max_rows: int = 1000,
        max_values_per_col: int = 5,
        seed: int = 0,
        use_parallel: bool = True,
        num_workers: int = 64,
    ):
        final_path = Path(self.save_path)
        if final_path.exists():
            return torch.load(final_path)

        rng = np.random.default_rng(seed)
        labels = self.labels

        if max_rows is not None and max_rows < len(labels):
            sampled_idx = rng.choice(labels.index.to_numpy(), size=max_rows, replace=False)
            rows = labels.loc[sampled_idx]
        else:
            rows = labels

        checkpoint_suffix = ""
        manifest = self._load_manifest(suffix=checkpoint_suffix)
        processed_keys = set(manifest["processed_keys"])
        next_part_idx = int(manifest["next_part_idx"])

        existing_samples = []
        if self.resume_checkpoints:
            existing_samples, _ = self._load_checkpoint_parts(suffix=checkpoint_suffix)

        remaining_rows = rows.loc[~rows.index.isin(processed_keys)].copy()

        samples_buffer = []
        current_processed = list(processed_keys)
        part_idx = next_part_idx

        def flush_buffer():
            nonlocal samples_buffer, part_idx, current_processed
            if not samples_buffer:
                return
            self._save_checkpoint_part(samples_buffer, part_idx, suffix=checkpoint_suffix)
            part_idx += 1
            self._save_manifest(current_processed, part_idx, suffix=checkpoint_suffix)
            samples_buffer = []

        if use_parallel:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for idx, (path, row) in enumerate(remaining_rows.iterrows()):
                    row_seed = int(seed + idx)
                    futures.append(
                        (path, executor.submit(
                            self._build_row_counterfactuals,
                            path,
                            row,
                            labels,
                            n_obs,
                            max_values_per_col,
                            row_seed,
                        ))
                    )

                for path, fut in futures:
                    row_samples = fut.result()
                    samples_buffer.extend(row_samples)
                    current_processed.append(str(path))

                    if len(current_processed) % self.checkpoint_interval == 0:
                        flush_buffer()
        else:
            for idx, (path, row) in enumerate(remaining_rows.iterrows()):
                row_seed = int(seed + idx)
                row_samples = self._build_row_counterfactuals(
                    path,
                    row,
                    labels,
                    n_obs,
                    max_values_per_col,
                    row_seed,
                )
                samples_buffer.extend(row_samples)
                current_processed.append(str(path))

                if len(current_processed) % self.checkpoint_interval == 0:
                    flush_buffer()

        flush_buffer()

        all_samples = existing_samples
        fresh_samples, part_files = self._load_checkpoint_parts(suffix=checkpoint_suffix)
        if fresh_samples:
            all_samples = fresh_samples

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(all_samples, self.save_path)

        self._clear_checkpoints(suffix=checkpoint_suffix)
        return all_samples

    def make_counterfactuals_safe(self, n_obs: int = 100):
        final_path = Path(self.save_path)
        # if final_path.exists():
        #     return torch.load(final_path)

        checkpoint_suffix = "_safe"
        manifest = self._load_manifest(suffix=checkpoint_suffix)
        processed_keys = set(manifest["processed_keys"])
        next_part_idx = int(manifest["next_part_idx"])

        existing_samples = []
        if self.resume_checkpoints:
            existing_samples, _ = self._load_checkpoint_parts(suffix=checkpoint_suffix)

        samples_buffer = []
        current_processed = list(processed_keys)
        part_idx = next_part_idx

        allowed_cols = ["Age", "Frontal/Lateral", "AP/PA", "No Finding"]
        self.labels["patient_id"] = self.labels.index.str.split("/").str[1]
        grouped = self.labels.groupby("patient_id")

        def flush_buffer():
            nonlocal samples_buffer, part_idx, current_processed
            if not samples_buffer:
                return
            self._save_checkpoint_part(samples_buffer, part_idx, suffix=checkpoint_suffix)
            part_idx += 1
            self._save_manifest(current_processed, part_idx, suffix=checkpoint_suffix)
            samples_buffer = []

        for patient_id, group in grouped:
            if patient_id in processed_keys:
                continue

            if len(group) < 2:
                current_processed.append(patient_id)
                continue

            group = group.reset_index()
            for i in range(len(group)):
                row = group.iloc[i]

                for col in allowed_cols:
                    col_idx = self.cols.index(col)

                    different = group[group[col] != row[col]]
                    if len(different) == 0:
                        continue

                    for _, cf_row in different.iterrows():
                        x_orig_img = self.get_image(row["Path"])
                        x_orig = self.match_dims(x_orig_img, row[self.cols])

                        sample_int = cf_row[col]
                        target_img = self.get_image(cf_row["Path"])
                        x_int = self.match_dims(target_img, cf_row[self.cols])

                        obs = self.get_observational_samples(n_obs)

                        sample_tuple = (
                            (
                                torch.tensor(sample_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                                torch.tensor(col_idx, dtype=torch.int64).unsqueeze(0).unsqueeze(-1),
                                torch.tensor(x_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                                torch.tensor(obs, dtype=torch.float32).unsqueeze(-1),
                            ),
                            torch.tensor(x_int, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                        )
                        samples_buffer.append(sample_tuple)

                        if len(samples_buffer) >= self.checkpoint_interval:
                            flush_buffer()

            current_processed.append(patient_id)

        flush_buffer()

        all_samples = existing_samples
        fresh_samples, _ = self._load_checkpoint_parts(suffix=checkpoint_suffix)
        if fresh_samples:
            all_samples = fresh_samples

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(all_samples, self.save_path)

        self._clear_checkpoints(suffix=checkpoint_suffix)
        return all_samples


class CheXpertClassificationDataset():
    def __init__(self, vae: nn.Module, path: str, mode: str, save_path=None, seed=42):
        self.normal_set = CheXpertDataset(vae=vae, path=path, mode=mode, save_path=save_path, seed=seed)
        self.save_path = save_path if save_path is not None else f"./data/chexpert/chexpert_count_{seed}.pt"

    def make_counterfactuals(
        self,
        n_obs: int = 50,
        max_rows=1000,
        max_values_per_col=5,
        seed: int = 0,
    ):
        norm_set = self.normal_set.make_counterfactuals(
            n_obs,
            max_rows,
            max_values_per_col,
            seed,
        )
        samples = []
        for (sample_int, col_idx, x_orig, obs), target in norm_set:
            samples.append(((sample_int, col_idx, x_orig, obs), target.reshape((1, -1, 1))[:, :4, :1]))

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(samples, self.save_path)

        return samples

    def make_counterfactuals_safe(self, n_obs: int = 100):
        norm_set = self.normal_set.make_counterfactuals_safe(n_obs)
        samples = []
        for (sample_int, col_idx, x_orig, obs), target in norm_set:
            samples.append(((sample_int, col_idx, x_orig, obs), sample_int))

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(samples, self.save_path)

        return samples