import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torch import nn
from torchvision import transforms as T

from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers=[],
        crawler_ext="tif",  # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform=None,
        image_resize=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
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
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext)

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize((image_resize, image_resize))
                    if image_resize is not None
                    else nn.Identity(),
                    T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                    T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                    T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                    T.ToTensor(),
                    # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                    # ToTensor16bit(),
                    # Normalize(), # [0, 1.0]
                    # T.ConvertImageDtype(torch.float),
                    T.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),  # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        return {"uid": rel_path_item.stem, "source": self.transform(img)}

    def load_item(self, path_item):
        return Image.open(path_item).convert("RGB")
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f"*.{extension}")]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(
        self,
        *args,
        validate_images: bool = True,
        use_cache: bool = True,
        cache_dir=None,
        cache_file_name: str = "chexpert_cache.pt",
        cache_builder: str = "checkpointed",  # options: 'single' or 'checkpointed'
        checkpoint_interval: int = 100,
        cache_workers = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_cache = use_cache
        self.cache_builder = cache_builder
        self.checkpoint_interval = checkpoint_interval
        self.cache_workers = cache_workers or max(1, min(32, os.cpu_count() or 4))

        self.cache_dir = Path(cache_dir) if cache_dir else self.path_root.parent / "tensor_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # directory to store temporary checkpoint part files when building cache
        self.checkpoint_dir = self.cache_dir / "checkpoint_parts"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / cache_file_name
        self.root_dir = Path(self.path_root).parent

        self.labels = self._load_or_build_labels(validate_images=validate_images)
        print(f"[CheXpert] Final dataset size: {len(self.labels)}")

        self.images = None
        self.targets = None
        self.uids = None

        if self.use_cache:
            if self.cache_file.exists():
                self._load_cache()
            else:
                if self.cache_builder == "checkpointed":
                    self._build_cache_with_checkpoints(self.checkpoint_interval)
                else:
                    self._build_cache()

    def _load_or_build_labels(self, validate_images: bool = True):
        csv_path = Path(self.path_root)
        safe_path = csv_path.parent / f"safe_{csv_path.name}"

        if safe_path.exists():
            print(f"[CheXpert] Loading cached labels from {safe_path}")
            labels = pd.read_csv(safe_path, index_col="Path")
            return labels

        print("[CheXpert] Building labels from scratch...")
        labels = pd.read_csv(csv_path, index_col="Path")
        mask = labels["No Finding"].isna()

        labels.loc[mask, "No Finding"] = (~labels.loc[mask, self.finding_cols].eq(1).any(axis=1).astype(int))
        labels["No Finding"] = (labels["No Finding"] == 1).astype(int)

        # Keep only frontal images
        labels = labels.loc[labels["Frontal/Lateral"] == "Frontal"].copy()

        # Fix metadata
        labels.loc[labels["Sex"] == "Unknown", "Sex"] = "Female"
        labels.fillna(2, inplace=True)

        str_2_int = {
            "Sex": {"Male": 0, "Female": 1},
            "Frontal/Lateral": {"Frontal": 0, "Lateral": 1},
            "AP/PA": {"AP": 0, "PA": 1},
        }
        labels.replace(str_2_int, inplace=True)

        existing_files = {
            str(p.relative_to(self.root_dir)).replace("\\", "/")
            for p in self.root_dir.rglob("*")
            if p.is_file()
        }
        labels = labels.loc[labels.index.isin(existing_files)]

        if validate_images:
            valid_indices = []
            removed = 0

            for uid in labels.index:
                path_item = self.root_dir / uid
                try:
                    with Image.open(path_item) as img:
                        img.convert("RGB")
                    valid_indices.append(uid)
                except Exception:
                    removed += 1

            print(f"[CheXpert] Removed {removed} corrupted images.")
            labels = labels.loc[valid_indices]

        labels.to_csv(safe_path)
        print(f"[CheXpert] Saved cleaned labels -> {safe_path}")
        return labels

    def _load_cache_sample(self, uid):
        path_item = self.root_dir / uid
        try:
            with Image.open(path_item) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
            target = int(self.labels.loc[uid, "Cardiomegaly"]) + 1
            return str(uid), tensor, target
        except Exception:
            return None

    def _save_cache_part(self, images, targets, uids, part_file):
        part_images = torch.stack(images, dim=0)
        part_targets = torch.tensor(targets, dtype=torch.long)
        torch.save(
            {
                "images": part_images,
                "targets": part_targets,
                "uids": uids,
            },
            part_file,
        )

    def _build_cache(self):
        print(f"[CheXpert] Building single RAM cache -> {self.cache_file}")

        images = []
        targets = []
        uids = []

        with ThreadPoolExecutor(max_workers=self.cache_workers) as ex:
            for result in ex.map(self._load_cache_sample, self.labels.index):
                if result is None:
                    continue
                uid, tensor, target = result
                images.append(tensor)
                targets.append(target)
                uids.append(uid)

        self.images = torch.stack(images, dim=0)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.uids = uids

        torch.save(
            {
                "images": self.images,
                "targets": self.targets,
                "uids": self.uids,
            },
            self.cache_file,
        )

        print("[CheXpert] RAM cache ready.")

    def _build_cache_with_checkpoints(self, checkpoint_interval: int = 100):
        """Build the RAM cache in checkpointed parts using parallel image loading."""
        print(
            f"[CheXpert] Building RAM cache with checkpoints -> {self.cache_file} "
            f"(interval={checkpoint_interval}, workers={self.cache_workers})"
        )

        part_files = []
        images = []
        targets = []
        uids = []
        part_idx = 0

        def flush_part():
            nonlocal images, targets, uids, part_idx
            if not uids:
                return
            part_file = self.checkpoint_dir / f"{self.cache_file.stem}.part{part_idx}.pt"
            self._save_cache_part(images, targets, uids, part_file)
            print(f"[CheXpert] Saved checkpoint part -> {part_file} (items={len(uids)})")
            part_files.append(part_file)
            images, targets, uids = [], [], []
            part_idx += 1

        with ThreadPoolExecutor(max_workers=self.cache_workers) as ex:
            for result in ex.map(self._load_cache_sample, self.labels.index):
                if result is None:
                    continue
                uid, tensor, target = result
                images.append(tensor)
                targets.append(target)
                uids.append(uid)

                if len(uids) >= checkpoint_interval:
                    flush_part()

        flush_part()

        print(f"[CheXpert] Combining {len(part_files)} parts into final cache -> {self.cache_file}")

        all_images = []
        all_targets = []
        all_uids = []

        all_existing_parts = sorted(self.checkpoint_dir.glob(f"{self.cache_file.stem}.part*.pt"))
        for pf in all_existing_parts:
            payload = torch.load(pf, map_location="cpu")
            all_images.append(payload["images"])
            all_targets.append(payload["targets"])
            all_uids.extend(payload["uids"])

        if len(all_images) > 0:
            self.images = torch.cat(all_images, dim=0)
        else:
            self.images = torch.empty((0,))

        if len(all_targets) > 0:
            self.targets = torch.cat(all_targets, dim=0)
        else:
            self.targets = torch.empty((0,), dtype=torch.long)

        self.uids = all_uids

        torch.save(
            {
                "images": self.images,
                "targets": self.targets,
                "uids": self.uids,
            },
            self.cache_file,
        )
        print(f"[CheXpert] Final cache saved -> {self.cache_file}")

        for pf in all_existing_parts:
            try:
                pf.unlink()
            except Exception:
                pass

    def _load_cache(self):
        print(f"[CheXpert] Loading RAM cache from {self.cache_file}")
        payload = torch.load(self.cache_file, map_location="cpu")
        self.images = payload["images"]
        self.targets = payload["targets"]
        self.uids = payload["uids"]
        print("[CheXpert] Cache loaded.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.use_cache:
            return {
                "uid": self.uids[index],
                "source": self.images[index],
                "target": self.targets[index],
            }

        uid = self.labels.index[index]
        path_item = self.root_dir / uid

        with Image.open(path_item) as img:
            img = img.convert("RGB")
            source = self.transform(img)

        target = torch.tensor(
            int(self.labels.loc[uid, "Cardiomegaly"]) + 1,
            dtype=torch.long,
        )

        return {
            "uid": str(uid),
            "source": source,
            "target": target,
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return []


class CheXPert_Extended(CheXpert_Dataset):
    def _load_cache_sample(self, uid):
        path_item = self.root_dir / uid
        try:
            with Image.open(path_item) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)

            target_df = self.labels.loc[uid][["Sex", "Age", "AP/PA", "No Finding"]].apply(
                pd.to_numeric,
                errors="coerce",
            )
            target_array = target_df.to_numpy(dtype=np.float32)
            target = torch.tensor(target_array, dtype=torch.float32)

            return str(uid), tensor, target
        except Exception:
            return None

    def _save_cache_part(self, images, targets, uids, part_file):
        part_images = torch.stack(images, dim=0)
        part_targets = torch.stack(targets, dim=0)
        torch.save(
            {
                "images": part_images,
                "targets": part_targets,
                "uids": uids,
            },
            part_file,
        )

    def _build_cache(self):
        print(f"[CheXpert] Building single RAM cache -> {self.cache_file}")

        images = []
        targets = []
        uids = []

        with ThreadPoolExecutor(max_workers=self.cache_workers) as ex:
            for result in ex.map(self._load_cache_sample, self.labels.index):
                if result is None:
                    continue
                uid, tensor, target = result
                images.append(tensor)
                targets.append(target)
                uids.append(uid)

        self.images = torch.stack(images, dim=0)
        self.targets = torch.stack(targets, dim=0)
        self.uids = uids

        torch.save(
            {
                "images": self.images,
                "targets": self.targets,
                "uids": self.uids,
            },
            self.cache_file,
        )

        print("[CheXpert] RAM cache ready.")

    def _build_cache_with_checkpoints(self, checkpoint_interval: int = 100):
        """Checkpointed cache builder for CheXPert_Extended with parallel loading."""
        print(
            f"[CheXpert] Building RAM cache (extended) with checkpoints -> {self.cache_file} "
            f"(interval={checkpoint_interval}, workers={self.cache_workers})"
        )

        part_files = []
        images = []
        targets = []
        uids = []
        part_idx = 0

        def flush_part():
            nonlocal images, targets, uids, part_idx
            if not uids:
                return
            part_file = self.checkpoint_dir / f"{self.cache_file.stem}.part{part_idx}.pt"
            self._save_cache_part(images, targets, uids, part_file)
            print(f"[CheXpert] Saved extended checkpoint part -> {part_file} (items={len(uids)})")
            part_files.append(part_file)
            images, targets, uids = [], [], []
            part_idx += 1

        with ThreadPoolExecutor(max_workers=self.cache_workers) as ex:
            for result in ex.map(self._load_cache_sample, self.labels.index):
                if result is None:
                    continue
                uid, tensor, target = result
                images.append(tensor)
                targets.append(target)
                uids.append(uid)

                if len(uids) >= checkpoint_interval:
                    flush_part()

        flush_part()

        print(f"[CheXpert] Combining {len(part_files)} extended parts into final cache -> {self.cache_file}")

        all_images = []
        all_targets = []
        all_uids = []

        all_existing_parts = sorted(self.checkpoint_dir.glob(f"{self.cache_file.stem}.part*.pt"))
        for pf in all_existing_parts:
            payload = torch.load(pf, map_location="cpu")
            all_images.append(payload["images"])
            all_targets.append(payload["targets"])
            all_uids.extend(payload["uids"])

        if len(all_images) > 0:
            self.images = torch.cat(all_images, dim=0)
        else:
            self.images = torch.empty((0,))

        if len(all_targets) > 0:
            self.targets = torch.cat(all_targets, dim=0)
        else:
            self.targets = torch.empty((0,))

        self.uids = all_uids

        torch.save(
            {
                "images": self.images,
                "targets": self.targets,
                "uids": self.uids,
            },
            self.cache_file,
        )
        print(f"[CheXpert] Final extended cache saved -> {self.cache_file}")

        for pf in all_existing_parts:
            try:
                pf.unlink()
            except Exception:
                pass

    def __getitem__(self, index):
        uid = self.labels.index[index]

        if self.use_cache:
            return {
                "uid": self.uids[index],
                "source": self.images[index],
                "target": self.targets[index],
            }

        path_item = self.root_dir / uid

        with Image.open(path_item) as img:
            img = img.convert("RGB")
            source = self.transform(img)

        target_df = self.labels.loc[uid][["Sex", "Frontal/Lateral", "Age", "AP/PA", "No Finding"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        target_df["No Finding"] = target_df["No Finding"] - 1
        target_array = target_df.to_numpy(dtype=np.float32)

        target = torch.tensor(target_array, dtype=torch.float32)

        return {
            "uid": str(uid),
            "source": source,
            "target": target,
        }


class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(self.path_root / "labels/cheXPert_label.csv", index_col=["Path", "Image Index"])  # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels["fold"] == "train"].copy()
        labels = labels.drop(labels="fold", axis=1)

        labels2 = pd.read_csv(self.path_root / "labels/train.csv", index_col="Path")
        labels2 = labels2.loc[labels2["Frontal/Lateral"] == "Frontal"].copy()
        labels2 = labels2[["Cardiomegaly"]].copy()
        labels2[(labels2 < 0) | labels2.isna()] = 2  # 0 = Negative, 1 = Positive, 2 = Uncertain
        labels = labels.join(labels2["Cardiomegaly"], on=["Path"], rsuffix="_true")
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root / "data" / f"{image_index:06}.png"
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), "Cardiomegaly"])
        return {"source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
            weights[index] = weight_per_class[target]
        return weights