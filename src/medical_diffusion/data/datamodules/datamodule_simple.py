import os
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds_train: object,
        ds_val: object = None,
        ds_test: object = None,
        batch_size: int = 1,
        num_workers: int = mp.cpu_count(),
        seed: int = 0,
        pin_memory: bool = False,
        weights: list = None,
        val_split: float = 0.2,
        cache_dir: str = None,
    ):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop("__class__")
        self.hyperparameters.pop("self")

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.weights = weights
        self.val_split = val_split
        self.cache_dir = cache_dir

        self.setup()

    def _cache_paths(self):
        if self.cache_dir is None:
            return None, None, None
        os.makedirs(self.cache_dir, exist_ok=True)
        train_path = os.path.join(self.cache_dir, "train_dataset.pt")
        val_path = os.path.join(self.cache_dir, "val_dataset.pt")
        test_path = os.path.join(self.cache_dir, "test_dataset.pt")
        return train_path, val_path, test_path

    def _save_if_missing(self, obj, path):
        if obj is None or path is None:
            return
        if not os.path.exists(path):
            torch.save(obj, path)

    def _load_if_exists(self, path):
        if path is not None and os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
        return None

    def setup(self, stage=None):
        train_path, val_path, test_path = self._cache_paths()

        # Try loading cached datasets first
        cached_train = self._load_if_exists(train_path)
        cached_val = self._load_if_exists(val_path)
        cached_test = self._load_if_exists(test_path)

        if cached_train is not None:
            self.ds_train = cached_train
        if cached_val is not None:
            self.ds_val = cached_val
        if cached_test is not None:
            self.ds_test = cached_test

        # If validation is missing, split train once and cache both halves
        if self.ds_val is None and self.ds_train is not None:
            total_len = len(self.ds_train)
            val_len = int(total_len * self.val_split)
            train_len = total_len - val_len

            generator = torch.Generator().manual_seed(self.seed)
            self.ds_train, self.ds_val = random_split(
                self.ds_train,
                [train_len, val_len],
                generator=generator,
            )

            self._save_if_missing(self.ds_train, train_path)
            self._save_if_missing(self.ds_val, val_path)

        # Cache test dataset if provided and missing
        self._save_if_missing(self.ds_test, test_path)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)

        if self.weights is not None:
            sampler = WeightedRandomSampler(
                self.weights,
                len(self.weights),
                generator=generator,
            )
        else:
            sampler = RandomSampler(self.ds_train, replacement=False, generator=generator)

        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            generator=generator,
            drop_last=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.ds_val is None:
            raise AssertionError("A validation set was not initialized.")

        generator = torch.Generator().manual_seed(self.seed)

        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            generator=generator,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.ds_test is None:
            raise AssertionError("A test set was not initialized.")

        generator = torch.Generator().manual_seed(self.seed)

        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            generator=generator,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )