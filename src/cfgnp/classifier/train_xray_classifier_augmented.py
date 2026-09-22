import copy
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import CheXPert_Extended
from cfgnp.classifier.disease_classifier import DiseaseClassifier
from cfgnp.util.util import DEVICE, REGRESSION_TARGET_INDICES


IMAGE_RESIZE = 128
BATCH_SIZE = 128
NUM_WORKERS = 16
RESNET_SIZE = 50

SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 10
MIN_LR = 1e-7

SEX_WEIGHT_DECAY = 1e-4
SEX_HEAD_LR_STAGE_1 = 5e-4
SEX_HEAD_LR_STAGE_2 = 5e-4
SEX_BACKBONE_LR_STAGE_2 = 1e-7
SEX_MAX_GRAD_NORM = 1.0
SEX_BACKBONE_EPOCHS = 2
SEX_FINETUNE_EPOCHS = 25
SEX_NUM_EPOCHS = 100

NO_FINDING_WEIGHT_DECAY = 1e-4
NO_FINDING_HEAD_LR_STAGE_1 = 5e-5
NO_FINDING_HEAD_LR_STAGE_2 = 5e-6
NO_FINDING_BACKBONE_LR_STAGE_2 = 5e-8
NO_FINDING_MAX_GRAD_NORM = 1.0
NO_FINDING_BACKBONE_EPOCHS = 40
NO_FINDING_FINETUNE_EPOCHS = 0
NO_FINDING_NUM_EPOCHS = 100

AP_PA_WEIGHT_DECAY = 1e-4
AP_PA_HEAD_LR_STAGE_1 = 5e-4
AP_PA_HEAD_LR_STAGE_2 = 5e-4
AP_PA_BACKBONE_LR_STAGE_2 = 1e-7
AP_PA_MAX_GRAD_NORM = 1.0
AP_PA_BACKBONE_EPOCHS = 2
AP_PA_FINETUNE_EPOCHS = 25
AP_PA_NUM_EPOCHS = 100

AGE_WEIGHT_DECAY = 1e-4
AGE_HEAD_LR_STAGE_1 = 1e-3
AGE_HEAD_LR_STAGE_2 = 1e-3
AGE_BACKBONE_LR_STAGE_2 = 1e-6
AGE_MAX_GRAD_NORM = 1.0
AGE_BACKBONE_EPOCHS = 250
AGE_FINETUNE_EPOCHS = 50
AGE_NUM_EPOCHS = 100


TASK_PARAMS = {
    "sex": {
        "weight_decay": SEX_WEIGHT_DECAY,
        "head_lr_stage_1": SEX_HEAD_LR_STAGE_1,
        "head_lr_stage_2": SEX_HEAD_LR_STAGE_2,
        "backbone_lr_stage_2": SEX_BACKBONE_LR_STAGE_2,
        "max_grad_norm": SEX_MAX_GRAD_NORM,
        "backbone_epochs": SEX_BACKBONE_EPOCHS,
        "finetune_epochs": SEX_FINETUNE_EPOCHS,
        "num_epochs": SEX_NUM_EPOCHS,
    },
    "no finding": {
        "weight_decay": NO_FINDING_WEIGHT_DECAY,
        "head_lr_stage_1": NO_FINDING_HEAD_LR_STAGE_1,
        "head_lr_stage_2": NO_FINDING_HEAD_LR_STAGE_2,
        "backbone_lr_stage_2": NO_FINDING_BACKBONE_LR_STAGE_2,
        "max_grad_norm": NO_FINDING_MAX_GRAD_NORM,
        "backbone_epochs": NO_FINDING_BACKBONE_EPOCHS,
        "finetune_epochs": NO_FINDING_FINETUNE_EPOCHS,
        "num_epochs": NO_FINDING_NUM_EPOCHS,
    },
    "ap/pa": {
        "weight_decay": AP_PA_WEIGHT_DECAY,
        "head_lr_stage_1": AP_PA_HEAD_LR_STAGE_1,
        "head_lr_stage_2": AP_PA_HEAD_LR_STAGE_2,
        "backbone_lr_stage_2": AP_PA_BACKBONE_LR_STAGE_2,
        "max_grad_norm": AP_PA_MAX_GRAD_NORM,
        "backbone_epochs": AP_PA_BACKBONE_EPOCHS,
        "finetune_epochs": AP_PA_FINETUNE_EPOCHS,
        "num_epochs": AP_PA_NUM_EPOCHS,
    },
    "age": {
        "weight_decay": AGE_WEIGHT_DECAY,
        "head_lr_stage_1": AGE_HEAD_LR_STAGE_1,
        "head_lr_stage_2": AGE_HEAD_LR_STAGE_2,
        "backbone_lr_stage_2": AGE_BACKBONE_LR_STAGE_2,
        "max_grad_norm": AGE_MAX_GRAD_NORM,
        "backbone_epochs": AGE_BACKBONE_EPOCHS,
        "finetune_epochs": AGE_FINETUNE_EPOCHS,
        "num_epochs": AGE_NUM_EPOCHS,
    },
}


def normalize_task_name(name):
    if name is None:
        return ""
    return str(name).strip().lower()


def get_task_params(task_name):
    task_key = normalize_task_name(task_name)
    return TASK_PARAMS.get(task_key, TASK_PARAMS["age"])


class XRayClassifier(nn.Module):
    """
    ResNet backbone with a task-specific head.

    For relearn_classes and regression:
      - stage 1: train head only, plus backbone batch norm parameters
      - stage 2: train the entire backbone + head
    """

    def __init__(self, img_size, num_classes=3, regression=False, relearn_classes=False, resnet_size=34):
        super().__init__()
        self.regression = regression
        self.num_classes = num_classes
        if regression:
            self.num_classes = 1
        self.relearn_classes = relearn_classes
        self.img_size = img_size

        if resnet_size == 34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim

        if regression:
            self.head = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.feature_dim, self.feature_dim * 5),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(self.feature_dim * 5, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, 1),
            )
        elif relearn_classes:
            self.head = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.feature_dim, self.feature_dim * 5),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(self.feature_dim * 5, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        # if self.regression:
        #     return logits.squeeze(-1)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_batchnorm_parameters(self):
        """
        Return backbone BatchNorm parameters so they can stay trainable in stage 1.
        """
        bn_params = []
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_params.extend(list(module.parameters()))
        return bn_params

    def unfreeze_backbone_batchnorm(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    def set_training_stage(self, stage: int):
        """
        stage 1: head + backbone batch norm params
        stage 2: head + entire backbone
        """
        if not (self.relearn_classes or self.regression):
            for param in self.parameters():
                param.requires_grad = True
            return

        for param in self.head.parameters():
            param.requires_grad = True

        if stage == 1:
            self.freeze_backbone()
            self.unfreeze_backbone_batchnorm()
        else:
            self.unfreeze_backbone()

    def load_from_weights(self, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)

        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("resnet.fc."):
                remapped[k.replace("resnet.fc.", "head.")] = v
            else:
                remapped[k] = v

        incompatible = self.load_state_dict(remapped, strict=False)

        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("Loaded with key mismatches:")
            print("  Missing:", incompatible.missing_keys)
            print("  Unexpected:", incompatible.unexpected_keys)

        return self


class RandomCutout(nn.Module):
    """
    Randomly applies 1-3 cutouts of size 10-20 pixels with probability p.
    Works on tensors of shape [B, C, H, W] or [C, H, W].
    """

    def __init__(self, p=0.25, min_holes=1, max_holes=3, min_size=10, max_size=20, fill_value=0.0):
        super().__init__()
        self.p = p
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_size = min_size
        self.max_size = max_size
        self.fill_value = fill_value

    def forward(self, x):
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        b, c, h, w = x.shape
        out = x.clone()

        for i in range(b):
            if torch.rand(1, device=x.device).item() > self.p:
                continue

            num_holes = int(torch.randint(self.min_holes, self.max_holes + 1, (1,), device=x.device).item())

            for _ in range(num_holes):
                hole_h = int(torch.randint(self.min_size, self.max_size + 1, (1,), device=x.device).item())
                hole_w = int(torch.randint(self.min_size, self.max_size + 1, (1,), device=x.device).item())

                if hole_h >= h or hole_w >= w:
                    continue

                y0 = int(torch.randint(0, h - hole_h + 1, (1,), device=x.device).item())
                x0 = int(torch.randint(0, w - hole_w + 1, (1,), device=x.device).item())

                out[i, :, y0:y0 + hole_h, x0:x0 + hole_w] = self.fill_value

        return out.squeeze(0) if single else out


class XRayTrainAugment(nn.Module):
    """
    Approximate:
      - random rotation up to 20 degrees
      - zoom up to 110%
      - lighting/contrast changes up to 20% with 75% probability
      - 1-3 cutouts of 10-20 px with 25% probability
    """

    def __init__(self):
        super().__init__()
        self.cutout = RandomCutout(
            p=0.25,
            min_holes=1,
            max_holes=3,
            min_size=10,
            max_size=20,
            fill_value=0.0,
        )

    def forward(self, x):
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        out = []
        for img in x:
            img = self._augment_single(img)
            out.append(img)

        out = torch.stack(out, dim=0)
        out = self.cutout(out)
        return out.squeeze(0) if single else out

    def _augment_single(self, img):
        if torch.rand(1, device=img.device).item() < 0.75:
            angle = float(torch.empty(1, device=img.device).uniform_(-20.0, 20.0).item())
            scale = float(torch.empty(1, device=img.device).uniform_(0.90, 1.10).item())
            img = TF.affine(
                img,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0,
            )

        if torch.rand(1, device=img.device).item() < 0.75:
            brightness = float(torch.empty(1, device=img.device).uniform_(0.8, 1.2).item())
            contrast = float(torch.empty(1, device=img.device).uniform_(0.8, 1.2).item())
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)

        return img


class AugmentedDataset(Dataset):
    """
    Wrap a cached dataset and apply train-time augmentations only on the source image.
    """

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        source = sample["source"].clone()
        if self.transform is not None:
            source = self.transform(source)
        sample = dict(sample)
        sample["source"] = source
        return sample


def compute_metrics(y_true, y_probs, regression=False):
    if regression:
        return None, None, None, None

    num_classes = y_probs.shape[-1]
    y_true_oh = np.eye(num_classes)[y_true]

    y_pred = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true_oh, y_probs, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true_oh, y_probs, average="macro")
    except ValueError:
        pr_auc = float("nan")

    return accuracy, roc_auc, pr_auc, None


def configure_model_for_stage(model, stage):
    model.train()

    if isinstance(model, XRayClassifier):
        model.set_training_stage(stage)


def build_optimizer(model, task_params, regression=False, relearn_classes=False, stage=1):
    """
    For relearn_classes and regression:
      - stage 1: head + backbone batch norm params
      - stage 2: head + entire backbone
    """
    weight_decay = task_params["weight_decay"]

    if isinstance(model, XRayClassifier) and (relearn_classes or regression):
        if stage == 1:
            params = list(model.head.parameters()) + list(model.backbone_batchnorm_parameters())
            return optim.AdamW(
                params,
                lr=task_params["head_lr_stage_1"],
                weight_decay=weight_decay,
            )

        return optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": task_params["head_lr_stage_2"]},
                {"params": model.backbone.parameters(), "lr": task_params["backbone_lr_stage_2"]},
            ],
            weight_decay=weight_decay,
        )

    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=weight_decay,
    )


def train_one_epoch(loader, model, optimizer, criterion, target, max_grad_norm, stage=1):
    configure_model_for_stage(model, stage)

    total_loss = 0.0
    regression = isinstance(criterion, nn.MSELoss)

    for dict_batch in loader:
        x = dict_batch["source"].to(DEVICE, non_blocking=True)
        y = dict_batch["target"][:, target].to(DEVICE, non_blocking=True).nan_to_num(0)
        y = torch.clamp(y, min=0)

        if not regression:
            y = y.long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        loss = criterion(outputs, y)

        if torch.isnan(loss):
            print("NaN loss encountered, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def validate(loader, model, criterion, target):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []
    all_preds = []
    regression = isinstance(criterion, nn.MSELoss)

    with torch.no_grad():
        for dict_batch in loader:
            x = dict_batch["source"].to(DEVICE, non_blocking=True)
            y = dict_batch["target"][:, target].to(DEVICE, non_blocking=True).nan_to_num(0)
            y = torch.clamp(y, min=0)

            if not regression:
                y = y.long()

            outputs = model(x)
            loss = criterion(outputs, y)

            if not regression:
                if torch.isnan(outputs).any():
                    print("Had nans in validation outputs")
                probs = torch.softmax(outputs.float(), dim=1)
                all_targets.append(y.detach().cpu())
                all_probs.append(probs.cpu())
            else:
                all_targets.append(y.detach().cpu().float())
                all_preds.append(outputs.detach().cpu().float())

            total_loss += loss.item()

    if not regression:
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()
        accuracy, roc_auc, pr_auc, _ = compute_metrics(all_targets, all_probs)
        mae = None
    else:
        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        accuracy, roc_auc, pr_auc = None, None, None
        mae = mean_absolute_error(all_targets, all_preds)

    return total_loss / max(1, len(loader)), mae, accuracy, roc_auc, pr_auc


def build_datasets():
    train_dataset_path = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/classifier_datasets/train_dataset.pt"
    test_dataset_path = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/classifier_datasets/test_dataset.pt"

    if not os.path.exists(train_dataset_path):
        ds_train = CheXPert_Extended(
            image_resize=IMAGE_RESIZE,
            augment_horizontal_flip=False,
            augment_vertical_flip=False,
            path_root="/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/train_cheXbert.csv",
            use_cache=True,
            cache_dir="/tmp/$USER/chexpert_cache",
        )
    else:
        ds_train = None

    if not os.path.exists(test_dataset_path):
        ds_test = CheXPert_Extended(
            image_resize=IMAGE_RESIZE,
            augment_horizontal_flip=False,
            augment_vertical_flip=False,
            path_root="/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/valid.csv",
            use_cache=True,
            cache_dir="/tmp/$USER/chexpert_cache_valid",
        )
    else:
        ds_test = None
    
    dm = SimpleDataModule(
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        cache_dir="/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/classifier_datasets/",
    )

    train_dataset = AugmentedDataset(dm.ds_train, transform=XRayTrainAugment())
    val_dataset = dm.ds_test
    test_dataset = dm.ds_test
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, test_loader


def train_classifier(name, target, num_classes, save_dir=None):
    if save_dir is None:
        save_dir = f"./outputs_{name}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}"
    os.makedirs(save_dir, exist_ok=True)

    task_name = normalize_task_name(name)
    task_params = get_task_params(task_name)

    regression = target in REGRESSION_TARGET_INDICES or task_name == "age"
    relearn_classes = task_name == "ap/pa"

    if task_name == "sex":
        num_classes = 2

    train_dataset, val_dataset, test_dataset = build_datasets()
    train_loader, val_loader, test_loader = build_dataloaders(train_dataset, val_dataset, test_dataset)

    output_dim = 1 if regression else num_classes

    model = XRayClassifier(
        img_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        num_classes=output_dim,
        regression=regression,
        relearn_classes=relearn_classes,
        resnet_size=RESNET_SIZE,
    ).to(DEVICE)

    if task_name == "no finding":
        model = DiseaseClassifier().to(DEVICE)

    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
    }

    best_val_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pt")

    if isinstance(model, XRayClassifier) and (relearn_classes or regression):
        phase_specs = [
            {"stage": 1, "epochs": task_params["backbone_epochs"]},
            {"stage": 2, "epochs": task_params["finetune_epochs"]},
        ]
    else:
        phase_specs = [
            {"stage": 1, "epochs": task_params["num_epochs"]},
        ]

    global_epoch = 0

    for phase in phase_specs:
        if phase["epochs"] <= 0:
            continue

        optimizer = build_optimizer(
            model=model,
            task_params=task_params,
            regression=regression,
            relearn_classes=relearn_classes,
            stage=phase["stage"],
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )
        # scheduler = ExponentialLR(optimizer, gamma=0.96)

        for _ in range(phase["epochs"]):
            global_epoch += 1

            train_loss = train_one_epoch(
                train_loader,
                model,
                optimizer,
                criterion,
                target,
                max_grad_norm=task_params["max_grad_norm"],
                stage=phase["stage"],
            )
            val_loss, val_mae, accuracy, roc_auc, pr_auc = validate(val_loader, model, criterion, target)

            scheduler.step(val_loss)
            # scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["val_roc_auc"].append(roc_auc)
            history["val_pr_auc"].append(pr_auc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {global_epoch}")
            print(f"Train Loss: {train_loss:.4f}")

            if regression:
                print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
            elif accuracy is None:
                print(f"Val Loss: {val_loss:.4f} | Accuracy: N/A | ROC-AUC: N/A | PR-AUC: N/A")
            else:
                print(
                    f"Val Loss: {val_loss:.4f} | "
                    f"Accuracy: {accuracy:.4f} | "
                    f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
                )

    final_model_path = os.path.join(save_dir, "classifier_model.pt")
    torch.save(model.state_dict(), final_model_path)

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_loss, test_mae, test_accuracy, test_roc_auc, test_pr_auc = validate(test_loader, model, criterion, target)

    test_metrics = {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
    }

    test_metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("\n=== Test Results (Best Model) ===")
    print(f"Loss: {test_loss:.4f}")
    if regression:
        print(f"MAE: {test_mae:.4f}")
    if test_accuracy is not None:
        print(f"Accuracy: {test_accuracy:.4f}")
    if test_roc_auc is None:
        print("ROC-AUC: N/A")
        print("PR-AUC: N/A")
    else:
        print(f"ROC-AUC: {test_roc_auc:.4f}")
        print(f"PR-AUC: {test_pr_auc:.4f}")


if __name__ == "__main__":
    # Example:
    # train_classifier(name="sex", target=0, num_classes=2)
    pass
