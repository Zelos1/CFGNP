import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet34_Weights, resnet34, ResNet50_Weights, resnet50

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import CheXPert_Extended
from cfgnp.util.util import CLASS_MAPPING, DEVICE, REGRESSION_TARGET_INDICES, IMAGE_RESIZE
from cfgnp.util.data_paths import ChexpertPath
from cfgnp.classifier.disease_classifier import DiseaseClassifier



BATCH_SIZE = 128
NUM_WORKERS = 16
RESNET_SIZE = 50

SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5
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
NO_FINDING_HEAD_LR_STAGE_1 = 5e-4
NO_FINDING_HEAD_LR_STAGE_2 = 5e-4
NO_FINDING_BACKBONE_LR_STAGE_2 = 1e-7
NO_FINDING_MAX_GRAD_NORM = 1.0
NO_FINDING_BACKBONE_EPOCHS = 2
NO_FINDING_FINETUNE_EPOCHS = 25
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
AGE_HEAD_LR_STAGE_1 = 5e-4
AGE_HEAD_LR_STAGE_2 = 5e-4
AGE_BACKBONE_LR_STAGE_2 = 1e-7
AGE_MAX_GRAD_NORM = 1.0
AGE_BACKBONE_EPOCHS = 15
AGE_FINETUNE_EPOCHS = 25
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
        if self.regression:
            self.num_classes = 1
        self.relearn_classes = relearn_classes
        self.img_size = img_size

        if resnet_size == 34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        feature_dim = backbone.fc.in_features

        if regression or relearn_classes:
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
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim // 2, num_classes),
            )
        else:
            self.global_pool = nn.AvgPool2d(img_size)
            self.output_dim = num_classes
            self.head = nn.Linear(1000, self.output_dim)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
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
        x = dict_batch["source"].to(DEVICE)
        y = dict_batch["target"][:, target].to(DEVICE).nan_to_num(0)
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
            x = dict_batch["source"].to(DEVICE)
            y = dict_batch["target"][:, target].to(DEVICE).nan_to_num(0)
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

    train_dataset_path = ChexpertPath.get_classification_dataset_path("train")
    test_dataset_path = ChexpertPath.get_classification_dataset_path("test")

    if not os.path.exists(train_dataset_path):
        ds_3 = CheXPert_Extended(
            image_resize=IMAGE_RESIZE,
            augment_horizontal_flip=False,
            augment_vertical_flip=False,
            path_root=ChexpertPath.get_raw_data_csv("train_cheXbert"),
            use_cache=True,
            cache_dir="/tmp/$USER/chexpert_cache",
        )
    else:
        ds_3 = None

    if not os.path.exists(test_dataset_path):
        ds_test = CheXPert_Extended(
            image_resize=IMAGE_RESIZE,
            augment_horizontal_flip=False,
            augment_vertical_flip=False,
            path_root=ChexpertPath.get_raw_data_csv("valid"),
            use_cache=True,
            cache_dir="/tmp/$USER/chexpert_cache_valid",
        )
    else:
        ds_test = None

    dm = SimpleDataModule(
        ds_train=ds_3,
        ds_test=ds_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        cache_dir=ChexpertPath.get_classification_cache_dir(),
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

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