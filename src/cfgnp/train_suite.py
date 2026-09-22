import os
import json
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

from cfgnp.util.util import DEVICE, mape_loss, set_mog_var_floor, REGRESSION_TARGET_INDICES, NUM_CLASS_CHEXPERT, IMAGE_RESIZE
from cfgnp.util.data_paths import ChexpertPath, OtherExperimentsPath
from cfgnp.data import create_dataset_artifical, generate_cf_ellipse, create_dataset_Mshape, create_dataset_triangle, create_dataset_loan, CheXpertClassificationDataset
from cfgnp.graph_approach.train_graph import create_chexpert_reduced_ood_dataset, create_classified_ood_graph_dataset, create_classified_ood_masked_graph_dataset, create_dataset_artificial_graph
from cfgnp.models.size_invariant_model import build_cfnp_gnn_model
from cfgnp.models.graph_model import build_cfnp_gnn_model as not_size_invariant_model
from cfgnp.models.model import build_cfnp_model
from cfgnp.training.train import train_counterfactual
from cfgnp.image_data.image_model import VisionCFNP, factual_image_from_batch
from cfgnp.classifier import XRayClassifier, DiseaseClassifier
from cfgnp.loss import (ReconstructionRegularizedLoss, ReorderedMixedTargetLoss, compute_per_index_metrics_loader, eval_dloader_image_loss)


def _unwrap_base_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


class BaseExperimentConfig(ABC):
    def __init__(self, dataset_name: str, gnn_mode: bool, num_epochs: int = 100, save_code: bool = True, lr: float = 5e-5, seed: int = 42, size_invariant=True):
        self.dataset_name = dataset_name
        self.gnn_mode = gnn_mode
        self.num_epochs = num_epochs
        self.save_code = save_code
        self.lr = lr
        self.seed = seed
        self.device = DEVICE
        self.size_invariant = size_invariant
        self._set_defaults()
        self.configure()

    def _set_defaults(self):
        self.limit_paral_layers = False

        self.num_nodes = 5
        self.num_samples = 10000
        self.batch_size = 128
        self.feature_dim = 5
        self.hidden_dim = 80
        self.num_heads_att = 7
        self.num_count = 1
        self.num_obs = 500
        self.train_loss_indices = True
        self.loss_fn = nn.MSELoss()
        self.target_indices = list(range(self.num_nodes))
        self.target_indice_map = None
        self.ood_target_indice_map = None
        self.regression_target_indices = REGRESSION_TARGET_INDICES
        self.train_fn = None
        self.feat_indices = None
        self.model = None
        self.preloaded_split = False
        self.output_classified = False
        self.recon_target_fn = None
        self.dataset = self.train_dataset = self.val_dataset = self.test_dataset = None
        self.graph_ds_path = self.train_cache_path = self.val_cache_path = self.test_cache_path = None

    @abstractmethod
    def configure(self):
        raise NotImplementedError

    def split_datasets(self):
        if self.preloaded_split:
            return
        train_size = int(0.75 * len(self.dataset))
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, len(self.dataset) - train_size])
        val_size = int(0.3 * len(self.train_dataset))
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [len(self.train_dataset) - val_size, val_size])

    def build_graph_datasets(self):
        if not self.gnn_mode:
            return
        self.train_dataset = self._to_graph_dataset(self.train_dataset, self.train_cache_path, "train")
        self.val_dataset = self._to_graph_dataset(self.val_dataset, self.val_cache_path, "val")
        self.test_dataset = self._to_graph_dataset(self.test_dataset, self.test_cache_path, "test")

    def _to_graph_dataset(self, dataset, cache_path, split_name):
        default_cache_path = OtherExperimentsPath.get_graph_ds_path(self.dataset_name, split_name, self.seed)
        return create_dataset_artificial_graph(_unwrap_base_dataset(dataset), cache_path=cache_path or default_cache_path)

    def build_loaders(self):
        loader_cls = GeoDataLoader if self.gnn_mode else DataLoader
        self.train_loader = loader_cls(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = loader_cls(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = loader_cls(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        if self.model is not None:
            return
        if self.gnn_mode:
            if self.size_invariant:
                self.model = build_cfnp_gnn_model(hidden_dim=self.hidden_dim, in_features=self.feature_dim, num_count=self.num_count, num_nodes=self.num_nodes, num_obs=self.num_obs, mog_comp=3, num_heads_att=self.num_heads_att, limit_paral_layers=self.limit_paral_layers)
            else:
                self.model = not_size_invariant_model(hidden_dim=self.hidden_dim, in_features=self.feature_dim, num_count=self.num_count, num_nodes=self.num_nodes, num_obs=self.num_obs, mog_comp=3, num_heads_att=self.num_heads_att)
        else:
            self.model = build_cfnp_model(hidden_dim=self.hidden_dim, in_features=self.feature_dim, num_count=self.num_count, num_obs=self.num_obs, mog_comp=3)

    def build(self):
        self.split_datasets()
        self.build_graph_datasets()
        self.build_loaders()
        self.build_model()
        return self


class ArtificialExperimentConfig(BaseExperimentConfig):
    def configure(self):
        self.hidden_dim = 50
        self.batch_size = 32
        self.limit_paral_layers = True
        self.dataset = create_dataset_artifical(feature_dim=self.feature_dim, num_nodes=self.num_nodes, num_samples=self.num_samples, batch_size=self.batch_size, num_obs=self.num_obs, num_count=self.num_count)


class ArtificialSizeExperimentConfig(BaseExperimentConfig):
    def __init__(self, num_nodes:int, mechanism: str, seed: int, *args, **kwargs):
        self.n_nodes = num_nodes
        self.n_mechanism = mechanism
        self.n_seed = seed
        super().__init__(*args, **kwargs)

    def configure(self):
        self.hidden_dim = 50
        self.batch_size = 36
        self.num_nodes = self.n_nodes
        self.mechanism = self.n_mechanism
        self.seed = self.n_seed
        self.num_obs = 100
        self.limit_paral_layers = True
        self.noise_scale = 1.0
        self.dataset = create_dataset_artifical(feature_dim=self.feature_dim, num_nodes=self.num_nodes, num_samples=self.num_samples, num_obs=self.num_obs, num_count=self.num_count, batch_size=self.batch_size, seed=self.seed, mechanism=self.mechanism, noise_scale=self.noise_scale)
        self.target_indice_map = torch.tensor([[j for j in range(self.num_nodes) if j != i] for i in range(self.num_nodes)], dtype=torch.int64)

        if self.mechanism != "linear":
            set_mog_var_floor(1e-2)

class GraphMShapeExperimentConfig(BaseExperimentConfig):
    def configure(self):
        self.num_obs = 100
        self.num_samples = 50000
        self.batch_size = 256
        self.feature_dim = 1
        self.hidden_dim = 30
        self.num_count = 1
        self.num_nodes = 5
        self.target_indice_map = torch.tensor([[2, 3], [3, 4]], dtype=torch.int64)
        self.target_indices = None
        _, variant, seed = self.dataset_name.split("_")
        self.seed = int(seed)
        self.dataset = create_dataset_Mshape(self.num_samples, n_obs=self.num_obs, variant=variant, counterfactuals=True, seed=self.seed)


class GraphTriangleExperimentConfig(BaseExperimentConfig):
    def configure(self):
        self.num_obs = 100
        self.num_samples = 50000
        self.batch_size = 256
        self.feature_dim = 1
        self.hidden_dim = 30
        self.num_count = 1
        self.num_nodes = 3
        self.target_indice_map = torch.tensor([[1, 2], [2, 2]], dtype=torch.int64)
        self.target_indices = None
        _, variant, seed = self.dataset_name.split("_")
        self.seed = int(seed)
        self.dataset = create_dataset_triangle(self.num_samples, n_obs=self.num_obs, variant=variant, counterfactuals=True, seed=self.seed)


class GermanLoanExperimentConfig(BaseExperimentConfig):
    def configure(self):
        self.num_obs = 100
        self.num_heads_att = 8
        self.num_nodes = 7
        self.num_samples = 50000
        self.batch_size = 128
        self.feature_dim = 1
        self.hidden_dim = 32
        self.num_count = 1
        self.seed = int(self.dataset_name.split("_")[-1])
        self.dataset = create_dataset_loan(self.num_samples, self.num_obs, counterfactuals=True, seed=self.seed)
        self.target_indice_map = torch.tensor(7 * [[5, 6]], dtype=torch.int64)
        self.target_indices = None
        
class EllipseExperimentConfig(BaseExperimentConfig):
    def configure(self):
        self.num_obs = 100
        self.num_nodes = 5
        self.num_samples = 10000
        self.batch_size = 32
        self.feature_dim = 1
        self.hidden_dim = 10
        self.num_heads_att = 7
        self.num_count = 1
        self.loss_fn = mape_loss
        self.target_indices = [3, 4]
        self.dataset = generate_cf_ellipse(self.num_samples, self.num_obs)


class CheXpertGeneralConfig(BaseExperimentConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "size_invariant"):
            self.size_invariant = True

    def configure(self):
        self.batch_size = 32
        self.preloaded_split = True
        self.feature_dim = 256
        self.num_nodes = 12
        self.node_dim = 256
        self.hidden_dim = 256
        self.num_heads_att = 9
        self.num_count = 1
        self.num_obs = 100
        self.feat_indices = 4
        self.output_classified = True
        self.recon_weight = 10.0
        disease_classifier = DiseaseClassifier()
        disease_classifier.load_state_dict(torch.load(ChexpertPath.get_chexpert_finding_classifier_path(), map_location="cpu"))
        target_classifiers = [
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), num_classes=NUM_CLASS_CHEXPERT["Sex"], resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_sex_classifier_path(), map_location="cpu"),
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), regression=True, resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_age_regressor_path(), map_location="cpu"),
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), num_classes=NUM_CLASS_CHEXPERT["AP/PA"], relearn_classes=True, resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_appa_classifier_path(), map_location="cpu"),
            disease_classifier,
        ]
        self.target_classifiers = target_classifiers
        self.model = VisionCFNP(vae_path=ChexpertPath.get_chexpert_vae_path(), target_classifiers=target_classifiers, num_features=self.feat_indices, 
                                output_nodes=5, node_dim=self.node_dim, in_features=self.feature_dim, hidden_dim=self.hidden_dim, 
                                num_nodes=self.num_nodes, num_obs=self.num_obs, num_count=self.num_count, iterations=3, mog_comp=3, 
                                num_heads_att=self.num_heads_att, freeze_vae=True, freeze_classifiers=True, size_invariant=self.size_invariant)
        target_loss = ReorderedMixedTargetLoss(self.model.class_counts, regression_target_indices=REGRESSION_TARGET_INDICES)
        self.train_fn = self.loss_fn = ReconstructionRegularizedLoss(target_loss, weight=self.recon_weight)
        self.recon_target_fn = lambda batch: factual_image_from_batch(batch, self.feat_indices)
        self.target_indice_map = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.int64)

    def split_datasets(self):
        val_size = int(0.3 * len(self.train_dataset))
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [len(self.train_dataset) - val_size, val_size])


class CheXpertExperimentConfig(CheXpertGeneralConfig):
    def __init__(self, *args, **kwargs):
        self.size_invariant = False
        super().__init__(*args, **kwargs)
    
    def configure(self):
        super().configure()
        self.train_cache_path = ChexpertPath.get_graph_ds_path("train")
        self.val_cache_path = ChexpertPath.get_graph_ds_path("val")
        self.test_cache_path = ChexpertPath.get_graph_ds_path("test")
        print("Cache paths:", self.train_cache_path, self.val_cache_path, os.path.exists(self.train_cache_path), os.path.exists(self.val_cache_path))
        self.train_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("train_cheXbert"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="train_cheXbert")).make_counterfactuals(self.num_obs) if not (os.path.exists(self.train_cache_path) and os.path.exists(self.val_cache_path)) else []
        self.test_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("valid"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="valid")).make_counterfactuals(self.num_obs) if not os.path.exists(self.test_cache_path) else []

    def build_graph_datasets(self):
        if not self.gnn_mode:
            return
        self.train_dataset = create_dataset_artificial_graph(self.train_dataset, cache_path=self.train_cache_path, target_num_nodes=4)
        self.val_dataset = create_dataset_artificial_graph(self.val_dataset, cache_path=self.val_cache_path, target_num_nodes=4)
        self.test_dataset = create_dataset_artificial_graph(self.test_dataset, cache_path=self.test_cache_path, target_num_nodes=4)

class CheXpertOODNodeDropConfig(CheXpertGeneralConfig):
    def __init__(self, ood_index, masked, *args, **kwargs):
        self.ood_index = ood_index
        super().__init__(*args, **kwargs)

    def configure(self):
        super().configure()
        self.batch_size = 32
        self.num_heads_att = 9
        self.hidden_dim = 256
        self.recon_weight = 25.0
        self.num_nodes = 11

        disease_classifier = DiseaseClassifier()
        disease_classifier.load_state_dict(torch.load(ChexpertPath.get_chexpert_finding_classifier_path(), map_location="cpu"))
        target_classifiers = [
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), num_classes=NUM_CLASS_CHEXPERT["Sex"], resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_sex_classifier_path(), map_location="cpu"),
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), regression=True, resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_age_regressor_path(), map_location="cpu"),
            XRayClassifier(img_size=(IMAGE_RESIZE, IMAGE_RESIZE), num_classes=NUM_CLASS_CHEXPERT["AP/PA"], relearn_classes=True, resnet_size=50).load_from_weights(ChexpertPath.get_chexpert_appa_classifier_path(), map_location="cpu"),
            disease_classifier,
        ]
        target_classifiers = [t_cls for i, t_cls in enumerate(target_classifiers) if i != self.ood_index]
        self.target_classifiers = target_classifiers
        self.model = VisionCFNP(vae_path=ChexpertPath.get_chexpert_vae_path(), target_classifiers=target_classifiers, num_features=self.feat_indices - 1, output_nodes=None, node_dim=None, in_features=self.feature_dim, hidden_dim=self.hidden_dim, num_nodes=self.num_nodes, num_obs=self.num_obs, num_count=self.num_count, iterations=3, mog_comp=3, num_heads_att=self.num_heads_att)

        self.regression_target_indices = [i - (i > self.ood_index) for i in REGRESSION_TARGET_INDICES if i != self.ood_index]
        target_loss = ReorderedMixedTargetLoss(self.model.class_counts, regression_target_indices=self.regression_target_indices)
        self.train_fn = self.loss_fn = ReconstructionRegularizedLoss(target_loss, weight=self.recon_weight, lpips_weight=4, tv_weight=3)
        self.recon_target_fn = lambda batch: factual_image_from_batch(batch, self.feat_indices - 1)
        self.target_indice_map = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.int64)
        self.ood_target_indice_map = torch.arange(self.feat_indices - 1, dtype=torch.int64).unsqueeze(0)

        self.train_cache_path = ChexpertPath.get_graph_ds_ood_path(f"train_id_drop_{self.ood_index}")
        self.val_cache_path = ChexpertPath.get_graph_ds_ood_path(f"val_id_drop_{self.ood_index}")
        self.test_cache_path = ChexpertPath.get_graph_ds_ood_path(f"test_id_drop_{self.ood_index}")
        self.ood_test_cache_path = ChexpertPath.get_graph_ds_ood_path(f"test_ood_{self.ood_index}")
        self.train_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("train_cheXbert"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="train_cheXbert")).make_counterfactuals(self.num_obs) if not (os.path.exists(self.train_cache_path) and os.path.exists(self.val_cache_path)) else []
        self.test_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("valid"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="valid")).make_counterfactuals(self.num_obs) if not (os.path.exists(self.test_cache_path) and os.path.exists(self.ood_test_cache_path)) else []

    def build_graph_datasets(self):
        if not self.gnn_mode:
            return
        ood_creation_method = create_chexpert_reduced_ood_dataset
        self.train_dataset = ood_creation_method(self.train_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.train_cache_path, target_num_nodes=4)
        self.val_dataset = ood_creation_method(self.val_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.val_cache_path, target_num_nodes=4)
        original_test_dataset = self.test_dataset
        self.test_dataset = ood_creation_method(original_test_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.test_cache_path, target_num_nodes=4)
        self.ood_test_dataset = ood_creation_method(original_test_dataset, holdout_int_idx=self.ood_index, keep_holdout=True, cache_path=self.ood_test_cache_path, target_num_nodes=4)


class CheXpertOODConfig(CheXpertGeneralConfig):
    def __init__(self, ood_index, masked, *args, **kwargs):
        self.ood_index = ood_index
        self.masked = masked
        super().__init__(*args, **kwargs)

    def configure(self):
        super().configure()
        num_targets = 4
        full_rows = [[j for j in range(num_targets) if j != i] for i in range(num_targets)]
        if self.masked:
            width = num_targets - 2
            self.target_indice_map = torch.tensor(
                [[j for j in row if j != self.ood_index][:width] for row in full_rows], dtype=torch.int64
            )
            self.ood_target_indice_map = torch.tensor(full_rows, dtype=torch.int64)
        else:
            self.target_indice_map = torch.tensor(full_rows, dtype=torch.int64)

        self.batch_size = 36
        self.num_heads_att = 9
        self.hidden_dim = 256
        self.recon_weight = 25.0
        self.model.num_features = num_targets - 1
        target_loss = ReorderedMixedTargetLoss(self.model.class_counts, regression_target_indices=REGRESSION_TARGET_INDICES)
        self.train_fn = self.loss_fn = ReconstructionRegularizedLoss(target_loss, weight=self.recon_weight, lpips_weight=4, tv_weight=3)
        
        self.train_cache_path = ChexpertPath.get_graph_ds_ood_path(f"train_id_drop_{self.ood_index}")
        self.val_cache_path = ChexpertPath.get_graph_ds_ood_path(f"val_id_drop_{self.ood_index}")
        self.test_cache_path = ChexpertPath.get_graph_ds_ood_path(f"test_id_drop_{self.ood_index}")
        self.train_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("train_cheXbert"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="train_cheXbert")).make_counterfactuals(self.num_obs) if not (os.path.exists(self.train_cache_path) and os.path.exists(self.val_cache_path)) else []
        self.test_dataset = CheXpertClassificationDataset(self.model.vae, ChexpertPath.get_raw_data_csv("valid"), "notenc", save_path=ChexpertPath.get_counterfactuals_path(mode="valid")).make_counterfactuals(self.num_obs) if not os.path.exists(self.test_cache_path) else []

    def build_graph_datasets(self):
        if not self.gnn_mode:
            return
        ood_creation_method = create_classified_ood_graph_dataset if not self.masked else create_classified_ood_masked_graph_dataset
        self.train_dataset = ood_creation_method(self.train_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.train_cache_path, target_num_nodes=4)
        self.val_dataset = ood_creation_method(self.val_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.val_cache_path, target_num_nodes=4)
        original_test_dataset = self.test_dataset
        self.test_dataset = ood_creation_method(original_test_dataset, holdout_int_idx=self.ood_index, keep_holdout=False, cache_path=self.test_cache_path, target_num_nodes=4)
        self.ood_test_dataset = ood_creation_method(original_test_dataset, holdout_int_idx=self.ood_index, keep_holdout=True, cache_path=ChexpertPath.get_graph_ds_ood_path(f"test_ood_drop_{self.ood_index}"), target_num_nodes=4)


class ChexpertIndividualConfig(CheXpertExperimentConfig):
    def __init__(self, target_index, *args, **kwargs):
        self.target_index = target_index
        super().__init__(*args, **kwargs)
        self.target_indice_map = torch.tensor([[target_index], [target_index], [target_index], [target_index]], dtype=torch.int64)
        self.batch_size = 48
        self.num_heads_att = 9
        self.hidden_dim = 256


class ExperimentFactory:
    @staticmethod
    def create(dataset_name: str, gnn_mode: bool, num_epochs: int = 100, save_code: bool = True, lr: float = 5e-5, seed: int = 42):
        if dataset_name == "artificial": return ArtificialExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr, seed=seed)
        if dataset_name.startswith("artificial_"):
            # "artificial_{num_nodes}_{mechanism}_{seed}"; mechanism may itself contain
            # underscores (e.g. "sum_sine").
            parts = dataset_name.split("_")
            num_nodes_str = parts[1]
            mechanism = "_".join(parts[2:]) or "linear"
            return ArtificialSizeExperimentConfig(int(num_nodes_str), mechanism, seed, dataset_name, gnn_mode, num_epochs, save_code, lr)
        if dataset_name.startswith("mshape"): return GraphMShapeExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr, seed=seed)
        if dataset_name.startswith("triangle"): return GraphTriangleExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr, seed=seed)
        if dataset_name.startswith("german_loan"): return GermanLoanExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr, seed=seed)
        if dataset_name == "ellipse": return EllipseExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr)
        if dataset_name == "chexpert": return CheXpertExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr)
        if dataset_name.startswith("chexpert_individual"): return ChexpertIndividualConfig(int(dataset_name.split("_")[-1]), dataset_name, gnn_mode, num_epochs, save_code, lr)
        if dataset_name.startswith("chexpert_ood"): return CheXpertOODNodeDropConfig(int(dataset_name.split("_")[-1]), True, dataset_name, gnn_mode, num_epochs, save_code, lr)
        if dataset_name.startswith("two_node"):
            from cfgnp.experiments.two_node_identifiability import TwoNodeExperimentConfig
            return TwoNodeExperimentConfig(dataset_name, gnn_mode, num_epochs, save_code, lr)
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


def train_run(dataset_name: str, gnn_mode: bool, num_epochs: int = 100, save_code: bool = True, lr: float = 5e-5, save_path: str = None, cfg_file: str = None, use_acceleration = None, seed=42):
    ChexpertPath.load_from_json(cfg_file)
    OtherExperimentsPath.load_from_json(cfg_file)
    torch.manual_seed(seed)
    config = ExperimentFactory.create(dataset_name, gnn_mode, num_epochs, save_code, lr).build()
    if save_path is None:
        if dataset_name.startswith("chexpert"):
            date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            save_path = f"/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/saved_models/{dataset_name}/{dataset_name}_{seed}_model.pt"
        else:
            save_path = OtherExperimentsPath.get_model_save_path(dataset_name, config.seed)

    torch.manual_seed(seed)
    results = train_counterfactual(config.gnn_mode, config.model, config.train_loader, config.val_loader, config.test_loader, config.device, config.loss_fn, 
                                   config.target_indices, n_epochs=config.num_epochs, patience=25, save_path=save_path, print_every=1, 
                                   train_loss_indices=config.train_loss_indices, target_indice_map=config.target_indice_map, lr=config.lr, 
                                   train_fn=config.train_fn, output_classified=config.output_classified, use_acceleration=use_acceleration,
                                   recon_target_fn=config.recon_target_fn)

    if dataset_name.startswith("chexpert_ood"):
        ood_loader = GeoDataLoader(config.ood_test_dataset, batch_size=4, shuffle=False)
        ood_map = config.ood_target_indice_map if config.ood_target_indice_map is not None else config.target_indice_map
        results["ood_test_loss"] = eval_dloader_image_loss(config.train_fn, ood_loader, config.model, config.recon_target_fn,
                                                           target_indice_map=ood_map, device=config.device)

    if results["is_main_process"]:
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        os.makedirs(f"./results/{config.dataset_name}", exist_ok=True)
        results["filename"] = f"./results/{config.dataset_name}/graph_model_{date_time}.json"
        model_f_name = "./src/cfgnp/model.py" if not config.gnn_mode else "./src/cfgnp/models/graph_model.py"
        results["seed"] = seed
        results["code_base"] = open(model_f_name, "r").read() if config.save_code else "Code base was not logged"
        with open(results["filename"], "w") as f:
            json.dump(results, f, indent=4)

        if dataset_name.startswith("chexpert"):
            scores = compute_per_index_metrics_loader(config.test_loader, config.model, config.model.class_counts, config.regression_target_indices, config.target_indice_map, config.device)
            print(scores)
            results["scores"] = scores
            with open(results["filename"], "w") as f:
                json.dump(results, f, indent=4)

    return results