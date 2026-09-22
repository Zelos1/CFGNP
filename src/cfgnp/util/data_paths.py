import json


class ChexpertPath(object):
    raw_data_csv = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/{mode}.csv"
    counterfactuals_path = "/data/coml-intersection-joins/lina4921/data/chexpert_new/chexpert_count_{mode}.pt"
    classification_dataset_path = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/classifier_datasets/{mode}_dataset.pt"
    classification_cache_dir = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/classifier_datasets/"
    graph_ds_path = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/chexpert_graph_count_new/{mode}.pt"
    counterfactuals_ood_path = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/chexpert_graph_count_ood_new/{mode}.pt"
    
    disease_config_file = "/data/coml-intersection-joins/lina4921/src_files/Chexpert/config/example.json"
    disease_pretrained_weights = "/data/coml-intersection-joins/lina4921/src_files/Chexpert/config/pre_train.pth"
    chexpert_finding_classifier_path = "/home/lina4921/MScThesis/CFNP/outputs_No Finding_29-06-2026_07-58/best_model.pt"
    chexpert_sex_classifier_path = "/home/lina4921/MScThesis/CFNP/outputs_Sex_04-07-2026_11-12/best_model.pt"
    chexpert_age_regressor_path = "/home/lina4921/MScThesis/CFNP/outputs_Age_03-07-2026_09-19/best_model.pt"
    chexpert_appa_classifier_path = "/home/lina4921/MScThesis/CFNP/outputs_AP/PA_04-07-2026_11-12/best_model.pt"
    chexpert_vae_path = "/home/lina4921/MScThesis/CFNP/runs/2026_06_16_012826/last.ckpt"
    final_path = "/data/coml-intersection-joins/lina4921/artifacts/"


        
    @classmethod
    def get_raw_data_csv(cls, mode):
        return cls.raw_data_csv.format(mode=mode)
    
    @classmethod
    def set_raw_data_csv(cls, path):
        cls.raw_data_csv = path

    @classmethod
    def get_counterfactuals_path(cls, mode):
        return cls.counterfactuals_path.format(mode=mode)
    
    @classmethod
    def set_counterfactuals_path(cls, path):
        cls.counterfactuals_path = path

    @classmethod
    def get_classification_dataset_path(cls, mode):
        return cls.classification_dataset_path.format(mode=mode)
    
    @classmethod
    def set_classification_dataset_path(cls, path):
        cls.classification_dataset_path = path
    
    @classmethod
    def get_classification_cache_dir(cls):
        return cls.classification_cache_dir
    
    @classmethod
    def set_classification_cache_dir(cls, path):
        cls.classification_cache_dir = path
    
    @classmethod
    def get_graph_ds_path(cls, mode):
        return cls.graph_ds_path.format(mode=mode)
    
    @classmethod
    def set_graph_ds_path(cls, path):
        cls.graph_ds_path = path
    
    @classmethod
    def get_graph_ds_ood_path(cls, mode):
        return cls.counterfactuals_ood_path.format(mode=mode)

    @classmethod
    def set_graph_ds_ood_path(cls, path):
        cls.counterfactuals_ood_path = path

    @classmethod
    def get_disease_config_file(cls):
        return cls.disease_config_file

    @classmethod
    def set_disease_cofig_file(cls, path):
        cls.disease_config_file = path
    
    @classmethod
    def get_disease_pretrained_model_path(cls):
        return cls.disease_pretrained_weights
    
    @classmethod
    def set_disease_pretrained_model_path(cls, path):
        cls.disease_pretrained_weights = path
    
    @classmethod
    def get_chexpert_finding_classifier_path(cls):
        return cls.chexpert_finding_classifier_path

    @classmethod
    def set_chexpert_finding_classifier_path(cls, path):
        cls.chexpert_finding_classifier_path = path

    @classmethod
    def get_chexpert_sex_classifier_path(cls):
        return cls.chexpert_sex_classifier_path

    @classmethod
    def set_chexpert_sex_classifier_path(cls, path):
        cls.chexpert_sex_classifier_path = path

    @classmethod
    def get_chexpert_age_regressor_path(cls):
        return cls.chexpert_age_regressor_path

    @classmethod
    def set_chexpert_age_regressor_path(cls, path):
        cls.chexpert_age_regressor_path = path

    @classmethod
    def get_chexpert_appa_classifier_path(cls):
        return cls.chexpert_appa_classifier_path

    @classmethod
    def set_chexpert_appa_classifier_path(cls, path):
        cls.chexpert_appa_classifier_path = path

    @classmethod
    def get_chexpert_vae_path(cls):
        return cls.chexpert_vae_path

    @classmethod
    def set_chexpert_vae_path(cls, path):
        cls.chexpert_vae_path = path

    @classmethod
    def set_final_path(cls, path):
        cls.final_path = path

    @classmethod
    def get_final_path(cls):
        return cls.final_path
    
    @classmethod
    def load_from_json(cls, json_path):
        if json_path is None:
            return  # No configuration file provided, use default paths
        with open(json_path, "r") as f:
            config = json.load(f)

        setters = {
            "raw_data_csv": cls.set_raw_data_csv,
            "counterfactuals_path": cls.set_counterfactuals_path,
            "classification_dataset_path": cls.set_classification_dataset_path,
            "classification_cache_dir": cls.set_classification_cache_dir,
            "chexpert_graph_ds_path": cls.set_graph_ds_path,
            "counterfactuals_ood_path": cls.set_graph_ds_ood_path,
            "disease_config_file": cls.set_disease_cofig_file,
            "disease_pretrained_weights": cls.set_disease_pretrained_model_path,
            "chexpert_finding_classifier_path": cls.set_chexpert_finding_classifier_path,
            "chexpert_sex_classifier_path": cls.set_chexpert_sex_classifier_path,
            "chexpert_age_regressor_path": cls.set_chexpert_age_regressor_path,
            "chexpert_appa_classifier_path": cls.set_chexpert_appa_classifier_path,
            "chexpert_vae_path": cls.set_chexpert_vae_path,
            "final_path": cls.set_final_path,
        }

        for key, value in config.items():
            if key not in setters:
                continue
            setters[key](value)


class OtherExperimentsPath(object):
    """Paths for the non-chexpert experiments: german_loan, mshape and triangle."""

    mshape_dataset_path = "/data/coml-intersection-joins/lina4921/data/artificial_data/m_data/dataset_{variant}_{seed}.pt"
    triangle_dataset_path = "/data/coml-intersection-joins/lina4921/data/artificial_data/triangle/dataset_{variant}_{seed}.pt"
    loan_dataset_path = "/data/coml-intersection-joins/lina4921/data/artificial_data/loan/dataset_seed_{seed}_{n_obs}.pt"
    graph_ds_path = "/data/coml-intersection-joins/lina4921/data/{dataset_name}/{dataset_name}_{split}_{seed}.pt"
    model_save_path = "/data/coml-intersection-joins/lina4921/artifacts/saved_models/{dataset_name}_{seed}_model.pt"

    loan_bgm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/loan_model_bgm_{seed}.pt"
    mshape_bgm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/mshape_model_bgm_{variant}_{seed}.pt"
    triangle_bgm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/triangle_model_bgm_{variant}_{seed}.pt"
    artificial_bgm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/artificial_model_bgm_{num_nodes}_{mechanism}_{seed}.pt"

    loan_cfm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/loan_model_cfm_{seed}.pt"
    mshape_cfm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/mshape_model_cfm_{variant}_{seed}.pt"
    triangle_cfm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/triangle_model_cfm_{variant}_{seed}.pt"
    artificial_cfm_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/artificial_model_cfm_{num_nodes}_{mechanism}_{seed}.pt"

    loan_vaca_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/loan_model_vaca_{seed}.pt"
    mshape_vaca_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/mshape_model_vaca_{variant}_{seed}.pt"
    triangle_vaca_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/triangle_model_vaca_{variant}_{seed}.pt"
    artificial_vaca_path = "/data/coml-intersection-joins/lina4921/artifacts/other_models/artificial_model_vaca_{num_nodes}_{mechanism}_{seed}.pt"

    two_node_model_dir = "/data/coml-intersection-joins/lina4921/artifacts/other_models/two_node"
    bivariate_model_dir = "/data/coml-intersection-joins/lina4921/artifacts/other_models/bivariate"

    @classmethod
    def get_mshape_dataset_path(cls, variant, seed):
        return cls.mshape_dataset_path.format(variant=variant, seed=seed)

    @classmethod
    def set_mshape_dataset_path(cls, path):
        cls.mshape_dataset_path = path

    @classmethod
    def get_triangle_dataset_path(cls, variant, seed):
        return cls.triangle_dataset_path.format(variant=variant, seed=seed)

    @classmethod
    def set_triangle_dataset_path(cls, path):
        cls.triangle_dataset_path = path

    @classmethod
    def get_loan_dataset_path(cls, seed, n_obs):
        return cls.loan_dataset_path.format(seed=seed, n_obs=n_obs)

    @classmethod
    def set_loan_dataset_path(cls, path):
        cls.loan_dataset_path = path

    @classmethod
    def get_graph_ds_path(cls, dataset_name, split, seed=42):
        return cls.graph_ds_path.format(dataset_name=dataset_name, split=split, seed=seed)

    @classmethod
    def set_graph_ds_path(cls, path):
        cls.graph_ds_path = path

    @classmethod
    def get_model_save_path(cls, dataset_name, seed=42):
        return cls.model_save_path.format(dataset_name=dataset_name, seed=seed)

    @classmethod
    def set_model_save_path(cls, path):
        cls.model_save_path = path

    @classmethod
    def get_loan_bgm_path(cls, seed):
        return cls.loan_bgm_path.format(seed=seed)

    @classmethod
    def set_loan_bgm_path(cls, path):
        cls.loan_bgm_path = path

    @classmethod
    def get_mshape_bgm_path(cls, variant, seed):
        return cls.mshape_bgm_path.format(variant=variant, seed=seed)

    @classmethod
    def set_mshape_bgm_path(cls, path):
        cls.mshape_bgm_path = path

    @classmethod
    def get_triangle_bgm_path(cls, variant, seed):
        return cls.triangle_bgm_path.format(variant=variant, seed=seed)

    @classmethod
    def set_triangle_bgm_path(cls, path):
        cls.triangle_bgm_path = path

    @classmethod
    def get_artificial_bgm_path(cls, num_nodes, mechanism, seed):
        return cls.artificial_bgm_path.format(num_nodes=num_nodes, mechanism=mechanism, seed=seed)

    @classmethod
    def set_artificial_bgm_path(cls, path):
        cls.artificial_bgm_path = path

    @classmethod
    def get_loan_cfm_path(cls, seed):
        return cls.loan_cfm_path.format(seed=seed)

    @classmethod
    def set_loan_cfm_path(cls, path):
        cls.loan_cfm_path = path

    @classmethod
    def get_mshape_cfm_path(cls, variant, seed):
        return cls.mshape_cfm_path.format(variant=variant, seed=seed)

    @classmethod
    def set_mshape_cfm_path(cls, path):
        cls.mshape_cfm_path = path

    @classmethod
    def get_triangle_cfm_path(cls, variant, seed):
        return cls.triangle_cfm_path.format(variant=variant, seed=seed)

    @classmethod
    def set_triangle_cfm_path(cls, path):
        cls.triangle_cfm_path = path

    @classmethod
    def get_artificial_cfm_path(cls, num_nodes, mechanism, seed):
        return cls.artificial_cfm_path.format(num_nodes=num_nodes, mechanism=mechanism, seed=seed)

    @classmethod
    def set_artificial_cfm_path(cls, path):
        cls.artificial_cfm_path = path

    @classmethod
    def get_loan_vaca_path(cls, seed):
        return cls.loan_vaca_path.format(seed=seed)

    @classmethod
    def set_loan_vaca_path(cls, path):
        cls.loan_vaca_path = path

    @classmethod
    def get_mshape_vaca_path(cls, variant, seed):
        return cls.mshape_vaca_path.format(variant=variant, seed=seed)

    @classmethod
    def set_mshape_vaca_path(cls, path):
        cls.mshape_vaca_path = path

    @classmethod
    def get_triangle_vaca_path(cls, variant, seed):
        return cls.triangle_vaca_path.format(variant=variant, seed=seed)

    @classmethod
    def set_triangle_vaca_path(cls, path):
        cls.triangle_vaca_path = path

    @classmethod
    def get_artificial_vaca_path(cls, num_nodes, mechanism, seed):
        return cls.artificial_vaca_path.format(num_nodes=num_nodes, mechanism=mechanism, seed=seed)

    @classmethod
    def set_artificial_vaca_path(cls, path):
        cls.artificial_vaca_path = path

    @classmethod
    def get_two_node_model_dir(cls):
        return cls.two_node_model_dir

    @classmethod
    def set_two_node_model_dir(cls, path):
        cls.two_node_model_dir = path

    @classmethod
    def get_bivariate_model_dir(cls):
        return cls.bivariate_model_dir

    @classmethod
    def set_bivariate_model_dir(cls, path):
        cls.bivariate_model_dir = path

    @classmethod
    def load_from_json(cls, json_path):
        if json_path is None:
            return  # No configuration file provided, use default paths
        with open(json_path, "r") as f:
            config = json.load(f)

        setters = {
            "mshape_dataset_path": cls.set_mshape_dataset_path,
            "triangle_dataset_path": cls.set_triangle_dataset_path,
            "loan_dataset_path": cls.set_loan_dataset_path,
            "graph_ds_path": cls.set_graph_ds_path,
            "model_save_path": cls.set_model_save_path,
            "loan_bgm_path": cls.set_loan_bgm_path,
            "mshape_bgm_path": cls.set_mshape_bgm_path,
            "triangle_bgm_path": cls.set_triangle_bgm_path,
            "loan_cfm_path": cls.set_loan_cfm_path,
            "mshape_cfm_path": cls.set_mshape_cfm_path,
            "triangle_cfm_path": cls.set_triangle_cfm_path,
            "loan_vaca_path": cls.set_loan_vaca_path,
            "mshape_vaca_path": cls.set_mshape_vaca_path,
            "triangle_vaca_path": cls.set_triangle_vaca_path,
            "two_node_model_dir": cls.set_two_node_model_dir,
            "bivariate_model_dir": cls.set_bivariate_model_dir,
        }

        for key, value in config.items():
            if key not in setters:
                continue
            setters[key](value)