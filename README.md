# CFGNP

Counterfactual Graph Neural Processes — amortised counterfactual estimation over structural
causal models. Instead of fitting one model per SCM, a single model is
conditioned on a set of observations from a system and predicts counterfactual outcomes
for a queried intervention, in the neural-process style.

Two model families are provided:

- **CFGNP** ([src/cfgnp/models/graph_model.py](src/cfgnp/models/graph_model.py))
  — the graph variant, where the causal DAG (the fully connected graph) is given to the model explicitly
  (`gnn_mode=True`).

- For the CheXpert image experiments, `VisionCFNP`
([src/cfgnp/image_data/image_model.py](src/cfgnp/image_data/image_model.py)) works in the
latent space of a VAE from [src/medical_diffusion/](src/medical_diffusion/), with
attribute classifiers/regressors ([src/cfgnp/classifier/](src/cfgnp/classifier/))
supervising the counterfactual attributes.

## Layout

| Path | Contents |
| --- | --- |
| [src/main.py](src/main.py) | Entry point; `--mode` selects the pipeline |
| [src/cfgnp/train_suite.py](src/cfgnp/train_suite.py) | Per-dataset experiment configs + `train_run` |
| [src/cfgnp/models/](src/cfgnp/models/) | Attention layers, tabular and graph CFNP models |
| [src/cfgnp/data/](src/cfgnp/data/) | Synthetic SCMs (M-shape, triangle, two-node, ellipse), German Loan, CheXpert |
| [src/cfgnp/training/](src/cfgnp/training/), [src/cfgnp/loss.py](src/cfgnp/loss.py) | Training loops, losses and per-index metrics |
| [src/cfgnp/experiments/](src/cfgnp/experiments/) | Two-node identifiability study |
| [src/medical_diffusion/](src/medical_diffusion/) | VAE / diffusion components for the X-ray experiments |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python ./src/main.py --mode=<task> --seed=<seed>
```

Available tasks (see `main` in [src/main.py](src/main.py)):

| `--mode` | What it does |
| --- | --- |
| `graph_train` | Train on the synthetic M-shape SCM |
| `chexpert_graph_train` | Train the graph CFNP on CheXpert (default) |
| `chexpert_ood_<k>` | CheXpert with OOD split `k` |
| `chexpert_individual_<k>` | Per-attribute CheXpert run `k` |
| `med_image` | Train the image-space model |
| `classifier` | Train an attribute classifier |
| `chexpert_count`, `chexpert_count_classification` | Precompute the CheXpert counterfactual datasets |
| `two_node`, `two_node_id`, `two_node_unid` | Two-node identifiability sweep |
| `benchmark` | Benchmark baselines on the artificial datasets |
| `bgm_train`, `bgm_train_mshape`, `bgm_train_triangle` | Train the OT-BCM baselines |

Datasets can be trained directly via `train_run(dataset_name, gnn_mode, num_epochs, save_code, lr)`;
`ExperimentFactory.create` resolves `dataset_name` to a config, so the supported names are
`artificial`, `mshape*`, `triangle*`, `german_loan`, `ellipse`, `chexpert`,
`chexpert_individual_<k>`, `chexpert_ood_<k>` and `two_node*`.

The CheXpert pipelines expect the precomputed counterfactual tensors to exist — run the
`chexpert_count*` modes first.


## Paths

Filesystem locations for CheXpert data, cached datasets and pretrained checkpoints live in
`ChexpertPath` ([src/cfgnp/util/data_paths.py](src/cfgnp/util/data_paths.py)). The defaults
point at the author's cluster storage; override them with the setters, or pass a JSON
config to `train_run(..., cfg_file=...)` which loads them via `ChexpertPath.load_from_json`.
