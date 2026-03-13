# TEGB-YOLO

TEGB-YOLO is a research framework for analyzing YOLO feature spaces with evidential classification, granular-ball construction, geometric three-way decision, and visualization-oriented embedding export.

The default pipeline uses:

- entropy-guided spherical granular-ball construction
- geometric spherical-collision three-way decision

It supports training, full evaluation, ablation, and figure export for `yolo11`, `yolo12`, and `yolo26` backbones.

## Highlights

- YOLO feature extraction through configurable hook layers
- trainable `EvidentialHead` for Dirichlet evidence and uncertainty
- trainable `VisProjectionHead` for 2D embedding supervision
- granular-ball builder with entropy-guided splitting
- pure geometric three-way decision by spherical collision
- VSF / GTF / SNH evaluation
- automatic artifact export:
  - class embeddings
  - grouped embeddings
  - collision overlays
  - uncertainty maps
  - granular-ball overlays
  - datamapplot-style semantic landscapes

## Repository Layout

- `configs/tegb/`: training, evaluation, and ablation configs
- `experiments/run_train.py`: training entry
- `experiments/run_eval.py`: evaluation entry
- `experiments/run_ablation.py`: ablation entry
- `experiments/summarize_results.py`: ablation summarization
- `tegb/models/`: YOLO adapter, evidential head, projection head
- `tegb/granular/`: granular-ball builders
- `tegb/decision/`: three-way decision modules
- `tegb/metrics/`: VSF, clustering, uncertainty, diagnostics
- `tegb/runner/`: trainer, evaluator, ablation runner, visualization exporter

## Requirements

Use your local Python environment, preferably your existing conda environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `scipy`
- `umap-learn`
- `gudhi`
- `pot`
- `matplotlib`
- `datamapplot`

## Dataset Format

The dataset reader expects YOLO-style image and label folders:

```text
images/
  xxx.jpg
  subdir/yyy.jpg
labels/
  xxx.txt
  subdir/yyy.txt
```

Each label file follows standard YOLO detection format:

```text
class_id center_x center_y width height
```

All paths are configured in the YAML file:

- `data.images_dir`
- `data.labels_dir`

## Supported Configs

Main configs:

- `configs/tegb/yolo11_coco.yaml`
- `configs/tegb/yolo12_coco.yaml`
- `configs/tegb/yolo26_coco.yaml`
- `configs/tegb/yolo11_coco128.yaml`
- `configs/tegb/yolo12_coco128.yaml`

Ablation configs:

- `configs/tegb/ablation_base.yaml`
- `configs/tegb/ablation_yolo11.yaml`
- `configs/tegb/ablation_yolo12.yaml`
- `configs/tegb/ablation_yolo26.yaml`

## Quick Start

### 1. Train

```bash
python experiments/run_train.py --config configs/tegb/yolo11_coco128.yaml
```

Override experiment name if needed:

```bash
python experiments/run_train.py --config configs/tegb/yolo11_coco128.yaml --exp-name tegb_demo
```

### 2. Evaluate

```bash
python experiments/run_eval.py --config configs/tegb/yolo11_coco128.yaml --ckpt runs/tegb_yolo11_coco128/checkpoints/best.pt
```

`run_eval` always evaluates on the full configured dataset by forcing `data.val_split_ratio=0.0` internally.

### 3. Run Ablation

Single config:

```bash
python experiments/run_ablation.py --config configs/tegb/ablation_base.yaml
```

Built-in suites:

```bash
python experiments/run_ablation.py --suite yolo11
python experiments/run_ablation.py --suite yolo12 --weights n,s,m,l,x
python experiments/run_ablation.py --suite all --weights s --summarize --summary-root runs --summary-out-dir runs/analysis
```

`--weights` accepts scale letters directly:

- `n`
- `s`
- `m`
- `l`
- `x`

Default is `s`.

## Key Config Fields

### Experiment

- `experiment.exp_name`
- `experiment.output_root`
- `experiment.model_family`
- `experiment.weights`
- `experiment.target_layer`
- `experiment.num_classes`

### Data

- `data.images_dir`
- `data.labels_dir`
- `data.image_size`
- `data.crop_size`
- `data.batch_size`
- `data.max_samples`

### Training / Loss

- `optim.epochs`
- `optim.lr`
- `optim.train_backbone`
- `loss.lambda_edl`
- `loss.lambda_gw`
- `loss.lambda_ph`
- `loss.lambda_hn`

### Granular Balls

- `granular.mode`
- `granular.radius_policy`
- `granular.radius_percentile`
- `granular.entropy_threshold`
- `granular.entropy_log_base`
- `granular.min_entropy_gain`
- `granular.max_split_depth`
- `granular.split_min_child`

### Three-Way Decision

- `decision.mode`
- `decision.collision_gamma`

### Visualization

- `visualization.embedding_source`
- `visualization.embedding_random_subset_per_class`
- `visualization.class_groups`
- `visualization.include_classes`
- `visualization.class_compare_mode`
- `visualization.class_compare_topk`
- `visualization.datamap_enabled`
- `visualization.balls_3d_enabled`

## Outputs

All outputs are written to:

```text
runs/<exp_name>/
```

Typical files:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics.json`
- `eval_metrics.json`
- `best_eval_metrics.json`
- `vsf_report.json`
- `three_way_regions.npy`
- `embeddings_2d.npy`
- `artifacts/granular_balls.json`
- `artifacts/collision_pairs.json`
- `artifacts/collision_mask.npy`
- `artifacts/nearest_ball_index.npy`
- `artifacts/manifold_diagnostics.json`
- `artifacts/embedding_payload.npz`
- `artifacts/visualization_meta.json`
- `artifacts/visualizations/*.png`

When ablation summarization is enabled, the analysis directory also includes:

- `ablation_summary_long.csv`
- `ablation_summary_wide.csv`
- `significance_tests.csv`
- `significance.json`
- `conclusion_report.json`

## Visualization Notes

The exporter supports:

- learned head embedding
- high-dimensional PCA + t-SNE / UMAP projection
- grouped class views
- semantic landscapes via `datamapplot`
- collision / uncertainty / three-way overlays

Label display is automatically shortened for long COCO class names in plots, while config parsing still uses the original class names.

## Typical Workflow

1. Choose a config from `configs/tegb/`
2. Set dataset paths and YOLO weights
3. Train with `run_train.py`
4. Evaluate with `run_eval.py`
5. Run ablations if needed
6. Use the exported figures and JSON artifacts for analysis

## License

See [LICENSE](LICENSE).
