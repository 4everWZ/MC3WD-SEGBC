# TEGB-YOLO

TEGB-YOLO 是一个面向 YOLO 特征空间分析的研究框架，提供证据分类头、粒球构建、几何三支决策、投影评估与图像导出能力。

当前默认主线使用：

- 基于熵分裂的球形粒球构建
- 基于球碰撞的几何三支决策

支持 `yolo11`、`yolo12`、`yolo26` 三个系列的训练、评估、消融和可视化导出。

## 功能概览

- 通过可配置 hook 层提取 YOLO 特征
- `EvidentialHead` 输出 Dirichlet evidence 与 uncertainty
- `VisProjectionHead` 提供 2D 投影监督
- 粒球构建：熵驱动分裂 + 分位数半径
- 三支决策：纯几何球碰撞判定
- 指标：`VSF / GTF / SNH / FPR95 / Coverage@Risk`
- 图像导出：
  - 类别嵌入图
  - 分组嵌入图
  - 三支区域图
  - 碰撞覆盖图
  - 不确定性热力图
  - 粒球 overlay / 3D 粒球图
  - `datamapplot` 语义景观图

## 目录结构

- `configs/tegb/`：训练、评估、消融配置
- `experiments/run_train.py`：训练入口
- `experiments/run_eval.py`：评估入口
- `experiments/run_ablation.py`：消融入口
- `experiments/summarize_results.py`：消融结果汇总
- `tegb/models/`：YOLO 适配器、证据头、投影头
- `tegb/granular/`：粒球构建
- `tegb/decision/`：三支决策
- `tegb/metrics/`：VSF、聚类、不确定性、诊断指标
- `tegb/runner/`：训练器、评估器、消融执行器、可视化导出

## 环境安装

建议直接使用你本地已有的 Python / conda 环境。

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

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

## 数据格式

数据读取器使用 YOLO 风格的图像与标签目录：

```text
images/
  xxx.jpg
  subdir/yyy.jpg
labels/
  xxx.txt
  subdir/yyy.txt
```

标签文件格式：

```text
class_id center_x center_y width height
```

路径在 YAML 里配置：

- `data.images_dir`
- `data.labels_dir`

## 配置文件

主配置：

- `configs/tegb/yolo11_coco.yaml`
- `configs/tegb/yolo12_coco.yaml`
- `configs/tegb/yolo26_coco.yaml`
- `configs/tegb/yolo11_coco128.yaml`
- `configs/tegb/yolo12_coco128.yaml`

消融配置：

- `configs/tegb/ablation_base.yaml`
- `configs/tegb/ablation_yolo11.yaml`
- `configs/tegb/ablation_yolo12.yaml`
- `configs/tegb/ablation_yolo26.yaml`

## 快速开始

### 1. 训练

```bash
python experiments/run_train.py --config configs/tegb/yolo11_coco128.yaml
```

如果需要指定新的实验名：

```bash
python experiments/run_train.py --config configs/tegb/yolo11_coco128.yaml --exp-name tegb_demo
```

### 2. 评估

```bash
python experiments/run_eval.py --config configs/tegb/yolo11_coco128.yaml --ckpt runs/tegb_yolo11_coco128/checkpoints/best.pt
```

`run_eval` 会在内部强制将 `data.val_split_ratio=0.0`，按配置中的完整评估集执行。

### 3. 消融

单配置消融：

```bash
python experiments/run_ablation.py --config configs/tegb/ablation_base.yaml
```

按内置系列运行：

```bash
python experiments/run_ablation.py --suite yolo11
python experiments/run_ablation.py --suite yolo12 --weights n,s,m,l,x
python experiments/run_ablation.py --suite all --weights s --summarize --summary-root runs --summary-out-dir runs/analysis
```

`--weights` 可以直接写尺度字母：

- `n`
- `s`
- `m`
- `l`
- `x`

默认值为 `s`。

## 关键配置项

### 实验

- `experiment.exp_name`
- `experiment.output_root`
- `experiment.model_family`
- `experiment.weights`
- `experiment.target_layer`
- `experiment.num_classes`

### 数据

- `data.images_dir`
- `data.labels_dir`
- `data.image_size`
- `data.crop_size`
- `data.batch_size`
- `data.max_samples`

### 训练与损失

- `optim.epochs`
- `optim.lr`
- `optim.train_backbone`
- `loss.lambda_edl`
- `loss.lambda_gw`
- `loss.lambda_ph`
- `loss.lambda_hn`

### 粒球构建

- `granular.mode`
- `granular.radius_policy`
- `granular.radius_percentile`
- `granular.entropy_threshold`
- `granular.entropy_log_base`
- `granular.min_entropy_gain`
- `granular.max_split_depth`
- `granular.split_min_child`

### 三支决策

- `decision.mode`
- `decision.collision_gamma`

### 可视化

- `visualization.embedding_source`
- `visualization.embedding_random_subset_per_class`
- `visualization.class_groups`
- `visualization.include_classes`
- `visualization.class_compare_mode`
- `visualization.class_compare_topk`
- `visualization.datamap_enabled`
- `visualization.balls_3d_enabled`

## 输出内容

所有结果默认写入：

```text
runs/<exp_name>/
```

常见输出文件：

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

启用消融汇总后，还会生成：

- `ablation_summary_long.csv`
- `ablation_summary_wide.csv`
- `significance_tests.csv`
- `significance.json`
- `conclusion_report.json`

## 可视化说明

当前支持：

- 训练得到的 head 投影
- 高维特征经 `PCA + t-SNE / UMAP` 的导出投影
- 类别分组视图
- `datamapplot` 语义景观
- 三支 / 碰撞 / 不确定性 / 粒球 overlay

图上显示名会自动对一批 COCO 长标签做缩短，但配置解析仍然使用原始类别名。

## 推荐使用流程

1. 选择一个 `configs/tegb/` 下的配置
2. 设置数据路径和 YOLO 权重
3. 用 `run_train.py` 训练
4. 用 `run_eval.py` 导出评估与图像
5. 如需对比，运行 `run_ablation.py`
6. 使用 `runs/<exp_name>/` 下的图像与 JSON 结果进行分析

## 许可证

见 [LICENSE](LICENSE)。
