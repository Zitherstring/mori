# MORI-seg: Object-Aware RTMDet-Ins for Renal Pathology Instance Segmentation

### [[arXiv Paper]](https://arxiv.org/abs/2605.28261) [[MMDetection]](https://github.com/open-mmlab/mmdetection) [[RTMDet]](https://arxiv.org/abs/2212.07784) [[Object-aware Embedding]](https://arxiv.org/abs/2004.09821)<br />

MORI-seg is an instance segmentation model for renal pathology: RTMDet-Ins-L extended with Object-Aware auxiliary branches (distance / embedding / boundary band), implemented as an **MMDetection plugin**. <br />

![Method](icon/methodv4.png)<br />

**MORI-Seg Paper** <br />
> [MORI-Seg: Learning Morphological Geometry for Instance Segmentation without Instance Annotations](https://arxiv.org/abs/2605.28261) <br />
> Leiyue Zhao, Tianyu Shi, Daniel Reisenbuchler, Xinzi He, Junchao Zhu, Tianyuan Yao, Yuechen Yang, Yanfan Zhu, Junlin Guo, Gelei Xu, Haichun Yang, Yuankai Huo, Mert R. Sabuncu, Yihe Yang, Ruining Deng. <br />
> *arXiv:2605.28261* <br />

## Abstract

Instance segmentation on renal pathology images is difficult because objects are dense and touching: tubules form connected sheets and peritubular capillaries are small and tightly packed, so neighbouring instances are easily merged into one. MORI-seg attaches two **training-only** branches to the instance mask predictor of RTMDet-Ins:

- **Instance Disentanglement Branch** — a per-pixel embedding head supervised by a cosine metric regularization `L_disentangle`, pulling pixels of one instance together and pushing neighbouring instances apart.
- **Morphological Geometry Branch** — a distance head supervised by an exponentially reparameterized distance map (`L_dist`, balanced weighted MSE) and a boundary head supervised by the boundary band derived from the semantic mask (`L_boundary`, BCEWithLogits).

Both branches are dropped at inference, so the inference path stays identical to stock RTMDet-Ins.

## Quick Start

```bash
cd MORI-seg
export KPMP_TEST_ROOT=/path/to/test_dataset     # test set root
bash scripts/eval_core4.sh --devices cuda:0,cuda:0,cuda:0,cuda:1
```

This runs `checkpoint/Mori_seg.pth` over the test set and writes results to `work_dirs/eval_core4/Mori_seg/`.

## Installation

This repository **does not contain mmdetection itself** — it only provides the model code and configs that register into MMDetection. Please refer to [MMDetection GitHub](https://github.com/open-mmlab/mmdetection) and [get_started](https://mmdetection.readthedocs.io/en/latest/get_started.html) for full installation instructions.

```bash
conda create -n mori-seg python=3.10 -y
conda activate mori-seg

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
mim install mmdet==3.3.0

pip install pycocotools opencv-python tqdm
```

## Model

Download the pretrained weights from [Google Drive](<LINK>) and place the file at `checkpoint/Mori_seg.pth`.

## Training

```bash
# single GPU, batch 8
CUDA_VISIBLE_DEVICES=<GPU_ID> python tools/train.py configs/stage1_objaware_boundary_distexp3_100e.py

# multi-GPU
bash tools/dist_train.sh configs/stage1_objaware_boundary_distexp3_100e.py <NUM_GPUS>
```

Set `data_root` in the config to your COCO-format dataset (6 classes: `cap, dt, pt, ptc, tuft, ves`). Output goes to `work_dirs/stage1_100e/`.

## Evaluation

#### 4-class mapping evaluation

```bash
# default checkpoint, 4 shards (3 on cuda:0, 1 on cuda:1)
bash scripts/eval_core4.sh --devices cuda:0,cuda:0,cuda:0,cuda:1

# another checkpoint
bash scripts/eval_core4.sh --checkpoint path/to/epoch_100.pth --devices cuda:0

# run detached from the session
setsid nohup bash scripts/eval_core4.sh --devices cuda:0,cuda:0 > eval.log 2>&1 < /dev/null & disown
```

| Argument | Description | Default |
|---|---|---|
| `--checkpoint` | checkpoint path | `checkpoint/Mori_seg.pth` |
| `--config` | config file | `configs/stage1_objaware_boundary_distexp3_100e.py` |
| `--devices` | one shard process per entry, a GPU may repeat | `cuda:0` |
| `--out-dir` | output directory | `work_dirs/eval_core4/<ckpt name>/` |
| `--amp` / `--no-amp` | mixed-precision inference | off |
| `--contain-thres` | Mask NMS containment dedup, `1.01` disables it | `1.01` |

The 6 training classes are mapped to 4 evaluation classes (`cap → glomeruli`, `dt, pt → tubules`, `ptc → peritubular-capillaries`, `ves → arteries`, `tuft` dropped); 10x images are scored for glomeruli / tubules / arteries and 40x images for ptc. Mapping rules are in `mori_seg/eval/category_spaces.json`.

Results are written to `eval_results_<config>.json` (overall and per-class AP/AP50/AP75, semantic IoU/Dice, F1) and `per_image_metrics_<config>.csv`.

#### 6-class COCO evaluation

```bash
python tools/test.py configs/stage1_objaware_boundary_distexp3_100e.py <CHECKPOINT>
```

## Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection) / [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)
- [Object-aware Embedding (Chen et al., MICCAI 2019)](https://arxiv.org/abs/2004.09821)

## Citation

If you use this project, please cite:

```bibtex
@misc{zhao2026moriseglearningmorphologicalgeometry,
      title={MORI-Seg: Learning Morphological Geometry for Instance Segmentation without Instance Annotations}, 
      author={Leiyue Zhao and Tianyu Shi and Daniel Reisenbuchler and Xinzi He and Junchao Zhu and Tianyuan Yao and Yuechen Yang and Yanfan Zhu and Junlin Guo and Gelei Xu and Haichun Yang and Yuankai Huo and Mert R. Sabuncu and Yihe Yang and Ruining Deng},
      year={2026},
      eprint={2605.28261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.28261}, 
}
```
