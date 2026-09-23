#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NDJSON GPU evaluator and infer/transfer/finalize/eval orchestration entry point."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

try:
    import orjson as _orjson
except Exception:
    _orjson = None

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch is not available, falling back to CPU mode")

from .category_utils import (
    DEFAULT_CATEGORY_ORDER as CATEGORY_DEFAULT_ORDER,
    DEFAULT_CATEGORY_SPACE_NAME,
    DEFAULT_CATEGORY_SPACES_PATH,
    get_valid_categories_for_magnification as get_valid_categories_for_magnification_from_space,
    load_category_spaces,
    remap_coco_dataset,
    remap_predictions_to_category_space,
    resolve_category_space,
)
from .postprocess.validate_final_predictions import validate_prediction_file


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT_JSON = WORKSPACE_ROOT / "KC/annotations/test.json"
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("registry_kc_models.json")
WORK_DIRS_ENV_VAR = "MMDET_WORK_DIRS"
DEFAULT_WORK_DIRS_ROOT = WORKSPACE_ROOT / "work_dirs"
DEFAULT_CATEGORY_ORDER = list(CATEGORY_DEFAULT_ORDER)
SIZE_PATTERN_10X = re.compile(r"_2048x2048\.png$", re.IGNORECASE)
SIZE_PATTERN_40X = re.compile(r"_512x512\.png$", re.IGNORECASE)


def get_work_dirs_root():
    """Root directory of the training work dirs; override with the MMDET_WORK_DIRS env var."""
    env_value = os.environ.get(WORK_DIRS_ENV_VAR)
    return Path(env_value) if env_value else DEFAULT_WORK_DIRS_ROOT


def resolve_path_arg(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def resolve_existing_or_candidate_path(path_value, base_dirs=None):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = []
    for base_dir in base_dirs or []:
        candidates.append(Path(base_dir) / path)
    candidates.append(WORKSPACE_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def write_json_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def infer_checkpoint_from_config(config_path):
    config_stem = Path(config_path).stem
    work_dir = get_work_dirs_root() / config_stem
    if not work_dir.exists():
        return None

    patterns = [
        "best_coco_segm_mAP*.pth",
        "best*.pth",
        "latest*.pth",
        "*.pth",
    ]
    for pattern in patterns:
        checkpoints = sorted(work_dir.glob(pattern))
        if checkpoints:
            if pattern == "*.pth":
                checkpoints = sorted(checkpoints, key=lambda item: item.stat().st_mtime)
            return str(checkpoints[-1])
    return None


def extract_checkpoint_tag(checkpoint_path):
    ckpt_name = Path(checkpoint_path).name
    match = re.search(r"epoch_(\d+)", ckpt_name)
    if match:
        return match.group(1)
    if re.search(r"latest", ckpt_name, flags=re.IGNORECASE):
        return "latest"
    return Path(checkpoint_path).stem


def resolve_output_dir_from_config_and_checkpoint(config_path, checkpoint_path):
    config_stem = Path(config_path).stem
    checkpoint_tag = extract_checkpoint_tag(checkpoint_path)
    return get_work_dirs_root() / config_stem / checkpoint_tag


def infer_preferred_model_names_from_config(config_path):
    config_stem = Path(config_path).stem
    output_name = re.sub(r"_ki_split_\d+$", "", config_stem)
    preferred_names = []
    if output_name:
        preferred_names.append(output_name)
    if config_stem not in preferred_names:
        preferred_names.append(config_stem)
    return preferred_names


def infer_model_name_from_pred_path(pred_path):
    pred_path = Path(pred_path)
    stem = pred_path.stem
    if stem == "predictions" and pred_path.parent.name:
        return pred_path.parent.parent.name if pred_path.parent.name in {"infer", "transfer", "postprocess", "final", "eval"} and pred_path.parent.parent.name else pred_path.parent.name
    return re.sub(r"_merge_v2$", "", stem)


def sanitize_output_token(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "model"


def load_registry(path=DEFAULT_REGISTRY_PATH):
    path = resolve_path_arg(path) or DEFAULT_REGISTRY_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_config(registry, model_name):
    if not model_name:
        return None, {}

    models = registry.get("models", {})
    if model_name in models:
        return model_name, models[model_name]

    for canonical_name, cfg in models.items():
        if model_name in cfg.get("aliases", []):
            return canonical_name, cfg
    return model_name, {}


def print_registry_models(registry):
    print("Available models in registry:")
    for model_name, cfg in sorted(registry.get("models", {}).items()):
        aliases = cfg.get("aliases", [])
        alias_text = f" aliases={aliases}" if aliases else ""
        infer_text = " infer=yes" if (cfg.get("infer") or {}).get("script") or (cfg.get("infer") or {}).get("command") else " infer=no"
        transfer_text = " transfer=yes" if cfg.get("requires_transfer") or cfg.get("transfer") else " transfer=no"
        postprocess_text = " finalize=yes" if cfg.get("postprocess") else " finalize=no"
        print(f"- {model_name}{alias_text}{infer_text}{transfer_text}{postprocess_text}")
        if cfg.get("default_pred"):
            print(f"    source: {cfg['default_pred']}")
        if cfg.get("final_pred"):
            print(f"    final : {cfg['final_pred']}")
        if cfg.get("default_output_dir"):
            print(f"    out   : {cfg['default_output_dir']}")
        if cfg.get("category_space"):
            print(f"    space : {cfg['category_space']}")


def load_subset_image_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"subset image list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        image_ids = payload.get("image_ids", [])
    elif isinstance(payload, list):
        image_ids = payload
    else:
        raise ValueError(f"unsupported subset image list format: {path}")
    return {int(x) for x in image_ids}


def parse_subset_image_ids(raw):
    tokens = [token.strip() for token in re.split(r"[,;\s]+", raw) if token.strip()]
    return {int(token) for token in tokens}


def resolve_subset_image_ids_from_args(args):
    subset_ids = set()
    if getattr(args, "subset_image_ids", None):
        subset_ids |= parse_subset_image_ids(args.subset_image_ids)
    if getattr(args, "subset_image_list", None):
        subset_path = resolve_existing_or_candidate_path(args.subset_image_list, base_dirs=[WORKSPACE_ROOT])
        subset_ids |= load_subset_image_ids(subset_path)
    return subset_ids or None


def build_filtered_gt_skeleton(coco_gt):
    dataset = coco_gt.dataset
    return {
        "info": copy.deepcopy(dataset.get("info", {})),
        "licenses": copy.deepcopy(dataset.get("licenses", [])),
        "type": copy.deepcopy(dataset.get("type", "instances")),
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(dataset.get("categories", [])),
    }


def filter_gt_by_image_ids(coco_gt, image_ids, fix_iscrowd=True):
    image_ids = {int(x) for x in image_ids}
    gt_data = build_filtered_gt_skeleton(coco_gt)

    for img in coco_gt.dataset.get("images", []):
        if int(img.get("id", -1)) in image_ids:
            gt_data["images"].append(copy.deepcopy(img))

    valid_img_ids = {int(img["id"]) for img in gt_data["images"]}
    for ann in coco_gt.dataset.get("annotations", []):
        if int(ann.get("image_id", -1)) in valid_img_ids:
            ann_copy = copy.deepcopy(ann)
            if fix_iscrowd:
                ann_copy["iscrowd"] = 0
            gt_data["annotations"].append(ann_copy)
    return gt_data


def compute_iou_gpu_batch(masks_dt, masks_gt):
    num_dt = masks_dt.shape[0]
    num_gt = masks_gt.shape[0]
    height_dt, width_dt = masks_dt.shape[1], masks_dt.shape[2]
    height_gt, width_gt = masks_gt.shape[1], masks_gt.shape[2]

    if height_dt != height_gt or width_dt != width_gt:
        target_h = max(height_dt, height_gt)
        target_w = max(width_dt, width_gt)

        if height_dt != target_h or width_dt != target_w:
            masks_dt = masks_dt.unsqueeze(1)
            masks_dt = F.interpolate(masks_dt, size=(target_h, target_w), mode="nearest")
            masks_dt = (masks_dt.squeeze(1) > 0.5).float()

        if height_gt != target_h or width_gt != target_w:
            masks_gt = masks_gt.unsqueeze(1)
            masks_gt = F.interpolate(masks_gt, size=(target_h, target_w), mode="nearest")
            masks_gt = (masks_gt.squeeze(1) > 0.5).float()

    dt_flat = masks_dt.reshape(num_dt, -1)
    gt_flat = masks_gt.reshape(num_gt, -1)
    area_dt = dt_flat.sum(dim=1, keepdim=True)
    area_gt = gt_flat.sum(dim=1, keepdim=True)
    intersection = torch.mm(dt_flat, gt_flat.t())
    union = area_dt + area_gt.t() - intersection
    return intersection / (union + 1e-6)


def masks_to_gpu_tensor(masks_np, device):
    if not masks_np:
        return torch.empty(0, 512, 512, dtype=torch.float32, device=device)
    stacked = np.stack(masks_np, axis=0).astype(np.float32)
    return torch.from_numpy(stacked).to(device)


def compute_semantic_iou_gpu(gt_mask, pred_mask, device):
    gt_tensor = torch.from_numpy(gt_mask.astype(np.float32)).to(device)
    pred_tensor = torch.from_numpy(pred_mask.astype(np.float32)).to(device)
    intersection = torch.logical_and(gt_tensor > 0, pred_tensor > 0).sum().item()
    union = torch.logical_or(gt_tensor > 0, pred_tensor > 0).sum().item()
    gt_area = (gt_tensor > 0).sum().item()
    pred_area = (pred_tensor > 0).sum().item()
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (gt_area + pred_area) if (gt_area + pred_area) > 0 else 0.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "gt_area": int(gt_area),
        "pred_area": int(pred_area),
        "intersection": int(intersection),
        "union": int(union),
    }


def load_ndjson(path):
    arr = []
    bad_lines = 0
    with open(path, "rb") as f:
        for idx, ln in enumerate(f, 1):
            s = ln.strip()
            if not s:
                continue
            try:
                if _orjson is not None:
                    arr.append(_orjson.loads(s))
                else:
                    arr.append(json.loads(s.decode("utf-8")))
            except Exception:
                bad_lines += 1
                if bad_lines <= 3:
                    print(f"[WARN] skipping invalid JSON line: {path} (line {idx})")
                continue
    if bad_lines:
        print(f"[WARN] skipped {bad_lines} invalid JSON lines: {path}")
    return arr


def get_image_magnification_from_filename(filename):
    if filename.startswith("10x/") or SIZE_PATTERN_10X.search(filename):
        return "10x"
    if filename.startswith("40x/") or SIZE_PATTERN_40X.search(filename):
        return "40x"
    return "unknown"


def build_image_magnification_map(coco_gt):
    mag_map = {}
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs([img_id])[0]
        mag_map[img_id] = get_image_magnification_from_filename(img_info.get("file_name", ""))
    return mag_map


def filter_predictions_by_magnification(predictions, img_mag_map, coco_gt, category_space_cfg=None):
    cat_id_to_name = {int(c["id"]): c["name"] for c in coco_gt.dataset["categories"]}
    filtered = []
    for pred in predictions:
        img_id = int(pred["image_id"])
        img_mag = img_mag_map.get(img_id, "unknown")
        valid_cats = get_valid_categories_for_magnification_from_space(img_mag, category_space_cfg=category_space_cfg)
        cat_name = cat_id_to_name.get(int(pred["category_id"]))
        if cat_name in valid_cats:
            filtered.append(pred)
    return filtered


def filter_gt_by_magnification(coco_gt, img_mag_map, category_space_cfg=None, fix_iscrowd=True):
    gt_data = build_filtered_gt_skeleton(coco_gt)
    cat_id_to_name = {int(c["id"]): c["name"] for c in gt_data["categories"]}

    for img in coco_gt.dataset["images"]:
        img_id = int(img["id"])
        img_mag = img_mag_map.get(img_id, "unknown")
        valid_cats = get_valid_categories_for_magnification_from_space(img_mag, category_space_cfg=category_space_cfg)
        if not valid_cats:
            continue

        gt_data["images"].append(copy.deepcopy(img))
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            cat_name = cat_id_to_name.get(int(ann["category_id"]))
            if cat_name in valid_cats:
                ann_copy = copy.deepcopy(ann)
                if fix_iscrowd:
                    ann_copy["iscrowd"] = 0
                gt_data["annotations"].append(ann_copy)
    return gt_data


def run_official_coco_eval(coco_gt, predictions, iou_type="segm", img_mag_map=None, category_space_cfg=None):
    if img_mag_map is not None:
        gt_data_filtered = filter_gt_by_magnification(coco_gt, img_mag_map, category_space_cfg=category_space_cfg, fix_iscrowd=True)
        predictions_eval = filter_predictions_by_magnification(predictions, img_mag_map, coco_gt, category_space_cfg=category_space_cfg)
        print(f"    GT annotations after filtering: {len(gt_data_filtered['annotations'])}")
        print(f"    Predictions after filtering: {len(predictions_eval)}")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_data_filtered, f)
            gt_file = f.name
        coco_gt_eval = COCO(gt_file)
        os.unlink(gt_file)
    else:
        coco_gt_eval = coco_gt
        predictions_eval = predictions

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions_eval, f)
        pred_file = f.name

    try:
        if len(predictions_eval) == 0:
            print("    [WARN] no predictions left after filtering")
            return None, None

        coco_dt = coco_gt_eval.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt_eval, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        cat_ids = coco_gt_eval.getCatIds()
        ap_per_class = {}
        for cat_id in cat_ids:
            cat_name = coco_gt_eval.loadCats([cat_id])[0]["name"]
            gt_anns_cat = len(coco_gt_eval.getAnnIds(catIds=[cat_id]))
            pred_anns_cat = sum(1 for pred in predictions_eval if int(pred["category_id"]) == int(cat_id))

            if gt_anns_cat == 0:
                ap_per_class[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "gt_count": 0, "pred_count": pred_anns_cat}
                continue
            if pred_anns_cat == 0:
                ap_per_class[cat_name] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "gt_count": gt_anns_cat, "pred_count": 0}
                continue

            coco_eval_cat = COCOeval(coco_gt_eval, coco_dt, iou_type)
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            coco_eval_cat.summarize()
            stats = coco_eval_cat.stats
            ap_per_class[cat_name] = {
                "AP": float(stats[0]) if stats[0] >= 0 else 0.0,
                "AP50": float(stats[1]) if stats[1] >= 0 else 0.0,
                "AP75": float(stats[2]) if stats[2] >= 0 else 0.0,
                "gt_count": gt_anns_cat,
                "pred_count": pred_anns_cat,
            }

        overall = {
            "mAP": float(coco_eval.stats[0]) if coco_eval.stats[0] >= 0 else 0.0,
            "AP50": float(coco_eval.stats[1]) if coco_eval.stats[1] >= 0 else 0.0,
            "AP75": float(coco_eval.stats[2]) if coco_eval.stats[2] >= 0 else 0.0,
        }
        return overall, ap_per_class
    finally:
        os.unlink(pred_file)


def calculate_semantic_iou(coco_gt, predictions, categories, img_mag_map=None, category_space_cfg=None, device=None):
    use_gpu = device is not None and TORCH_AVAILABLE and torch.cuda.is_available()
    if use_gpu:
        print(f"    [GPU] computing semantic IoU on {device}")

    iou_by_category = {}
    per_image_iou = defaultdict(dict)
    predictions_by_image_and_cat = defaultdict(list)
    for pred in predictions:
        predictions_by_image_and_cat[(int(pred["image_id"]), int(pred["category_id"]))].append(pred)

    for cat in categories:
        cat_id = int(cat["id"])
        cat_name = cat["name"]
        total_intersection = 0
        total_union = 0
        total_gt_area = 0
        total_pred_area = 0
        iou_list = []
        valid_img_count = 0

        for img_id in tqdm(coco_gt.getImgIds(), desc=f"IoU-{cat_name[:10]}", leave=False):
            if img_mag_map is not None:
                img_mag = img_mag_map.get(img_id, "unknown")
                valid_cats = get_valid_categories_for_magnification_from_space(img_mag, category_space_cfg=category_space_cfg)
                if cat_name not in valid_cats:
                    continue

            valid_img_count += 1
            img_info = coco_gt.loadImgs([img_id])[0]
            height, width = img_info["height"], img_info["width"]

            gt_mask = np.zeros((height, width), dtype=np.uint8)
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
            anns = coco_gt.loadAnns(ann_ids)
            for ann in anns:
                if "segmentation" in ann:
                    gt_mask = np.maximum(gt_mask, coco_gt.annToMask(ann))

            pred_mask = np.zeros((height, width), dtype=np.uint8)
            img_preds = predictions_by_image_and_cat.get((int(img_id), cat_id), [])
            for pred in img_preds:
                if "segmentation" not in pred:
                    continue
                if isinstance(pred["segmentation"], dict):
                    mask = maskUtils.decode(pred["segmentation"])
                else:
                    rle = maskUtils.frPyObjects(pred["segmentation"], height, width)
                    mask = maskUtils.decode(rle)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                pred_mask = np.maximum(pred_mask, mask)

            if use_gpu:
                result = compute_semantic_iou_gpu(gt_mask, pred_mask, device)
                intersection = result["intersection"]
                union = result["union"]
                gt_area = result["gt_area"]
                pred_area = result["pred_area"]
                iou = result["iou"]
                dice = result["dice"]
            else:
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                gt_area = gt_mask.sum()
                pred_area = pred_mask.sum()
                iou = intersection / union if union > 0 else 0.0
                dice = (2 * intersection) / (gt_area + pred_area) if (gt_area + pred_area) > 0 else 0.0

            if union > 0:
                iou_list.append(iou)

            per_image_iou[img_id][cat_name] = {
                "iou": float(iou),
                "dice": float(dice),
                "gt_area": int(gt_area),
                "pred_area": int(pred_area),
                "intersection": int(intersection),
                "union": int(union),
            }
            total_intersection += intersection
            total_union += union
            total_gt_area += gt_area
            total_pred_area += pred_area

        overall_iou = total_intersection / total_union if total_union > 0 else 0.0
        mean_iou = np.mean(iou_list) if iou_list else 0.0
        dice = (2 * total_intersection) / (total_gt_area + total_pred_area) if (total_gt_area + total_pred_area) > 0 else 0.0
        iou_by_category[cat_name] = {
            "overall_iou": float(overall_iou),
            "mean_iou_per_image": float(mean_iou),
            "dice": float(dice),
            "total_gt_area": int(total_gt_area),
            "total_pred_area": int(total_pred_area),
            "total_intersection": int(total_intersection),
            "num_images_with_iou": len(iou_list),
            "valid_images": valid_img_count,
        }

    if use_gpu:
        torch.cuda.empty_cache()
    return iou_by_category, dict(per_image_iou)


def calculate_f1_by_category(coco_gt, predictions, categories, iou_threshold=0.5, img_mag_map=None, category_space_cfg=None, device=None):
    use_gpu = device is not None and TORCH_AVAILABLE and torch.cuda.is_available()
    if use_gpu:
        print(f"    [GPU] computing F1 on {device}")

    f1_by_category = {}
    per_image_f1 = defaultdict(dict)
    predictions_by_image_and_cat = defaultdict(list)
    for pred in predictions:
        predictions_by_image_and_cat[(int(pred["image_id"]), int(pred["category_id"]))].append(pred)

    for cat in categories:
        cat_id = int(cat["id"])
        cat_name = cat["name"]
        tp = 0
        fp = 0
        fn = 0
        valid_img_count = 0

        for img_id in tqdm(coco_gt.getImgIds(), desc=f"F1-{cat_name[:10]}", leave=False):
            if img_mag_map is not None:
                img_mag = img_mag_map.get(img_id, "unknown")
                valid_cats = get_valid_categories_for_magnification_from_space(img_mag, category_space_cfg=category_space_cfg)
                if cat_name not in valid_cats:
                    continue

            valid_img_count += 1
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
            gt_anns = coco_gt.loadAnns(ann_ids)
            img_preds = predictions_by_image_and_cat.get((int(img_id), cat_id), [])
            img_tp = 0

            if len(gt_anns) == 0 and len(img_preds) == 0:
                per_image_f1[img_id][cat_name] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "gt_count": 0,
                    "pred_count": 0,
                }
                continue

            if len(gt_anns) == 0:
                img_fp = len(img_preds)
                fp += img_fp
                per_image_f1[img_id][cat_name] = {
                    "tp": 0,
                    "fp": img_fp,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 1.0,
                    "f1": 0.0,
                    "gt_count": 0,
                    "pred_count": len(img_preds),
                }
                continue

            if len(img_preds) == 0:
                img_fn = len(gt_anns)
                fn += img_fn
                per_image_f1[img_id][cat_name] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": img_fn,
                    "precision": 1.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "gt_count": len(gt_anns),
                    "pred_count": 0,
                }
                continue

            img_info = coco_gt.loadImgs([img_id])[0]
            height, width = img_info["height"], img_info["width"]
            gt_masks = [coco_gt.annToMask(gt_ann) for gt_ann in gt_anns]

            pred_masks = []
            for pred in img_preds:
                if "segmentation" not in pred:
                    continue
                if isinstance(pred["segmentation"], dict):
                    pred_mask = maskUtils.decode(pred["segmentation"])
                else:
                    rle = maskUtils.frPyObjects(pred["segmentation"], height, width)
                    pred_mask = maskUtils.decode(rle)
                    if len(pred_mask.shape) == 3:
                        pred_mask = pred_mask[:, :, 0]
                pred_masks.append(pred_mask)

            if use_gpu and gt_masks and pred_masks:
                gt_tensor = masks_to_gpu_tensor(gt_masks, device)
                pred_tensor = masks_to_gpu_tensor(pred_masks, device)
                with torch.no_grad():
                    iou_mat = compute_iou_gpu_batch(pred_tensor, gt_tensor)
                    ious = iou_mat.t().cpu().numpy()
                del gt_tensor, pred_tensor
            else:
                ious = np.zeros((len(gt_anns), len(pred_masks)))
                for i, gt_mask in enumerate(gt_masks):
                    for j, pred_mask in enumerate(pred_masks):
                        intersection = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        ious[i, j] = intersection / union if union > 0 else 0.0

            matched_gt = set()
            matched_pred = set()
            iou_pairs = []
            for i in range(len(gt_anns)):
                for j in range(len(pred_masks)):
                    if ious[i, j] >= iou_threshold:
                        iou_pairs.append((ious[i, j], i, j))
            iou_pairs.sort(reverse=True)

            for _, i, j in iou_pairs:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    img_tp += 1

            img_fp = len(pred_masks) - len(matched_pred)
            img_fn = len(gt_anns) - len(matched_gt)
            tp += img_tp
            fp += img_fp
            fn += img_fn

            img_precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
            img_recall = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0.0
            img_f1 = 2 * (img_precision * img_recall) / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0.0
            per_image_f1[img_id][cat_name] = {
                "tp": img_tp,
                "fp": img_fp,
                "fn": img_fn,
                "precision": float(img_precision),
                "recall": float(img_recall),
                "f1": float(img_f1),
                "gt_count": len(gt_anns),
                "pred_count": len(pred_masks),
            }

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_by_category[cat_name] = {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "valid_images": valid_img_count,
        }

    if use_gpu:
        torch.cuda.empty_cache()
    return f1_by_category, dict(per_image_f1)


def collect_per_image_metrics(coco_gt, per_image_iou, per_image_f1, img_mag_map, categories):
    cat_names = [c["name"] for c in categories]
    results = []
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs([img_id])[0]
        row = {
            "image_id": img_id,
            "image_path": img_info.get("file_name", ""),
            "magnification": img_mag_map.get(img_id, "unknown"),
        }
        for cat_name in cat_names:
            iou_data = per_image_iou.get(img_id, {}).get(cat_name, {})
            row[f"{cat_name}_iou"] = iou_data.get("iou", "")
            row[f"{cat_name}_dice"] = iou_data.get("dice", "")
            row[f"{cat_name}_gt_area"] = iou_data.get("gt_area", "")
            row[f"{cat_name}_pred_area"] = iou_data.get("pred_area", "")

            f1_data = per_image_f1.get(img_id, {}).get(cat_name, {})
            row[f"{cat_name}_tp"] = f1_data.get("tp", "")
            row[f"{cat_name}_fp"] = f1_data.get("fp", "")
            row[f"{cat_name}_fn"] = f1_data.get("fn", "")
            row[f"{cat_name}_precision"] = f1_data.get("precision", "")
            row[f"{cat_name}_recall"] = f1_data.get("recall", "")
            row[f"{cat_name}_f1"] = f1_data.get("f1", "")
            row[f"{cat_name}_gt_count"] = f1_data.get("gt_count", "")
            row[f"{cat_name}_pred_count"] = f1_data.get("pred_count", "")
        results.append(row)
    return results


def save_per_image_results_to_csv(results, output_path, categories):
    if not results:
        print("    [WARN] no results to save")
        return

    cat_names = [c["name"] for c in categories]
    fieldnames = ["image_id", "image_path", "magnification"]
    for cat_name in cat_names:
        fieldnames.extend(
            [
                f"{cat_name}_iou",
                f"{cat_name}_dice",
                f"{cat_name}_gt_area",
                f"{cat_name}_pred_area",
                f"{cat_name}_tp",
                f"{cat_name}_fp",
                f"{cat_name}_fn",
                f"{cat_name}_precision",
                f"{cat_name}_recall",
                f"{cat_name}_f1",
                f"{cat_name}_gt_count",
                f"{cat_name}_pred_count",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"    Per-image metrics saved: {output_path}")


def resolve_device(device_arg, no_gpu):
    if no_gpu or not TORCH_AVAILABLE:
        print("[INFO] using CPU mode")
        return None
    if torch.cuda.is_available():
        print(f"[INFO] GPU acceleration enabled: {device_arg}")
        try:
            device_index = int(str(device_arg).split(":")[-1]) if ":" in str(device_arg) else 0
        except ValueError:
            device_index = 0
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device_index)}")
        return device_arg
    print("[WARN] CUDA is not available, using CPU mode")
    return None


def capture_coco_eval(coco_gt, predictions, img_mag_map, category_space_cfg=None):
    buffer = io.StringIO()
    segm_overall = None
    segm_per_class = None
    try:
        with redirect_stdout(buffer):
            segm_overall, segm_per_class = run_official_coco_eval(
                coco_gt,
                predictions,
                "segm",
                img_mag_map,
                category_space_cfg=category_space_cfg,
            )
    finally:
        coco_stdout = buffer.getvalue()
    return segm_overall, segm_per_class, coco_stdout


def evaluate_model(
    model_name,
    pred_path,
    out_dir,
    gt_json=DEFAULT_GT_JSON,
    device=None,
    subset_image_ids=None,
    registry_model_name=None,
    registry_entry=None,
    category_space_name=None,
    category_space_cfg=None,
    eval_dir=None,
    final_pred_path=None,
):
    print("=" * 80)
    print(f"Evaluating: {model_name}")
    if registry_model_name and registry_model_name != model_name:
        print(f"Registry model name: {registry_model_name}")
    if device:
        print(f"GPU acceleration: {device}")
    if category_space_name:
        print(f"Category space: {category_space_name}")
    print("=" * 80)

    gt_json = Path(gt_json)
    pred_path = Path(pred_path)
    out_dir = Path(out_dir)
    if not gt_json.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_json}")
    if not pred_path.exists():
        print(f"[ERROR] prediction file not found: {pred_path}")
        if registry_entry and (registry_entry.get("requires_transfer") or registry_entry.get("transfer")):
            print("[HINT] this model supports transfer; add --run-transfer to generate the NDJSON")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = Path(eval_dir or (out_dir / "eval"))
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1] Loading data...")
    predictions = load_ndjson(pred_path)
    coco_gt_all = COCO(str(gt_json))
    temp_subset_gt = None
    temp_category_gt = None

    try:
        if subset_image_ids:
            subset_image_ids = {int(x) for x in subset_image_ids}
            predictions = [pred for pred in predictions if int(pred.get("image_id", -1)) in subset_image_ids]
            gt_subset = filter_gt_by_image_ids(coco_gt_all, subset_image_ids, fix_iscrowd=False)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(gt_subset, f)
                temp_subset_gt = f.name
            coco_gt_source = COCO(temp_subset_gt)
            gt_data_source = coco_gt_source.dataset
            print(f"    Subset GT images: {len(gt_data_source['images'])}")
            print(f"    Subset GT annotations: {len(gt_data_source['annotations'])}")
            print(f"    Subset predictions: {len(predictions)}")
        else:
            coco_gt_source = coco_gt_all
            gt_data_source = coco_gt_source.dataset
            print(f"    GT images: {len(gt_data_source['images'])}")
            print(f"    GT annotations: {len(gt_data_source['annotations'])}")
            print(f"    Predictions: {len(predictions)}")

        if category_space_cfg is not None:
            predictions = remap_predictions_to_category_space(predictions, category_space_cfg, source_categories=gt_data_source.get("categories", []))
            gt_data = remap_coco_dataset(gt_data_source, category_space_cfg, fix_iscrowd=False)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(gt_data, f)
                temp_category_gt = f.name
            coco_gt = COCO(temp_category_gt)
            print(f"    Categories after remap: {len(gt_data['categories'])}")
            print(f"    GT annotations after remap: {len(gt_data['annotations'])}")
            print(f"    Predictions after remap: {len(predictions)}")
        else:
            coco_gt = coco_gt_source
            gt_data = coco_gt.dataset

        print("\n[2] Building image magnification map...")
        img_mag_map = build_image_magnification_map(coco_gt)
        count_10x = sum(1 for mag in img_mag_map.values() if mag == "10x")
        count_40x = sum(1 for mag in img_mag_map.values() if mag == "40x")
        print(f"    10x images: {count_10x}")
        print(f"    40x images: {count_40x}")

        print("\n[3] Running COCOeval...")
        try:
            segm_overall, segm_per_class, coco_stdout = capture_coco_eval(coco_gt, predictions, img_mag_map, category_space_cfg=category_space_cfg)
            if coco_stdout:
                print(coco_stdout, end="" if coco_stdout.endswith("\n") else "\n")
                coco_stdout_path = eval_output_dir / "coco_stdout.txt"
                coco_stdout_path.write_text(coco_stdout, encoding="utf-8")
                legacy_coco_stdout_path = out_dir / "coco_stdout.txt"
                if legacy_coco_stdout_path != coco_stdout_path:
                    legacy_coco_stdout_path.write_text(coco_stdout, encoding="utf-8")
            if segm_overall:
                print(f"\nOverall mAP@[.50:.95]: {segm_overall['mAP']:.4f}")
                print(f"    Overall AP50: {segm_overall['AP50']:.4f}")
                print(f"    Overall AP75: {segm_overall['AP75']:.4f}")
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            segm_overall = None
            segm_per_class = None

        print("\n[4] Computing semantic IoU...")
        semantic_iou_results, per_image_iou = calculate_semantic_iou(
            coco_gt,
            predictions,
            gt_data["categories"],
            img_mag_map,
            category_space_cfg=category_space_cfg,
            device=device,
        )
        print("\nSemantic IoU:")
        for cat_name, result in semantic_iou_results.items():
            print(f"      {cat_name}: IoU={result['overall_iou']:.4f}, Dice={result['dice']:.4f}")

        print("\n[5] Computing instance-level F1...")
        f1_results, per_image_f1 = calculate_f1_by_category(
            coco_gt,
            predictions,
            gt_data["categories"],
            iou_threshold=0.5,
            img_mag_map=img_mag_map,
            category_space_cfg=category_space_cfg,
            device=device,
        )
        print("\nF1 scores:")
        for cat_name, result in f1_results.items():
            print(f"      {cat_name}: F1={result['f1']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}")

        print("\n[6] Saving detailed metrics...")
        per_image_results = collect_per_image_metrics(coco_gt, per_image_iou, per_image_f1, img_mag_map, gt_data["categories"])
        csv_output_path = eval_output_dir / "per_image_metrics.csv"
        legacy_csv_output_path_root = out_dir / "per_image_metrics.csv"
        legacy_csv_output_path = out_dir / f"per_image_metrics_{model_name}.csv"
        save_per_image_results_to_csv(per_image_results, csv_output_path, gt_data["categories"])
        if legacy_csv_output_path_root != csv_output_path:
            shutil.copyfile(csv_output_path, legacy_csv_output_path_root)
        if legacy_csv_output_path != csv_output_path:
            shutil.copyfile(csv_output_path, legacy_csv_output_path)

        output = {
            "model": model_name,
            "registry_model_name": registry_model_name,
            "pred_file": str(pred_path),
            "final_pred_file": str(final_pred_path or pred_path),
            "output_dir": str(out_dir),
            "eval_output_dir": str(eval_output_dir),
            "gt_json": str(gt_json),
            "gpu_accelerated": device is not None,
            "subset_eval": bool(subset_image_ids),
            "subset_image_count": len(coco_gt.getImgIds()),
            "requires_transfer": bool(registry_entry and (registry_entry.get("requires_transfer") or registry_entry.get("transfer"))),
            "category_space": category_space_name,
            "category_space_order": [cat["name"] for cat in gt_data.get("categories", [])],
            "magnification_stats": {"10x_images": count_10x, "40x_images": count_40x},
            "segm": {"overall": segm_overall, "per_class": segm_per_class} if segm_overall is not None else None,
            "semantic_iou": semantic_iou_results,
            "f1": f1_results,
        }
        result_json_path = eval_output_dir / "eval_results.json"
        legacy_result_json_path_root = out_dir / "eval_results.json"
        legacy_result_json_path = out_dir / f"eval_results_{model_name}.json"
        write_json_file(result_json_path, output)
        if legacy_result_json_path_root != result_json_path:
            write_json_file(legacy_result_json_path_root, output)
        if legacy_result_json_path != result_json_path:
            write_json_file(legacy_result_json_path, output)
        print(f"\n[OK] results saved: {result_json_path}")
        return output
    finally:
        if temp_subset_gt is not None and os.path.exists(temp_subset_gt):
            os.unlink(temp_subset_gt)
        if temp_category_gt is not None and os.path.exists(temp_category_gt):
            os.unlink(temp_category_gt)


def print_results_table(result):
    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    if result is None or result.get("segm") is None:
        print("[WARN] no evaluation results")
        return

    model_name = result.get("model", "Unknown")
    segm = result["segm"]["overall"]
    category_space_name = result.get("category_space") or DEFAULT_CATEGORY_SPACE_NAME
    category_order = result.get("category_space_order") or DEFAULT_CATEGORY_ORDER
    print(f"\nModel: {model_name}")
    print(f"Category space: {category_space_name}")
    print(f"mAP@[.50:.95]: {segm['mAP']:.4f}")
    print(f"AP50: {segm['AP50']:.4f}")
    print(f"AP75: {segm['AP75']:.4f}")

    print("\nPer-category AP@[.50:.95]:")
    print("-" * 60)
    for cat in category_order:
        ap = result["segm"]["per_class"].get(cat, {}).get("AP", 0)
        print(f"  {cat}: {ap:.4f}")

    print("\nPer-category F1@0.5:")
    print("-" * 60)
    for cat in category_order:
        f1 = result["f1"].get(cat, {}).get("f1", 0)
        precision = result["f1"].get(cat, {}).get("precision", 0)
        recall = result["f1"].get(cat, {}).get("recall", 0)
        print(f"  {cat}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    print("\nPer-category semantic IoU:")
    print("-" * 60)
    for cat in category_order:
        iou = result["semantic_iou"].get(cat, {}).get("overall_iou", 0)
        dice = result["semantic_iou"].get(cat, {}).get("dice", 0)
        print(f"  {cat}: IoU={iou:.4f}, Dice={dice:.4f}")


def build_stage_dirs(output_root):
    output_root = Path(output_root)
    stage_dirs = {
        "root": output_root,
        "infer": output_root / "infer",
        "transfer": output_root / "transfer",
        "postprocess": output_root / "postprocess",
        "final": output_root / "final",
        "eval": output_root / "eval",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for key, path in stage_dirs.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return stage_dirs


def flatten_context_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def build_runtime_context(
    *,
    model_name,
    output_root,
    stage_dirs,
    config_path=None,
    checkpoint_path=None,
    device=None,
    image_root=None,
    gt_json=None,
    extra=None,
):
    output_root = Path(output_root)
    context = {
        "model_name": model_name,
        "output_root": output_root,
        "root": output_root,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "config_stem": Path(config_path).stem if config_path is not None else None,
        "checkpoint_tag": extract_checkpoint_tag(checkpoint_path) if checkpoint_path is not None else None,
        "device": device,
        "image_root": image_root,
        "gt_json": gt_json,
    }
    for key, value in stage_dirs.items():
        context[f"{key}_dir"] = value
    if extra:
        context.update(extra)
    return context


def render_template_string(template, context):
    formatter = string.Formatter()
    field_names = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
    missing = [field_name for field_name in field_names if context.get(field_name) in (None, "")]
    if missing:
        raise ValueError(f"command template is missing context values: {template} -> {missing}")
    safe_context = {key: flatten_context_value(value) for key, value in context.items()}
    return template.format(**safe_context)


def render_command_items(items, context):
    rendered = []
    if items is None:
        return rendered
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            rendered.extend(render_command_items(item, context))
            continue
        item_str = str(item)
        placeholder_match = re.fullmatch(r"\{([A-Za-z0-9_]+)\}", item_str)
        if placeholder_match is not None:
            placeholder_name = placeholder_match.group(1)
            placeholder_value = context.get(placeholder_name, None)
            if placeholder_value in (None, ""):
                continue
        item_text = render_template_string(item_str, context)
        if item_text != "":
            rendered.append(item_text)
    return rendered


def find_prediction_from_output_dir(output_dir, config_path=None):
    output_dir = Path(output_dir)
    preferred_names = infer_preferred_model_names_from_config(config_path) if config_path else []
    search_dirs = [output_dir / "final", output_dir / "postprocess", output_dir / "transfer", output_dir / "infer", output_dir]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for candidate_name in preferred_names:
            candidate_pred = search_dir / f"{candidate_name}_merge_v2.ndjson"
            if candidate_pred.exists():
                return candidate_pred, candidate_name
        candidate = search_dir / "predictions.ndjson"
        if candidate.exists():
            return candidate, infer_model_name_from_pred_path(candidate)

    ndjson_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            ndjson_files.extend(sorted(search_dir.glob("*.ndjson")))
    if ndjson_files:
        pred_path = ndjson_files[-1]
        return pred_path, infer_model_name_from_pred_path(pred_path)
    return None, None


def run_infer_for_model(model_name, model_cfg, *, stage_dirs, config_path=None, checkpoint_path=None, device=None, image_root=None, gt_json=None, infer_options=None):
    infer_cfg = model_cfg.get("infer") or {}
    script_path = resolve_path_arg(infer_cfg.get("script")) if infer_cfg.get("script") else None
    if script_path is None and not infer_cfg.get("command"):
        raise ValueError(f"model {model_name} has no infer script/command configured")
    if script_path is not None and not script_path.exists():
        raise FileNotFoundError(f"infer script not found: {script_path}")

    infer_options = infer_options or argparse.Namespace()
    raw_output_dir = resolve_path_arg(infer_cfg.get("raw_output_dir")) or stage_dirs["infer"]
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config_path or resolve_path_arg(infer_cfg.get("config"))
    resolved_checkpoint = checkpoint_path or resolve_path_arg(infer_cfg.get("checkpoint"))
    infer_pred_root = resolve_path_arg(infer_cfg.get("pred_root")) or raw_output_dir
    infer_out_pred = resolve_path_arg(infer_cfg.get("out_pred"))
    resolved_image_root = resolve_path_arg(image_root) if image_root is not None else None
    if device is not None:
        device_for_infer = str(device)
    elif bool(getattr(infer_options, "no_gpu", False)):
        device_for_infer = "cpu"
    else:
        device_for_infer = str(getattr(infer_options, "device", None) or "cuda:0")
    export_objaware_map = bool(getattr(infer_options, "export_objaware_map", False))
    objaware_map_dir = resolve_path_arg(getattr(infer_options, "objaware_map_dir", None))
    if objaware_map_dir is None:
        objaware_map_dir = raw_output_dir / "objaware_maps"

    context = build_runtime_context(
        model_name=model_name,
        output_root=stage_dirs["root"],
        stage_dirs=stage_dirs,
        config_path=resolved_config,
        checkpoint_path=resolved_checkpoint,
        device=device_for_infer,
        image_root=resolved_image_root,
        gt_json=gt_json,
        extra={
            "raw_output_dir": raw_output_dir,
            "pred_root": infer_pred_root,
            "out_pred": infer_out_pred,
            "img_root_flag": "--img-root" if resolved_image_root is not None else "",
            "image_root": resolved_image_root if resolved_image_root is not None else "",
            "workers": int(getattr(infer_options, "workers", 4)),
            "preload_mode": str(getattr(infer_options, "preload_mode", "thread")),
            "batch_size": int(getattr(infer_options, "batch_size", 1)),
            "amp_flag": "--amp" if bool(getattr(infer_options, "amp", False)) else "",
            "log_every": int(getattr(infer_options, "log_every", 200)),
            "num_shards": int(getattr(infer_options, "num_shards", 1)),
            "shard_id": int(getattr(infer_options, "shard_id", 0)),
            "emit_source_category_space_flag": "--emit-source-category-space" if bool(infer_cfg.get("emit_source_category_space")) else "",
            "export_objaware_map_flag": "--export-objaware-map" if export_objaware_map else "",
            "objaware_map_dir_flag": "--objaware-map-dir" if export_objaware_map or getattr(infer_options, "objaware_map_dir", None) else "",
            "objaware_map_dir": objaware_map_dir if export_objaware_map or getattr(infer_options, "objaware_map_dir", None) else "",
            "isdf_refine_flag": "--isdf-refine" if bool(getattr(infer_options, "isdf_refine", False)) else "",
            "isdf_sigma": float(getattr(infer_options, "isdf_sigma", 1.0)),
            "isdf_h": float(getattr(infer_options, "isdf_h", 2.0)),
            "isdf_min_size": int(getattr(infer_options, "isdf_min_size", 10)),
            "isdf_downscale": float(getattr(infer_options, "isdf_downscale", 1.0)),
            "isdf_topk": int(getattr(infer_options, "isdf_topk", 0)),
            "isdf_min_area": int(getattr(infer_options, "isdf_min_area", 0)),
        },
    )

    if infer_cfg.get("command"):
        cmd = render_command_items(infer_cfg.get("command"), context)
    else:
        cmd = [sys.executable, str(script_path)]
        infer_args = infer_cfg.get("args") or []
        if infer_args:
            cmd.extend(render_command_items(infer_args, context))
        else:
            if resolved_config is not None:
                cmd.extend(["--config", str(resolved_config)])
            if resolved_checkpoint is not None:
                cmd.extend(["--checkpoint", str(resolved_checkpoint)])
            cmd.extend(["--output-dir", str(raw_output_dir)])
            if device_for_infer:
                cmd.extend(["--device", str(device_for_infer)])
            if resolved_image_root is not None:
                cmd.extend([str(infer_cfg.get("image_root_arg") or "--image-root"), str(resolved_image_root)])

    print(f"[INFO] running infer ({model_name})...")
    print("[INFO] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {
        "raw_output_dir": raw_output_dir,
        "pred_root": infer_pred_root,
        "out_pred": infer_out_pred,
    }


def run_transfer_for_model(model_name, model_cfg, *, stage_dirs, category_space_name=None, category_spaces_path=None, transfer_pred_root=None, transfer_gt_json=None, infer_runtime=None):
    transfer_cfg = model_cfg.get("transfer") or {}
    script_path = resolve_path_arg(transfer_cfg.get("script"))
    if script_path is None or not script_path.exists():
        raise FileNotFoundError(f"transfer script not found: {script_path}")

    pred_root_value = transfer_pred_root or transfer_cfg.get("pred_root") or (infer_runtime or {}).get("pred_root") or (infer_runtime or {}).get("raw_output_dir")
    if not pred_root_value:
        raise ValueError(f"model {model_name} has no transfer pred_root configured")
    pred_root = resolve_path_arg(pred_root_value)
    if not pred_root.exists():
        raise FileNotFoundError(f"transfer input directory not found: {pred_root}")

    transfer_gt_value = transfer_gt_json or transfer_cfg.get("gt_json")
    transfer_gt = resolve_path_arg(transfer_gt_value)
    if transfer_gt is None or not transfer_gt.exists():
        raise FileNotFoundError(f"transfer GT JSON not found: {transfer_gt}")

    out_path = resolve_path_arg(transfer_cfg.get("out_path")) or (stage_dirs["transfer"] / "predictions.ndjson")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--gt-json",
        str(transfer_gt),
        "--pred-root",
        str(pred_root),
        "--out-path",
        str(out_path),
    ]
    if transfer_cfg.get("conf_min") is not None:
        cmd.extend(["--conf-min", str(transfer_cfg["conf_min"])])
    if transfer_cfg.get("topk_per_image") is not None:
        cmd.extend(["--topk-per-image", str(transfer_cfg["topk_per_image"])])
    if transfer_cfg.get("procs") is not None:
        cmd.extend(["--procs", str(transfer_cfg["procs"])])
    if transfer_cfg.get("chunk_size") is not None:
        cmd.extend(["--chunk-size", str(transfer_cfg["chunk_size"])])
    if category_space_name or transfer_cfg.get("category_space"):
        cmd.extend(["--category-space", str(category_space_name or transfer_cfg.get("category_space"))])
    if category_spaces_path is not None:
        cmd.extend(["--category-spaces-json", str(category_spaces_path)])
    if transfer_cfg.get("source_space") is not None:
        cmd.extend(["--source-space", str(transfer_cfg.get("source_space"))])
    if transfer_cfg.get("source_class_names") is not None:
        source_class_names = transfer_cfg.get("source_class_names")
        if isinstance(source_class_names, list):
            source_class_names = ",".join(source_class_names)
        cmd.extend(["--source-class-names", str(source_class_names)])

    print(f"[INFO] running transfer ({model_name})...")
    print("[INFO] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_postprocess_for_model(
    model_name,
    model_cfg,
    *,
    input_pred,
    gt_json,
    stage_dirs,
    category_space_name,
    category_spaces_path,
    final_pred_path=None,
    postprocess_type=None,
    source_space_override=None,
    source_class_names_override=None,
):
    post_cfg = model_cfg.get("postprocess") or {}
    effective_type = postprocess_type or post_cfg.get("type") or "finalize_ndjson"
    effective_source_space = source_space_override if source_space_override is not None else post_cfg.get("source_space")
    effective_source_class_names = source_class_names_override if source_class_names_override is not None else post_cfg.get("source_class_names")
    stage_output_pred = stage_dirs["postprocess"] / "predictions.ndjson"
    output_pred = Path(final_pred_path or resolve_path_arg(post_cfg.get("output_pred")) or (stage_dirs["final"] / "predictions.ndjson"))
    stage_output_pred.parent.mkdir(parents=True, exist_ok=True)
    output_pred.parent.mkdir(parents=True, exist_ok=True)
    input_pred = Path(input_pred)

    if not input_pred.exists():
        raise FileNotFoundError(f"postprocess input predictions not found: {input_pred}")

    if effective_type in {"none", "identity"}:
        if input_pred.resolve() != stage_output_pred.resolve():
            shutil.copyfile(input_pred, stage_output_pred)
    else:
        script_path = resolve_path_arg(post_cfg.get("script")) or (Path(__file__).with_name("postprocess") / "remap_predictions_to_test_space.py")
        if script_path is None or not script_path.exists():
            raise FileNotFoundError(f"postprocess script not found: {script_path}")

        context = build_runtime_context(
            model_name=model_name,
            output_root=stage_dirs["root"],
            stage_dirs=stage_dirs,
            gt_json=gt_json,
            extra={
                "input_pred": input_pred,
                "output_pred": output_pred,
                "postprocess_output": stage_output_pred,
                "gt_json": gt_json,
                "category_space": category_space_name,
                "category_spaces_json": category_spaces_path,
                "postprocess_type": effective_type,
            },
        )

        if post_cfg.get("command"):
            cmd = render_command_items(post_cfg.get("command"), context)
        else:
            cmd = [
                sys.executable,
                str(script_path),
                "--pred",
                str(input_pred),
                "--out-pred",
                str(stage_output_pred),
                "--gt-json",
                str(gt_json),
                "--category-space",
                str(category_space_name),
                "--category-spaces-json",
                str(category_spaces_path),
            ]
            if effective_source_space is not None:
                cmd.extend(["--source-space", str(effective_source_space)])
            if effective_source_class_names is not None:
                source_class_names = effective_source_class_names
                if isinstance(source_class_names, list):
                    source_class_names = ",".join(source_class_names)
                cmd.extend(["--source-class-names", str(source_class_names)])
            if post_cfg.get("topk_per_image") is not None:
                cmd.extend(["--topk-per-image", str(post_cfg.get("topk_per_image"))])
            if post_cfg.get("args"):
                cmd.extend(render_command_items(post_cfg.get("args"), context))

        print(f"[INFO] running postprocess ({model_name}, type={effective_type})...")
        print("[INFO] " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    if stage_output_pred.resolve() != output_pred.resolve():
        shutil.copyfile(stage_output_pred, output_pred)
    return output_pred


def resolve_eval_request(args, registry):
    canonical_model_name, model_cfg = resolve_model_config(registry, getattr(args, "model_name", None))
    meta_cfg = registry.get("_meta", {})
    infer_cfg = model_cfg.get("infer") or {}

    gt_json = resolve_path_arg(getattr(args, "gt_json", None)) or resolve_path_arg(meta_cfg.get("default_gt_json")) or DEFAULT_GT_JSON
    config_path = resolve_path_arg(getattr(args, "config", None)) or resolve_path_arg(infer_cfg.get("config"))
    checkpoint_path = resolve_path_arg(getattr(args, "checkpoint", None)) or resolve_path_arg(infer_cfg.get("checkpoint"))
    if config_path is not None and checkpoint_path is None:
        inferred_checkpoint = infer_checkpoint_from_config(str(config_path))
        if inferred_checkpoint is not None:
            checkpoint_path = Path(inferred_checkpoint)
            print(f"[INFO] inferred checkpoint: {checkpoint_path}")

    default_output_dir = resolve_path_arg(model_cfg.get("default_output_dir")) if model_cfg.get("default_output_dir") else None
    output_dir_source = "fallback"
    if getattr(args, "output_dir", None) is not None:
        output_dir = resolve_path_arg(args.output_dir)
        output_dir_source = "arg"
    elif config_path is not None and checkpoint_path is not None:
        output_dir = resolve_output_dir_from_config_and_checkpoint(str(config_path), str(checkpoint_path))
        output_dir_source = "config"
        print(f"[INFO] output directory derived from config/checkpoint: {output_dir}")
    elif default_output_dir is not None:
        output_dir = default_output_dir
        output_dir_source = "default"
    else:
        fallback_name = canonical_model_name or getattr(args, "model_name", None) or (infer_preferred_model_names_from_config(str(config_path))[0] if config_path is not None else "unknown_model")
        output_dir = WORKSPACE_ROOT / "work_dirs/eval_core4" / sanitize_output_token(fallback_name)

    stage_dirs = build_stage_dirs(output_dir)
    use_registry_stage_paths = output_dir_source == "default" and default_output_dir is not None and output_dir.resolve() == default_output_dir.resolve()

    category_spaces_path = resolve_path_arg(getattr(args, "category_spaces_json", None)) or resolve_path_arg(meta_cfg.get("category_spaces_json")) or DEFAULT_CATEGORY_SPACES_PATH
    category_spaces = load_category_spaces(category_spaces_path)
    requested_category_space = getattr(args, "category_space", None) or model_cfg.get("category_space") or meta_cfg.get("default_category_space") or DEFAULT_CATEGORY_SPACE_NAME
    category_space_name, category_space_cfg = resolve_category_space(category_spaces, requested_category_space)

    eval_cfg = model_cfg.get("eval") or {}
    eval_output_dir = resolve_path_arg(eval_cfg.get("output_dir")) if use_registry_stage_paths and eval_cfg.get("output_dir") else stage_dirs["eval"]

    final_pred_override = resolve_path_arg(getattr(args, "final_pred", None))
    if final_pred_override is not None:
        final_pred_path = final_pred_override
    elif use_registry_stage_paths and model_cfg.get("final_pred"):
        final_pred_path = resolve_path_arg(model_cfg.get("final_pred"))
    elif use_registry_stage_paths and (model_cfg.get("postprocess") or {}).get("output_pred"):
        final_pred_path = resolve_path_arg((model_cfg.get("postprocess") or {}).get("output_pred"))
    else:
        final_pred_path = stage_dirs["final"] / "predictions.ndjson"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)

    pred_search_bases = [stage_dirs["final"], stage_dirs["postprocess"], stage_dirs["transfer"], stage_dirs["infer"], output_dir]
    source_pred_path = None
    inferred_model_name = None

    if getattr(args, "pred", None) is not None:
        source_pred_path = resolve_existing_or_candidate_path(args.pred, base_dirs=pred_search_bases)
        inferred_model_name = infer_model_name_from_pred_path(source_pred_path)
    elif config_path is not None:
        source_pred_path, inferred_model_name = find_prediction_from_output_dir(output_dir, str(config_path))
        if source_pred_path is None:
            preferred_names = infer_preferred_model_names_from_config(str(config_path))
            inferred_model_name = preferred_names[0]
            source_pred_path = output_dir / f"{inferred_model_name}_merge_v2.ndjson"
    elif model_cfg.get("default_pred"):
        source_pred_path = resolve_path_arg(model_cfg["default_pred"])
        inferred_model_name = canonical_model_name
    else:
        source_pred_path, inferred_model_name = find_prediction_from_output_dir(output_dir)

    infer_runtime = None
    if getattr(args, "run_infer", False):
        infer_runtime = run_infer_for_model(
            canonical_model_name or getattr(args, "model_name", None) or "unknown",
            model_cfg,
            stage_dirs=stage_dirs,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=getattr(args, "device", None),
            image_root=resolve_path_arg(getattr(args, "image_root", None)),
            gt_json=gt_json,
            infer_options=args,
        )
        infer_out_pred = infer_runtime.get("out_pred")
        if infer_out_pred is not None and Path(infer_out_pred).exists():
            source_pred_path = Path(infer_out_pred)
        else:
            infer_search_dir = Path(infer_runtime.get("raw_output_dir") or stage_dirs["infer"])
            source_pred_path, inferred_from_infer = find_prediction_from_output_dir(infer_search_dir, str(config_path) if config_path is not None else None)
            if inferred_from_infer is not None:
                inferred_model_name = inferred_from_infer

    if getattr(args, "run_transfer", False):
        if not model_cfg.get("requires_transfer") and not model_cfg.get("transfer"):
            raise ValueError(f"model {canonical_model_name or getattr(args, 'model_name', None)} has no transfer configured")
        source_pred_path = run_transfer_for_model(
            canonical_model_name or getattr(args, "model_name", None) or "unknown",
            model_cfg,
            stage_dirs=stage_dirs,
            category_space_name=category_space_name,
            category_spaces_path=category_spaces_path,
            transfer_pred_root=getattr(args, "transfer_pred_root", None),
            transfer_gt_json=getattr(args, "transfer_gt_json", None),
            infer_runtime=infer_runtime,
        )
        inferred_model_name = canonical_model_name or inferred_model_name

    pred_path = None
    if final_pred_path.exists() and not any(
        [
            getattr(args, "run_infer", False),
            getattr(args, "run_transfer", False),
            getattr(args, "run_postprocess", False),
            getattr(args, "pred", None) is not None,
        ]
    ):
        pred_path = final_pred_path
    elif source_pred_path is not None:
        source_pred_path = Path(source_pred_path)
        if source_pred_path.exists() and source_pred_path.resolve() == final_pred_path.resolve():
            pred_path = final_pred_path
        else:
            if not source_pred_path.exists():
                raise FileNotFoundError(f"source predictions not found: {source_pred_path}")
            print("[INFO] generating/refreshing the standard NDJSON under final/")
            pred_path = run_postprocess_for_model(
                canonical_model_name or getattr(args, "model_name", None) or inferred_model_name or "unknown",
                model_cfg,
                input_pred=source_pred_path,
                gt_json=gt_json,
                stage_dirs=stage_dirs,
                category_space_name=category_space_name,
                category_spaces_path=category_spaces_path,
                final_pred_path=final_pred_path,
                postprocess_type=getattr(args, "postprocess_type", None),
                source_space_override=getattr(args, "source_space", None),
                source_class_names_override=getattr(args, "source_class_names", None),
            )
    elif final_pred_path.exists():
        pred_path = final_pred_path

    model_name = canonical_model_name or getattr(args, "model_name", None) or inferred_model_name
    if model_name is None and pred_path is not None:
        model_name = infer_model_name_from_pred_path(pred_path)
    if model_name is None:
        raise ValueError("cannot determine model_name; pass --model-name explicitly")
    if pred_path is None:
        raise ValueError("cannot determine the final prediction file; pass --pred or use --run-infer/--run-transfer")
    if Path(pred_path).resolve() != Path(final_pred_path).resolve():
        raise ValueError(f"evaluation only accepts the standard NDJSON under final/, got: {pred_path}")

    final_validation = validate_prediction_file(
        pred_path=Path(pred_path),
        gt_json=Path(gt_json),
        category_space_name=category_space_name,
        source_space_name=getattr(args, "source_space", None) or (model_cfg.get("postprocess") or {}).get("source_space"),
        source_class_names=getattr(args, "source_class_names", None) or (model_cfg.get("postprocess") or {}).get("source_class_names"),
        category_spaces_json=category_spaces_path,
        verbose=True,
    )
    write_json_file(final_pred_path.parent / "validation_summary.json", final_validation)

    return {
        "model_name": model_name,
        "registry_model_name": canonical_model_name,
        "registry_entry": model_cfg,
        "pred_path": Path(pred_path),
        "output_dir": output_dir,
        "stage_dirs": stage_dirs,
        "gt_json": Path(gt_json),
        "category_space_name": category_space_name,
        "category_space_cfg": category_space_cfg,
        "category_spaces_path": category_spaces_path,
        "eval_output_dir": Path(eval_output_dir),
        "final_pred_path": Path(final_pred_path),
        "final_validation": final_validation,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
    }


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="NDJSON GPU evaluator / infer-transfer-finalize-eval orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 -m mori_seg.eval.eval_ndjson_gpu --model-name cbnet_swin_tiny
    python3 -m mori_seg.eval.eval_ndjson_gpu --model-name rtmdet-ins_l --config mmdetection/configs/rtmdet/rtmdet-ins_l_kc.py --checkpoint /path/to/latest.pth --run-infer
    python3 -m mori_seg.eval.eval_ndjson_gpu --model-name yolov12 --run-transfer --run-postprocess
    python3 -m mori_seg.eval.eval_ndjson_gpu --pred EVAL_KPMP/pipeline_scis/results/scis_r50_merge_v2.ndjson --model-name scis_r50 --output-dir EVAL_KPMP/results/kc/scis_smoke
        """,
    )
    parser.add_argument("--pred", type=str, default=None, help="Source prediction NDJSON; it is finalized into final/ first")
    parser.add_argument("--gt-json", type=str, default=None, help="Path to the GT JSON")
    parser.add_argument("--model-name", type=str, default=None, help="Model name in the registry")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU and run on CPU")
    parser.add_argument("--subset-image-list", type=str, default=None, help="JSON file with a subset of image IDs")
    parser.add_argument("--subset-image-ids", type=str, default=None, help="Comma-separated subset image IDs")
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY_PATH), help="Model registry JSON")
    parser.add_argument("--config", type=str, default=None, help="Model config path, used to derive the output directory and prediction file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path, combined with --config to derive the output directory")
    parser.add_argument("--image-root", type=str, default=None, help="Optional image root directory for inference")
    parser.add_argument("--list-models", action="store_true", help="List the models in the registry")
    parser.add_argument("--run-infer", action="store_true", help="Run inference first when the model defines an infer stage")
    parser.add_argument("--run-transfer", action="store_true", help="Run the transfer stage first when the model defines one")
    parser.add_argument("--run-postprocess", action="store_true", help="Run postprocess/finalize explicitly; it also runs automatically when final/ is missing")
    parser.add_argument("--transfer-pred-root", type=str, default=None, help="Override the transfer input directory")
    parser.add_argument("--transfer-gt-json", type=str, default=None, help="Override the GT JSON used by transfer")
    parser.add_argument("--category-space", type=str, default=None, help="Category space name, e.g. kpmp_test_core4 / kc_test_mixed_11")
    parser.add_argument("--source-space", type=str, default=None, help="Source category space for postprocess, e.g. kc_train_13 / ki_train_6 / kpmp_test_core4")
    parser.add_argument("--source-class-names", type=str, default=None, help="Comma-separated source category names; takes precedence over --source-space")
    parser.add_argument("--category-spaces-json", type=str, default=str(DEFAULT_CATEGORY_SPACES_PATH), help="Category space configuration JSON")
    parser.add_argument("--postprocess-type", type=str, default=None, help="Override postprocess.type from the registry")
    parser.add_argument("--final-pred", type=str, default=None, help="Final prediction NDJSON; defaults to output_dir/final/predictions.ndjson")
    parser.add_argument("--workers", type=int, default=4, help="Number of preload workers for infer (default: 4)")
    parser.add_argument("--preload-mode", type=str, default="thread", choices=["thread", "process"], help="Preload mode for infer")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for infer (default: 1)")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during infer")
    parser.add_argument("--log-every", type=int, default=200, help="Logging interval for infer (default: 200)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of infer shards (default: 1)")
    parser.add_argument("--shard-id", type=int, default=0, help="Infer shard index (default: 0)")
    parser.add_argument("--export-objaware-map", action="store_true", help="Export objaware maps during infer")
    parser.add_argument("--objaware-map-dir", type=str, default=None, help="Output directory for objaware maps")
    parser.add_argument("--isdf-refine", action="store_true", help="Enable iSDF refinement during infer")
    parser.add_argument("--isdf-sigma", type=float, default=1.0, help="iSDF sigma")
    parser.add_argument("--isdf-h", type=float, default=2.0, help="iSDF h")
    parser.add_argument("--isdf-min-size", type=int, default=10, help="Minimum instance size in pixels for iSDF")
    parser.add_argument("--isdf-downscale", type=float, default=1.0, help="iSDF downscale factor")
    parser.add_argument("--isdf-topk", type=int, default=0, help="iSDF top-k")
    parser.add_argument("--isdf-min-area", type=int, default=0, help="Minimum area threshold for iSDF")
    return parser


def main(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    registry = load_registry(resolve_path_arg(args.registry) or DEFAULT_REGISTRY_PATH)

    if args.list_models:
        print_registry_models(registry)
        return 0

    if args.config is not None and args.checkpoint is None:
        inferred_checkpoint = infer_checkpoint_from_config(args.config)
        if inferred_checkpoint is not None:
            args.checkpoint = inferred_checkpoint
            print(f"[INFO] inferred checkpoint: {args.checkpoint}")

    subset_ids = resolve_subset_image_ids_from_args(args)
    if subset_ids:
        print(f"[INFO] subset evaluation enabled, images: {len(subset_ids)}")

    device = resolve_device(args.device, args.no_gpu)
    request = resolve_eval_request(args, registry)
    result = evaluate_model(
        model_name=request["model_name"],
        pred_path=request["pred_path"],
        out_dir=request["output_dir"],
        gt_json=request["gt_json"],
        device=device,
        subset_image_ids=subset_ids,
        registry_model_name=request["registry_model_name"],
        registry_entry=request["registry_entry"],
        category_space_name=request["category_space_name"],
        category_space_cfg=request["category_space_cfg"],
        eval_dir=request["eval_output_dir"],
        final_pred_path=request["final_pred_path"],
    )
    print_results_table(result)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
