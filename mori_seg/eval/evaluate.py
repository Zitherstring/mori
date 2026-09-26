#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NDJSON GPU evaluator entry point."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
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
DEFAULT_CATEGORY_ORDER = list(CATEGORY_DEFAULT_ORDER)
SIZE_PATTERN_10X = re.compile(r"_2048x2048\.png$", re.IGNORECASE)
SIZE_PATTERN_40X = re.compile(r"_512x512\.png$", re.IGNORECASE)


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


def infer_model_name_from_pred_path(pred_path):
    pred_path = Path(pred_path)
    stem = pred_path.stem
    if stem == "predictions" and pred_path.parent.name:
        return pred_path.parent.parent.name if pred_path.parent.name in {"infer", "transfer", "postprocess", "final", "eval"} and pred_path.parent.parent.name else pred_path.parent.name
    return re.sub(r"_predictions$", "", stem)


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
    category_space_name=None,
    category_space_cfg=None,
    final_pred_path=None,
):
    print("=" * 80)
    print(f"Evaluating: {model_name}")
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
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1] Loading data...")
    predictions = load_ndjson(pred_path)
    coco_gt_all = COCO(str(gt_json))
    temp_category_gt = None

    try:
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
                coco_stdout_path = out_dir / "coco_stdout.txt"
                coco_stdout_path.write_text(coco_stdout, encoding="utf-8")
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
        csv_output_path = out_dir / f"per_image_metrics_{model_name}.csv"
        save_per_image_results_to_csv(per_image_results, csv_output_path, gt_data["categories"])

        output = {
            "model": model_name,
            "pred_file": str(pred_path),
            "final_pred_file": str(final_pred_path or pred_path),
            "output_dir": str(out_dir),
            "gt_json": str(gt_json),
            "gpu_accelerated": device is not None,
            "image_count": len(coco_gt.getImgIds()),
            "category_space": category_space_name,
            "category_space_order": [cat["name"] for cat in gt_data.get("categories", [])],
            "magnification_stats": {"10x_images": count_10x, "40x_images": count_40x},
            "segm": {"overall": segm_overall, "per_class": segm_per_class} if segm_overall is not None else None,
            "semantic_iou": semantic_iou_results,
            "f1": f1_results,
        }
        result_json_path = out_dir / f"eval_results_{model_name}.json"
        write_json_file(result_json_path, output)
        print(f"\n[OK] results saved: {result_json_path}")
        return output
    finally:
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


def resolve_eval_request(args):
    gt_json = resolve_path_arg(getattr(args, "gt_json", None)) or DEFAULT_GT_JSON
    config_path = resolve_path_arg(getattr(args, "config", None))
    checkpoint_path = resolve_path_arg(getattr(args, "checkpoint", None))

    if getattr(args, "output_dir", None) is not None:
        output_dir = resolve_path_arg(args.output_dir)
    else:
        fallback_name = getattr(args, "model_name", None) or "unknown_model"
        output_dir = WORKSPACE_ROOT / "work_dirs/eval" / fallback_name

    output_dir.mkdir(parents=True, exist_ok=True)

    category_spaces_path = resolve_path_arg(getattr(args, "category_spaces_json", None)) or DEFAULT_CATEGORY_SPACES_PATH
    category_spaces = load_category_spaces(category_spaces_path)
    requested_category_space = getattr(args, "category_space", None) or DEFAULT_CATEGORY_SPACE_NAME
    category_space_name, category_space_cfg = resolve_category_space(category_spaces, requested_category_space)

    final_pred_override = resolve_path_arg(getattr(args, "final_pred", None))
    if final_pred_override is not None:
        final_pred_path = final_pred_override
    else:
        final_pred_path = output_dir / "predictions.ndjson"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)

    pred_search_bases = [output_dir]
    source_pred_path = None
    inferred_model_name = None

    if getattr(args, "pred", None) is not None:
        source_pred_path = resolve_existing_or_candidate_path(args.pred, base_dirs=pred_search_bases)
        inferred_model_name = infer_model_name_from_pred_path(source_pred_path)

    pred_path = None
    if source_pred_path is not None:
        source_pred_path = Path(source_pred_path)
        if not source_pred_path.exists():
            raise FileNotFoundError(f"source predictions not found: {source_pred_path}")
        if source_pred_path.resolve() != final_pred_path.resolve():
            raise ValueError(
                "evaluation only accepts the final prediction NDJSON; "
                f"--pred ({source_pred_path}) and --final-pred ({final_pred_path}) must point at the same file"
            )
        pred_path = final_pred_path
    elif final_pred_path.exists():
        pred_path = final_pred_path

    model_name = getattr(args, "model_name", None) or inferred_model_name
    if model_name is None and pred_path is not None:
        model_name = infer_model_name_from_pred_path(pred_path)
    if model_name is None:
        raise ValueError("cannot determine model_name; pass --model-name explicitly")
    if pred_path is None:
        raise ValueError("cannot determine the final prediction file; pass --pred/--final-pred")

    final_validation = validate_prediction_file(
        pred_path=Path(pred_path),
        gt_json=Path(gt_json),
        category_space_name=category_space_name,
        source_space_name=getattr(args, "source_space", None),
        source_class_names=getattr(args, "source_class_names", None),
        category_spaces_json=category_spaces_path,
        verbose=True,
    )
    write_json_file(output_dir / "validation_summary.json", final_validation)

    return {
        "model_name": model_name,
        "pred_path": Path(pred_path),
        "output_dir": output_dir,
        "gt_json": Path(gt_json),
        "category_space_name": category_space_name,
        "category_space_cfg": category_space_cfg,
        "category_spaces_path": category_spaces_path,
        "final_pred_path": Path(final_pred_path),
        "final_validation": final_validation,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
    }


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="NDJSON GPU evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 -m mori_seg.eval.evaluate --pred work_dirs/eval/run/model_predictions.ndjson --final-pred work_dirs/eval/run/model_predictions.ndjson --model-name model --output-dir work_dirs/eval/run
        """,
    )
    parser.add_argument("--pred", type=str, default=None, help="Final prediction NDJSON to evaluate")
    parser.add_argument("--gt-json", type=str, default=None, help="Path to the GT JSON")
    parser.add_argument("--model-name", type=str, default=None, help="Model name used to name the output files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU and run on CPU")
    parser.add_argument("--config", type=str, default=None, help="Model config path, recorded with the evaluation request")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path, recorded with the evaluation request")
    parser.add_argument("--category-space", type=str, default=None, help="Category space name, e.g. core4")
    parser.add_argument("--source-space", type=str, default=None, help="Source category space used when validating predictions, e.g. train_6")
    parser.add_argument("--source-class-names", type=str, default=None, help="Comma-separated source category names; takes precedence over --source-space")
    parser.add_argument("--category-spaces-json", type=str, default=str(DEFAULT_CATEGORY_SPACES_PATH), help="Category space configuration JSON")
    parser.add_argument("--final-pred", type=str, default=None, help="Final prediction NDJSON; defaults to output_dir/final/predictions.ndjson")
    return parser


def main(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device, args.no_gpu)
    request = resolve_eval_request(args)
    result = evaluate_model(
        model_name=request["model_name"],
        pred_path=request["pred_path"],
        out_dir=request["output_dir"],
        gt_json=request["gt_json"],
        device=device,
        category_space_name=request["category_space_name"],
        category_space_cfg=request["category_space_cfg"],
        final_pred_path=request["final_pred_path"],
    )
    print_results_table(result)
    if result is None:
        return 1
    if result.get("segm") is None:
        print("[ERROR] COCOeval produced no segm metrics")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
