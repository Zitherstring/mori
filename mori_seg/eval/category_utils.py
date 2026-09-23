#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Category space definitions, category remapping and magnification filtering helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATEGORY_SPACES_PATH = Path(__file__).with_name("category_spaces.json")
DEFAULT_CATEGORY_ORDER = [
    "0_2_podocyte",
    "1_2_mesangial",
    "1_6_2_mv",
    "2_2_endo",
    "2_4_2_smooth",
    "2_5_2_vesselendo",
    "3_2_pecs",
    "arteries_arterioles",
    "non-globally-sclerotic_glomeruli",
    "peritubular-capillaries",
    "tubules",
]
DEFAULT_CATEGORY_SPACE_NAME = "kc_test_mixed_11"
CLASSES_10X_NAMES = {
    "0_2_podocyte",
    "1_2_mesangial",
    "1_6_2_mv",
    "2_2_endo",
    "2_4_2_smooth",
    "2_5_2_vesselendo",
    "3_2_pecs",
    "arteries_arterioles",
    "non-globally-sclerotic_glomeruli",
    "tubules",
}
CLASSES_40X_NAMES = {
    "0_2_podocyte",
    "1_2_mesangial",
    "1_6_2_mv",
    "2_2_endo",
    "2_4_2_smooth",
    "2_5_2_vesselendo",
    "3_2_pecs",
    "peritubular-capillaries",
}


def resolve_path_arg(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_category_spaces(path=DEFAULT_CATEGORY_SPACES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_category_space(category_spaces, space_name=None):
    spaces = category_spaces.get("spaces", {})
    desired = space_name or category_spaces.get("_meta", {}).get("default_category_space") or DEFAULT_CATEGORY_SPACE_NAME
    if desired in spaces:
        return desired, spaces[desired]

    for canonical_name, cfg in spaces.items():
        if desired in cfg.get("aliases", []):
            return canonical_name, cfg
    raise KeyError(f"unknown category space: {desired}")


def get_category_order(category_space_cfg):
    return list(category_space_cfg.get("categories", []))


def build_target_categories(category_space_cfg):
    return [{"id": idx + 1, "name": name} for idx, name in enumerate(get_category_order(category_space_cfg))]


def build_category_id_lookup(category_space_cfg):
    return {cat["name"]: int(cat["id"]) for cat in build_target_categories(category_space_cfg)}


def remap_category_name(source_name, category_space_cfg):
    if not source_name:
        return None

    category_remap = category_space_cfg.get("category_remap") or {}
    if source_name in category_remap:
        mapped = category_remap[source_name]
        return mapped if mapped else None

    target_categories = set(get_category_order(category_space_cfg))
    if source_name in target_categories:
        return source_name
    return None


def get_valid_categories_for_magnification(mag, category_space_cfg=None):
    if category_space_cfg is None:
        if mag == "40x":
            return set(CLASSES_40X_NAMES)
        if mag == "10x":
            return set(CLASSES_10X_NAMES)
        return set(DEFAULT_CATEGORY_ORDER)

    magnification_allowed = category_space_cfg.get("magnification_allowed") or {}
    if not magnification_allowed:
        return set(get_category_order(category_space_cfg))

    valid = magnification_allowed.get(mag)
    if valid is None:
        valid = magnification_allowed.get("default")
    if valid is None:
        valid = magnification_allowed.get("all")
    if valid is None and mag == "unknown":
        valid = magnification_allowed.get("unknown")
    if valid is None:
        valid = magnification_allowed.get("unknown", [])
    return set(valid or [])


def build_filtered_dataset_skeleton(dataset, category_space_cfg):
    return {
        "info": copy.deepcopy(dataset.get("info", {})),
        "licenses": copy.deepcopy(dataset.get("licenses", [])),
        "type": copy.deepcopy(dataset.get("type", "instances")),
        "images": [],
        "annotations": [],
        "categories": build_target_categories(category_space_cfg),
    }


def remap_coco_dataset(dataset, category_space_cfg, fix_iscrowd=False):
    source_cat_id_to_name = {int(cat["id"]): cat["name"] for cat in dataset.get("categories", [])}
    target_name_to_id = build_category_id_lookup(category_space_cfg)
    remapped = build_filtered_dataset_skeleton(dataset, category_space_cfg)
    remapped["images"] = [copy.deepcopy(img) for img in dataset.get("images", [])]

    for ann in dataset.get("annotations", []):
        source_name = source_cat_id_to_name.get(int(ann.get("category_id", -1)))
        target_name = remap_category_name(source_name, category_space_cfg)
        if target_name is None:
            continue

        ann_copy = copy.deepcopy(ann)
        ann_copy["category_id"] = target_name_to_id[target_name]
        if fix_iscrowd:
            ann_copy["iscrowd"] = 0
        remapped["annotations"].append(ann_copy)
    return remapped


def resolve_prediction_source_category_name(pred, source_cat_id_to_name=None, target_order=None):
    if pred.get("category_name"):
        return pred["category_name"]
    if pred.get("source_category_name"):
        return pred["source_category_name"]

    category_id = pred.get("category_id")
    if category_id is None:
        return None

    try:
        category_id = int(category_id)
    except (TypeError, ValueError):
        return None

    if source_cat_id_to_name and category_id in source_cat_id_to_name:
        return source_cat_id_to_name[category_id]
    if target_order and 1 <= category_id <= len(target_order):
        return target_order[category_id - 1]
    return None


def normalize_prediction_record(pred, *, target_name, target_id, source_name=None):
    segmentation = pred.get("segmentation")
    if segmentation is None:
        return None

    try:
        image_id = int(pred.get("image_id"))
    except (TypeError, ValueError):
        return None

    try:
        score = float(pred.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    out = {
        "image_id": image_id,
        "category_id": int(target_id),
        "category_name": target_name,
        "score": score,
        "segmentation": copy.deepcopy(segmentation),
    }

    source_category_id = pred.get("source_category_id")
    if source_category_id is None and pred.get("category_id") is not None:
        source_category_id = pred.get("category_id")

    if source_name:
        out["source_category_name"] = source_name
    if source_category_id is not None:
        try:
            out["source_category_id"] = int(source_category_id)
        except (TypeError, ValueError):
            pass

    for key in ("bbox", "area", "iscrowd", "image_path", "file_name"):
        if key in pred:
            out[key] = copy.deepcopy(pred[key])
    return out


def remap_predictions_to_category_space(
    predictions,
    category_space_cfg,
    source_categories: Optional[Sequence[Mapping[str, Any]]] = None,
):
    source_cat_id_to_name: Dict[int, str] = {}
    if isinstance(source_categories, Mapping):
        source_cat_id_to_name = {int(k): str(v) for k, v in source_categories.items()}
    elif source_categories:
        source_cat_id_to_name = {int(cat["id"]): cat["name"] for cat in source_categories}

    target_order = get_category_order(category_space_cfg)
    target_name_to_id = build_category_id_lookup(category_space_cfg)
    remapped_predictions: List[Dict[str, Any]] = []

    for pred in predictions:
        source_name = resolve_prediction_source_category_name(
            pred,
            source_cat_id_to_name=source_cat_id_to_name,
            target_order=target_order,
        )
        target_name = remap_category_name(source_name, category_space_cfg) if source_name else None
        if target_name is None:
            continue

        normalized = normalize_prediction_record(
            pred,
            target_name=target_name,
            target_id=target_name_to_id[target_name],
            source_name=source_name,
        )
        if normalized is not None:
            remapped_predictions.append(normalized)
    return remapped_predictions
