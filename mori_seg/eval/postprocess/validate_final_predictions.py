#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate that a final NDJSON lies entirely within the target test category space."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import orjson as _orjson
except Exception:
    _orjson = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ..category_utils import (
    DEFAULT_CATEGORY_SPACE_NAME,
    DEFAULT_CATEGORY_SPACES_PATH,
    load_category_spaces,
    resolve_category_space,
    resolve_path_arg,
)


DEFAULT_GT_JSON = "KC/annotations/test.json"


def load_ndjson(path):
    records = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if _orjson is not None:
                records.append(_orjson.loads(line))
            else:
                records.append(json.loads(line.decode("utf-8")))
    return records


def load_gt_category_index(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    categories = payload.get("categories", [])
    name_to_id = {str(cat["name"]): int(cat["id"]) for cat in categories}
    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
    return name_to_id, id_to_name


def infer_banned_source_names(category_spaces, *, source_space_name=None, source_class_names=None, target_names=None):
    target_names = set(target_names or [])
    if source_class_names:
        source_names = {token.strip() for token in str(source_class_names).split(",") if token.strip()}
        return source_names - target_names
    if source_space_name:
        _, source_space_cfg = resolve_category_space(category_spaces, source_space_name)
        source_names = set(source_space_cfg.get("categories", []))
        return source_names - target_names
    return set()


def validate_prediction_file(
    *,
    pred_path,
    gt_json,
    category_space_name=DEFAULT_CATEGORY_SPACE_NAME,
    source_space_name=None,
    source_class_names=None,
    category_spaces_json=DEFAULT_CATEGORY_SPACES_PATH,
    verbose=True,
):
    pred_path = resolve_path_arg(pred_path)
    gt_json_path = resolve_path_arg(gt_json)
    category_spaces_path = resolve_path_arg(category_spaces_json) or DEFAULT_CATEGORY_SPACES_PATH

    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"final predictions not found: {pred_path}")
    if gt_json_path is None or not gt_json_path.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_json_path}")

    category_spaces = load_category_spaces(category_spaces_path)
    canonical_space_name, category_space_cfg = resolve_category_space(category_spaces, category_space_name)
    target_name_to_id, target_id_to_name = load_gt_category_index(gt_json_path)
    expected_space_names = set(category_space_cfg.get("categories", []))
    target_gt_names = set(target_name_to_id.keys())
    banned_source_names = infer_banned_source_names(
        category_spaces,
        source_space_name=source_space_name,
        source_class_names=source_class_names,
        target_names=target_gt_names,
    )

    predictions = load_ndjson(pred_path)
    invalid_names = Counter()
    invalid_ids = Counter()
    invalid_pairs = Counter()
    residual_source_names = Counter()

    for pred in predictions:
        category_name = pred.get("category_name")
        category_id_raw = pred.get("category_id")
        try:
            category_id = int(category_id_raw)
        except (TypeError, ValueError):
            category_id = None

        resolved_name = category_name if category_name is not None else target_id_to_name.get(category_id)
        if resolved_name is None or resolved_name not in target_gt_names:
            invalid_names[str(resolved_name)] += 1

        if category_id is None or category_id not in target_id_to_name:
            invalid_ids[str(category_id_raw)] += 1

        if category_name is not None:
            expected_id = target_name_to_id.get(category_name)
            if expected_id is None or category_id != expected_id:
                invalid_pairs[f"{category_name}:{category_id_raw}"] += 1

        if category_name in banned_source_names:
            residual_source_names[str(category_name)] += 1

    summary = {
        "pred_path": str(pred_path),
        "gt_json": str(gt_json_path),
        "category_space": canonical_space_name,
        "target_category_count": len(target_name_to_id),
        "target_categories": [name for name, _ in sorted(target_name_to_id.items(), key=lambda item: item[1])],
        "target_space_matches_gt": expected_space_names == target_gt_names,
        "total_predictions": len(predictions),
        "source_space": source_space_name,
        "source_class_names": source_class_names,
        "banned_source_names": sorted(banned_source_names),
        "invalid_category_names": dict(invalid_names),
        "invalid_category_ids": dict(invalid_ids),
        "invalid_category_pairs": dict(invalid_pairs),
        "residual_source_names_in_category_name": dict(residual_source_names),
        "valid": not (invalid_names or invalid_ids or invalid_pairs or residual_source_names),
    }

    if verbose:
        print("=" * 60)
        print("Validate final predictions")
        print("=" * 60)
        print(f"Prediction file: {pred_path}")
        print(f"GT JSON: {gt_json_path}")
        print(f"Category space: {canonical_space_name}")
        print(f"Target categories: {len(target_name_to_id)}")
        print(f"Predictions: {len(predictions)}")
        if expected_space_names != target_gt_names:
            print("[WARN] category space differs from the GT categories; validating against the GT categories")
        if banned_source_names:
            print(f"Disallowed residual category names: {sorted(banned_source_names)}")

    if not summary["valid"]:
        raise ValueError(
            "final NDJSON validation failed: "
            f"invalid_names={dict(invalid_names)}, "
            f"invalid_ids={dict(invalid_ids)}, "
            f"invalid_pairs={dict(invalid_pairs)}, "
            f"residual_source_names={dict(residual_source_names)}"
        )

    if verbose:
        print("[OK] final NDJSON validation passed")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Validate that a final NDJSON lies entirely within the target test category space")
    parser.add_argument("--pred", type=str, required=True, help="Final NDJSON to validate")
    parser.add_argument("--gt-json", type=str, default=DEFAULT_GT_JSON, help="Target test GT JSON")
    parser.add_argument("--category-space", type=str, default=DEFAULT_CATEGORY_SPACE_NAME, help="Target category space name")
    parser.add_argument("--source-space", type=str, default=None, help="Source category space name, used to derive disallowed residual category names")
    parser.add_argument("--source-class-names", type=str, default=None, help="Comma-separated source category names; takes precedence over --source-space")
    parser.add_argument("--category-spaces-json", type=str, default=str(DEFAULT_CATEGORY_SPACES_PATH), help="Category space configuration JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_prediction_file(
        pred_path=args.pred,
        gt_json=args.gt_json,
        category_space_name=args.category_space,
        source_space_name=args.source_space,
        source_class_names=args.source_class_names,
        category_spaces_json=args.category_spaces_json,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
