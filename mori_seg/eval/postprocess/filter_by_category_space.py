#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter and remap predictions to a category space and write a standard NDJSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import orjson as _orjson
except Exception:
    _orjson = None

from ..category_utils import (
    DEFAULT_CATEGORY_SPACE_NAME,
    DEFAULT_CATEGORY_SPACES_PATH,
    build_target_categories,
    load_category_spaces,
    remap_predictions_to_category_space,
    resolve_category_space,
    resolve_path_arg,
)


def dumps_bytes(obj):
    if _orjson is not None:
        return _orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


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


def write_ndjson(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for record in records:
            f.write(dumps_bytes(record))
            f.write(b"\n")


def load_gt_categories(gt_json_path):
    if gt_json_path is None or not gt_json_path.exists():
        return []
    with open(gt_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("categories", [])


def build_source_categories(category_spaces, gt_categories, *, source_space=None, source_class_names=None):
    if source_class_names:
        names = [token.strip() for token in source_class_names.split(",") if token.strip()]
        return [{"id": idx + 1, "name": name} for idx, name in enumerate(names)]
    if source_space:
        _, source_space_cfg = resolve_category_space(category_spaces, source_space)
        return build_target_categories(source_space_cfg)
    return list(gt_categories or [])


def apply_topk_per_image(predictions, topk_per_image):
    if topk_per_image is None or topk_per_image <= 0:
        return predictions

    grouped = defaultdict(list)
    for pred in predictions:
        grouped[int(pred.get("image_id", -1))].append(pred)

    kept = []
    for image_id in sorted(grouped):
        image_preds = sorted(
            grouped[image_id],
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        kept.extend(image_preds[:topk_per_image])
    return kept


def sort_predictions(predictions):
    return sorted(
        predictions,
        key=lambda item: (
            int(item.get("image_id", -1)),
            -float(item.get("score", 0.0)),
            int(item.get("category_id", -1)),
        ),
    )


def process_prediction_file(
    *,
    pred,
    out_pred,
    gt_json=None,
    category_space=None,
    category_spaces_json=DEFAULT_CATEGORY_SPACES_PATH,
    source_space=None,
    source_class_names=None,
    topk_per_image=0,
):
    pred_path = resolve_path_arg(pred)
    out_pred_path = resolve_path_arg(out_pred)
    gt_json_path = resolve_path_arg(gt_json)
    category_spaces_path = resolve_path_arg(category_spaces_json) or DEFAULT_CATEGORY_SPACES_PATH

    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"input predictions not found: {pred_path}")

    category_spaces = load_category_spaces(category_spaces_path)
    canonical_space_name, category_space_cfg = resolve_category_space(
        category_spaces,
        category_space or DEFAULT_CATEGORY_SPACE_NAME,
    )
    gt_categories = load_gt_categories(gt_json_path)
    source_categories = build_source_categories(
        category_spaces,
        gt_categories,
        source_space=source_space,
        source_class_names=source_class_names,
    )

    predictions = load_ndjson(pred_path)
    remapped_predictions = remap_predictions_to_category_space(
        predictions,
        category_space_cfg,
        source_categories=source_categories,
    )
    remapped_predictions = apply_topk_per_image(remapped_predictions, topk_per_image)
    remapped_predictions = sort_predictions(remapped_predictions)
    write_ndjson(out_pred_path, remapped_predictions)

    return {
        "pred_path": pred_path,
        "out_pred_path": out_pred_path,
        "canonical_space_name": canonical_space_name,
        "input_count": len(predictions),
        "output_count": len(remapped_predictions),
        "source_category_count": len(source_categories),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and remap NDJSON predictions to a category space")
    parser.add_argument("--pred", type=str, required=True, help="Input NDJSON")
    parser.add_argument("--out-pred", type=str, required=True, help="Output NDJSON")
    parser.add_argument("--gt-json", type=str, default=None, help="Optional GT JSON, used to resolve category_id to category_name")
    parser.add_argument("--category-space", type=str, default=DEFAULT_CATEGORY_SPACE_NAME, help="Target category space name")
    parser.add_argument("--category-spaces-json", type=str, default=str(DEFAULT_CATEGORY_SPACES_PATH), help="Category space configuration JSON")
    parser.add_argument("--source-space", type=str, default=None, help="Source category space name, e.g. train_6 / core4")
    parser.add_argument("--source-class-names", type=str, default=None, help="Comma-separated source category names; takes precedence over --source-space")
    parser.add_argument("--topk-per-image", type=int, default=0, help="Keep at most the top N predictions per image; 0 means no limit")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = process_prediction_file(
        pred=args.pred,
        out_pred=args.out_pred,
        gt_json=args.gt_json,
        category_space=args.category_space,
        category_spaces_json=args.category_spaces_json,
        source_space=args.source_space,
        source_class_names=args.source_class_names,
        topk_per_image=args.topk_per_image,
    )

    print("=" * 60)
    print("Postprocess: filter_by_category_space")
    print("=" * 60)
    print(f"Input predictions: {summary['pred_path']}")
    print(f"Output predictions: {summary['out_pred_path']}")
    print(f"Category space: {summary['canonical_space_name']}")
    print(f"Source categories: {summary['source_category_count']}")
    print(f"Input records: {summary['input_count']}")
    print(f"Output records: {summary['output_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
