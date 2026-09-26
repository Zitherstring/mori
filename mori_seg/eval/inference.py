#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTMDet-Ins instance segmentation inference.

Runs (optionally sharded) inference over a COCO-style test split and writes the
post-processed predictions as NDJSON.

Speed options:
---------
1. Disable Mask NMS (--no-mask-nms): faster post-processing
2. Parallel image pre-loading (--workers/--preload-mode)
3. Larger batches (--batch-size)
4. Graceful Ctrl+C exit

Category mapping:
---------
6 training classes:
    0: cap, 1: dt, 2: pt, 3: ptc, 4: tuft, 5: ves

4 evaluation classes (GT category_id):
    1: arteries_arterioles
    2: non-globally-sclerotic_glomeruli
    3: peritubular-capillaries
    4: tubules

Mapping:
    cap (0)  -> glomeruli (2)
    tuft (4) -> dropped (not evaluated)
    dt (1)   -> tubules (4)
    pt (2)   -> tubules (4)
    ptc (3)  -> ptc (3)
    ves (5)  -> arteries (1)

Usage:
    python inference.py --config CFG --checkpoint CKPT --gt-json GT.json
    python inference.py --config CFG --checkpoint CKPT --gt-json GT.json --workers 4
    python inference.py --config CFG --checkpoint CKPT --gt-json GT.json --no-mask-nms
"""

import os
import sys
import re
import signal
import argparse
import json
import glob
import time
import random
import multiprocessing as mp
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    import orjson as _orjson
except Exception:
    _orjson = None

# Limit numerical library thread counts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Optional local mmdetection checkout (set MMDET_ROOT to prepend it to sys.path)
_MMDET_ROOT = os.environ.get("MMDET_ROOT")
if _MMDET_ROOT:
    sys.path.insert(0, _MMDET_ROOT)

import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from mmengine.structures import InstanceData
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


# ==================== Paths ====================
# Test-set GT JSON providing the image list and metadata (set with --gt-json)
GT_JSON = None
# Image root directory (set with --img-root, or inferred from the GT JSON)
IMG_ROOT = None
# Default output directory (override with --output-dir)
OUT_DIR = Path(__file__).parent / "results"
# Root directory holding mmdetection work_dirs (override with MMDET_WORK_DIRS)
WORK_DIRS_ROOT = Path(os.environ.get("MMDET_WORK_DIRS", "work_dirs"))

# ==================== Model ====================
MODEL_CONFIG = {
    "config": "",
    "checkpoint": "",
    "output_name": "rtmdet-ins_l",
}

# ==================== Categories ====================
# 6 class names (training model)
NAMES_6 = ["cap", "dt", "pt", "ptc", "tuft", "ves"]

# 4 class names (evaluation GT)
NAMES_4 = ["arteries_arterioles", "non-globally-sclerotic_glomeruli", "peritubular-capillaries", "tubules"]

# 6-class -> 4-class mapping table.
# Values are 4-class indices (0-3); +1 gives the GT category_id.
# None means the class is not evaluated.
MAP_6to4_ID = {
    0: 1,    # cap -> glomeruli (index 1 -> GT cat_id=2)
    1: 3,    # dt -> tubules (index 3 -> GT cat_id=4)
    2: 3,    # pt -> tubules
    3: 2,    # ptc -> ptc (index 2 -> GT cat_id=3)
    4: None, # tuft -> dropped
    5: 0,    # ves -> arteries (index 0 -> GT cat_id=1)
}

# 4-class prediction index -> GT category_id
PRED_TO_GT_CAT_ID = {
    0: 1,  # arteries_arterioles
    1: 2,  # non-globally-sclerotic_glomeruli
    2: 3,  # peritubular-capillaries
    3: 4,  # tubules
}

# Magnification to category mapping
# 6-class IDs evaluated on 10x images (tuft excluded)
CLASSES_6_FOR_10X = {0, 1, 2, 5}  # cap, dt, pt, ves
# 6-class IDs evaluated on 40x images
CLASSES_6_FOR_40X = {3}  # ptc only

# ==================== Inference ====================
# Per-class score thresholds (6-class model IDs)
PER_CLASS_SCORE_THRES = {
    0: 0.05,   # cap → glomeruli
    1: 0.10,   # dt → tubules
    2: 0.10,   # pt → tubules
    3: 0.15,   # ptc
    4: 0.50,   # tuft: not evaluated
    5: 0.05,   # ves → arteries
}

# Per-class minimum mask area (pixels)
MIN_MASK_AREA = {
    0: 100,    # cap (glomeruli)
    1: 50,     # dt (tubules)
    2: 50,     # pt (tubules)
    3: 30,     # ptc
    4: 50,     # tuft
    5: 50,     # ves (arteries)
}

# Mask NMS IoU thresholds (4 evaluation classes)
MASK_NMS_THRES = {
    0: 0.50,   # arteries
    1: 0.50,   # glomeruli
    2: 0.40,   # ptc
    3: 0.40,   # tubules
}

# Mask NMS containment threshold: inter / min(area_i, area_j).
# < 1 suppresses near-duplicates that are highly contained but have low IoU.
# >= 1 disables containment suppression and keeps IoU NMS only.
MASK_NMS_CONTAIN_THRES = 1.01

# Global settings
SCORE_THRES = 0.05  # global minimum score (fallback)
MAX_DETS = 300      # max detections per image
ENABLE_MASK_NMS = True  # enable Mask NMS

# Interrupt flag
_interrupted = False


def json_dumps_bytes(payload: dict) -> bytes:
    """Serialize to one NDJSON line as bytes (orjson when available)."""
    if _orjson is not None:
        return _orjson.dumps(payload)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_subset_image_ids(path: Path) -> set:
    """Load a subset image-ID list: list[int] or {"image_ids": [...]}."""
    if path is None:
        return set()
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


def stratified_sample_images(images: list,
                             subset_size: int = 0,
                             subset_ratio: float = 1.0,
                             subset_seed: int = 42) -> list:
    """Stratified sampling by magnification to keep the 10x/40x ratio stable."""
    total = len(images)
    if total == 0:
        return images

    ratio = float(subset_ratio) if subset_ratio is not None else 1.0
    ratio = min(max(ratio, 0.0), 1.0)

    if subset_size is not None and int(subset_size) > 0:
        target = int(subset_size)
    else:
        target = int(round(total * ratio))

    target = max(1, min(target, total))
    if target >= total:
        return images

    groups = {
        "10x": [],
        "40x": [],
        "unknown": [],
    }
    for im in images:
        mag = get_magnification_from_filename(im.get("file_name", ""))
        if mag not in groups:
            mag = "unknown"
        groups[mag].append(im)

    rng = random.Random(int(subset_seed))
    for k in groups:
        rng.shuffle(groups[k])

    alloc = {}
    frac_parts = []
    non_empty_keys = [k for k, v in groups.items() if len(v) > 0]
    if not non_empty_keys:
        rng.shuffle(images)
        sampled = images[:target]
        sampled.sort(key=lambda x: int(x["id"]))
        return sampled

    for k in non_empty_keys:
        exact = target * (len(groups[k]) / total)
        cnt = int(np.floor(exact))
        alloc[k] = min(cnt, len(groups[k]))
        frac_parts.append((exact - cnt, k))

    current = sum(alloc.values())

    # Distribute the remainder by fractional part
    if current < target:
        frac_parts.sort(reverse=True)
        for _, k in frac_parts:
            if current >= target:
                break
            if alloc[k] < len(groups[k]):
                alloc[k] += 1
                current += 1

    # Still short because of capacity limits: top up from groups with spare capacity
    if current < target:
        for k in non_empty_keys:
            if current >= target:
                break
            spare = len(groups[k]) - alloc[k]
            if spare <= 0:
                continue
            take = min(spare, target - current)
            alloc[k] += take
            current += take

    # Over target: reduce the largest allocations first
    if current > target:
        for k in sorted(non_empty_keys, key=lambda x: alloc[x], reverse=True):
            if current <= target:
                break
            reducible = min(alloc[k], current - target)
            alloc[k] -= reducible
            current -= reducible

    sampled = []
    for k in non_empty_keys:
        sampled.extend(groups[k][:alloc[k]])

    # Final adjustment to hit the exact count
    if len(sampled) < target:
        picked_ids = {int(x["id"]) for x in sampled}
        rest = [im for im in images if int(im["id"]) not in picked_ids]
        rng.shuffle(rest)
        sampled.extend(rest[: target - len(sampled)])
    elif len(sampled) > target:
        rng.shuffle(sampled)
        sampled = sampled[:target]

    sampled.sort(key=lambda x: int(x["id"]))
    return sampled


def extract_checkpoint_tag(checkpoint_path: str) -> str:
    """
    Derive a directory tag from the checkpoint file name.

    Prefers epoch_<number> (returns the number), then latest, otherwise the file stem.
    """
    ckpt_name = Path(checkpoint_path).name
    m = re.search(r'epoch_(\d+)', ckpt_name)
    if m:
        return m.group(1)
    if re.search(r'latest', ckpt_name, flags=re.IGNORECASE):
        return 'latest'
    return Path(checkpoint_path).stem


def resolve_default_output_dir(config_path: str, checkpoint_path: str) -> Path:
    """
    Resolve the default output directory: work_dirs/<config_stem>/<checkpoint_tag>/
    """
    config_stem = Path(config_path).stem
    checkpoint_tag = extract_checkpoint_tag(checkpoint_path)
    return WORK_DIRS_ROOT / config_stem / checkpoint_tag


def infer_default_image_root(gt_json_path: Path) -> Path:
    """Infer the image root directory from annotations/<split>.json."""
    gt_json_path = Path(gt_json_path)
    dataset_root = gt_json_path.parent.parent if gt_json_path.parent.name == "annotations" else gt_json_path.parent
    split_name = gt_json_path.stem
    candidates = [
        dataset_root / "images" / split_name,
        dataset_root / split_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def signal_handler(signum, frame):
    """Handle the Ctrl+C signal."""
    global _interrupted
    if _interrupted:
        print("\n\n[!] force exit...")
        sys.exit(1)
    _interrupted = True
    print("\n\n[!] interrupt received (Ctrl+C), stopping...")
    print("    press Ctrl+C again to force exit")


def is_interrupted():
    """Return True if an interrupt was received."""
    return _interrupted


def get_magnification_from_filename(file_name: str) -> str:
    """
    Determine the image magnification from the file name.
    
    10x: file name starts with "10x/" or contains "_2048x2048.png"
    40x: file name starts with "40x/" or contains "_512x512.png"
    """
    if file_name.startswith("10x/") or "_2048x2048.png" in file_name:
        return "10x"
    elif file_name.startswith("40x/") or "_512x512.png" in file_name:
        return "40x"
    return "unknown"


def load_single_image(img_info: dict, img_root: str) -> dict:
    """Load a single image (used by the thread/process pre-loader)."""
    file_name = img_info["file_name"]
    img_path = Path(img_root) / file_name

    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    magnification = get_magnification_from_filename(file_name)
    return {
        "image_id": img_info["id"],
        "file_name": file_name,
        "img": img,
        "img_path": str(img_path),
        "magnification": magnification,
        "img_info": img_info,
    }


def build_ndarray_test_pipeline(model) -> Compose:
    """Build the NDArray test pipeline once and reuse it."""
    cfg = model.cfg.copy()
    test_pipeline_cfg = get_test_pipeline_cfg(cfg)
    if isinstance(test_pipeline_cfg, (list, tuple)) and len(test_pipeline_cfg) > 0:
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    return Compose(test_pipeline_cfg)


def inference_detector_true_batch(model,
                                  imgs,
                                  test_pipeline: Compose,
                                  use_amp: bool = False,
                                  device: str = 'cuda:0'):
    """Batched inference: one test_step call per batch of samples."""
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if len(imgs) == 0:
        return []

    batch_inputs = []
    batch_data_samples = []
    for img in imgs:
        data_ = dict(img=img, img_id=0)
        data_ = test_pipeline(data_)
        batch_inputs.append(data_['inputs'])
        batch_data_samples.append(data_['data_samples'])

    batch_data = {
        'inputs': batch_inputs,
        'data_samples': batch_data_samples,
    }

    amp_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.float16)
        if use_amp and str(device).startswith('cuda')
        else nullcontext()
    )
    with torch.no_grad():
        with amp_ctx:
            results = model.test_step(batch_data)

    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def mask_nms(masks: list,
             scores: list,
             iou_threshold: float = 0.15,
             contain_threshold: float = 1.01) -> list:
    """NMS based on mask IoU plus containment ratio."""
    if len(masks) == 0:
        return []
    
    order = np.argsort(scores)[::-1].tolist()
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        mask_i = masks[i]
        area_i = mask_i.sum()
        remaining = []
        
        for j in order[1:]:
            mask_j = masks[j]
            intersection = np.logical_and(mask_i, mask_j).sum()
            area_j = mask_j.sum()
            union = area_i + area_j - intersection
            iou = intersection / union if union > 0 else 0
            min_area = min(area_i, area_j)
            contain = intersection / min_area if min_area > 0 else 0

            if (iou < iou_threshold) and (contain < contain_threshold):
                remaining.append(j)
        
        order = remaining
    
    return keep


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert a binary mask to COCO RLE format.
    
    Args:
        mask: binary mask array, shape=(H, W), dtype=uint8
    
    Returns:
        RLE dict with 'size' and 'counts' fields
    """
    # fortran order for COCO RLE
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # counts: bytes -> ascii string
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def apply_mask_nms_per_category(predictions: list,
                                contain_threshold: float = None,
                                ) -> list:
    """Apply Mask NMS per category."""
    if not ENABLE_MASK_NMS or len(predictions) == 0:
        return predictions

    contain_thr = MASK_NMS_CONTAIN_THRES if contain_threshold is None else float(contain_threshold)
    
    by_category = {}
    for pred in predictions:
        cat_id = pred["category_id"]
        if cat_id not in by_category:
            by_category[cat_id] = []
        by_category[cat_id].append(pred)
    
    filtered = []
    for cat_id, preds in by_category.items():
        if len(preds) <= 1:
            filtered.extend(preds)
            continue
        
        label_4 = cat_id - 1
        nms_thres = MASK_NMS_THRES.get(label_4, 0.3)
        
        masks = []
        scores = []
        for pred in preds:
            rle = pred["segmentation"]
            mask = maskUtils.decode(rle)
            masks.append(mask)
            scores.append(pred["score"])
        
        keep_indices = mask_nms(
            masks,
            scores,
            iou_threshold=nms_thres,
            contain_threshold=contain_thr)
        
        for idx in keep_indices:
            filtered.append(preds[idx])
    
    return filtered


def process_single_result(result, image_id: int, img_info: dict, magnification: str,
                          mask_nms_contain_thres: float = 1.01,
                          ) -> list:
    """
    Post-process the inference result of one image (per-class score and area filters).
    """
    predictions = []
    
    # Allowed 6-class IDs for this magnification
    allowed_classes_6 = CLASSES_6_FOR_10X if magnification == "10x" else CLASSES_6_FOR_40X
    
    # Prediction instances
    pred_instances = result.pred_instances
    
    if len(pred_instances) == 0:
        return predictions
    
    # Prediction tensors
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') else None
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    if masks is None or len(masks) == 0:
        return predictions
    
    for i in range(len(scores)):
        score = float(scores[i])
        label_6 = int(labels[i])
        source_name = NAMES_6[label_6] if 0 <= label_6 < len(NAMES_6) else f"class_{label_6}"
        
        # 1. Per-class score threshold
        class_score_thres = PER_CLASS_SCORE_THRES.get(label_6, SCORE_THRES)
        if score < class_score_thres:
            continue
        
        # 2. Magnification-based class filter
        if label_6 not in allowed_classes_6:
            continue
        
        # 5. Mask
        mask = masks[i]
        mask_area = mask.sum()
        if mask_area == 0:
            continue

        # 6. Mask area filter
        min_area = MIN_MASK_AREA.get(label_6, 0)
        if mask_area < min_area:
            continue
        
        rle = mask_to_rle(mask)

        # 3. 6-class -> 4-class mapping
        label_4 = MAP_6to4_ID.get(label_6)
        if label_4 is None:
            continue

        # 4. 4-class index -> GT category_id
        gt_cat_id = PRED_TO_GT_CAT_ID.get(label_4)
        if gt_cat_id is None:
            continue

        predictions.append({
            "image_id": image_id,
            "category_id": gt_cat_id,
            "category_name": NAMES_4[label_4],
            "score": score,
            "segmentation": rle,
        })
    
    # 7. Per-category Mask NMS
    predictions = apply_mask_nms_per_category(
        predictions,
        contain_threshold=mask_nms_contain_thres)
    
    # 8. Sort by score and truncate
    predictions.sort(key=lambda x: x["score"], reverse=True)
    if len(predictions) > MAX_DETS:
        predictions = predictions[:MAX_DETS]
    
    return predictions


def preload_images(image_infos: list, img_root: Path, queue: Queue, num_workers: int = 4,
                  preload_mode: str = "thread"):
    """Pre-load images into a queue (thread/process)."""
    img_root_str = str(img_root)
    workers = max(1, int(num_workers))

    if preload_mode == "process":
        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)

    try:
        with executor:
            futures = {executor.submit(load_single_image, info, img_root_str): info for info in image_infos}
            for future in as_completed(futures):
                if is_interrupted():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    result = future.result()
                except Exception as e:
                    print(f"\n    [WARN] preload worker failed, sample skipped: {e}")
                    continue
                if result is not None:
                    queue.put(result)
    except Exception as e:
        print(f"\n    [WARN] preload thread failed: {e}")
    finally:
        # Always send the end sentinel so the consumer does not block
        try:
            queue.put(None, timeout=5.0)
        except Exception:
            pass


def run_inference(device: str = "cuda:0",
                  enable_mask_nms: bool = True, num_workers: int = 4,
                  preload_mode: str = "thread",
                  batch_size: int = 1,
                  use_amp: bool = False,
                  log_every: int = 200,
                  num_shards: int = 1,
                  shard_id: int = 0,
                  subset_image_list: Path = None,
                  subset_size: int = 0,
                  subset_ratio: float = 1.0,
                  subset_seed: int = 42,
                  subset_save_list: Path = None,
                  output_dir: Path = None,
                  export_objaware_map: bool = False,
                  objaware_map_dir: Path = None,
                  mask_nms_contain_thres: float = 0.85,
                  ):
    """
    Run inference.
    """
    global ENABLE_MASK_NMS, _interrupted, OUT_DIR
    ENABLE_MASK_NMS = enable_mask_nms
    _interrupted = False

    # objaware map output directory (created once OUT_DIR is resolved)
    map_dir = None
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 80)
    print(f"RTMDet-Ins-L inference")
    print("=" * 80)
    
    # Model config
    config_file = MODEL_CONFIG["config"]
    checkpoint_file = MODEL_CONFIG["checkpoint"]
    
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"GT JSON: {GT_JSON}")
    print(f"Image root: {IMG_ROOT}")
    print(f"Device: {device}")
    print(f"Mask NMS: {'on' if enable_mask_nms else 'off'}")
    print(f"Mask NMS contain thres: {mask_nms_contain_thres} (>=1 disables containment suppression)")
    print(f"Preload workers: {num_workers}")
    print(f"Preload mode: {preload_mode}")
    print(f"Batch size: {batch_size}")
    print(f"AMP: {'on' if use_amp else 'off'}")
    print(f"Log interval: every {log_every} images")
    print(f"Shard: {shard_id}/{num_shards}")
    print(f"Export ObjAware map: {'on' if export_objaware_map else 'off'}")
    if subset_image_list is not None:
        print(f"Subset list: {subset_image_list}")
    else:
        print(f"Subset sampling: size={subset_size}, ratio={subset_ratio}, seed={subset_seed}")

    amp_runtime_enabled = bool(use_amp and str(device).startswith('cuda'))
    
    # Check that the files exist
    if not Path(config_file).exists():
        raise FileNotFoundError(f"config file not found: {config_file}")
    if not Path(checkpoint_file).exists():
        # Look for an alternative checkpoint
        work_dir = Path(checkpoint_file).parent
        if work_dir.exists():
            checkpoints = list(work_dir.glob("*.pth"))
            if checkpoints:
                # Prefer best or latest
                for ckpt in checkpoints:
                    if "best" in ckpt.name or "latest" in ckpt.name:
                        checkpoint_file = str(ckpt)
                        print(f"    using fallback checkpoint: {checkpoint_file}")
                        break
                else:
                    checkpoint_file = str(checkpoints[-1])
                    print(f"    using fallback checkpoint: {checkpoint_file}")
            else:
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")
        else:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")

    # Default output directory: work_dirs/<config_stem>/<checkpoint_tag>/
    default_out_dir = resolve_default_output_dir(config_file, checkpoint_file)
    if output_dir is not None:
        OUT_DIR = Path(output_dir)
    else:
        OUT_DIR = default_out_dir

    # Use the derived output_name when set, otherwise fall back to config_stem.
    output_name = str(MODEL_CONFIG.get("output_name", "")).strip() or Path(config_file).stem
    print(f"Output dir: {OUT_DIR}")
    print(f"Output name: {output_name}")

    if export_objaware_map:
        map_dir = Path(objaware_map_dir) if objaware_map_dir is not None else OUT_DIR / "objaware_maps"
        map_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1] Loading model...")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    model = init_detector(config_file, checkpoint_file, device=device)
    batch_test_pipeline = build_ndarray_test_pipeline(model)
    print(f"    model loaded")
    
    # Load GT to get the image list
    print("\n[2] Loading test set info...")
    coco_gt = COCO(str(GT_JSON))
    images = coco_gt.dataset["images"]
    total_images = len(images)

    num_shards = max(1, int(num_shards))
    shard_id = int(shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}], got: {shard_id}")
    if num_shards > 1:
        images = images[shard_id::num_shards]

    # Subset selection
    subset_ids = None
    if subset_image_list is not None:
        subset_ids = load_subset_image_ids(Path(subset_image_list))
        before = len(images)
        images = [im for im in images if int(im["id"]) in subset_ids]
        print(f"    images after subset filter: {len(images)} / {before}")
    else:
        if int(subset_size) > 0 or float(subset_ratio) < 1.0:
            before = len(images)
            images = stratified_sample_images(
                images,
                subset_size=int(subset_size),
                subset_ratio=float(subset_ratio),
                subset_seed=int(subset_seed),
            )
            print(f"    images after sampling: {len(images)} / {before}")

    if len(images) == 0:
        raise RuntimeError("no images left after subset filtering; check the subset arguments")

    subset_ids = {int(im["id"]) for im in images}

    print(f"    test images: {len(images)} (total: {total_images})")
    
    # Magnification distribution
    count_10x = sum(1 for im in images if get_magnification_from_filename(im["file_name"]) == "10x")
    count_40x = sum(1 for im in images if get_magnification_from_filename(im["file_name"]) == "40x")
    print(f"    10x images: {count_10x}")
    print(f"    40x images: {count_40x}")
    
    # Ensure the output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_tag = output_name
    if num_shards > 1:
        output_tag = f"{output_name}_shard{shard_id}"
    output_path = OUT_DIR / f"{output_tag}_predictions.ndjson"

    # Save the image-ID subset used for this run
    dump_subset = subset_save_list
    if dump_subset is None and (subset_image_list is not None or int(subset_size) > 0 or float(subset_ratio) < 1.0):
        dump_subset = OUT_DIR / f"{output_tag}_subset_image_ids.json"
    if dump_subset is not None:
        dump_subset = Path(dump_subset)
        dump_subset.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_ids": sorted(subset_ids),
            "count": len(subset_ids),
            "num_shards": int(num_shards),
            "shard_id": int(shard_id),
            "subset_size": int(subset_size),
            "subset_ratio": float(subset_ratio),
            "subset_seed": int(subset_seed),
        }
        with open(dump_subset, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"    subset IDs saved: {dump_subset}")
    
    # Inference
    print(f"\n[3] Running inference ({num_workers} preload workers)...")
    print(f"    Output file: {output_path}")
    
    total_predictions = 0
    cat_counts = {}
    processed = 0
    processed_mag_counts = {"10x": 0, "40x": 0, "unknown": 0}
    pred_mag_counts = {"10x": 0, "40x": 0, "unknown": 0}
    batch_hist = {}
    total_batch_items = 0
    total_batch_iters = 0
    t_start = time.time()
    t_last_log = t_start
    processed_last_log = 0
    
    queue_size = max(int(num_workers) * 2, int(batch_size) * 2, 4)
    img_queue = Queue(maxsize=queue_size)
    
    preload_thread = Thread(
        target=preload_images,
        args=(images, IMG_ROOT, img_queue, num_workers, preload_mode)
    )
    preload_thread.start()
    
    interrupted_early = False
    
    with open(output_path, "wb") as fo:
        pbar = None
        pbar = tqdm(total=len(images), desc="inference", ncols=100)
        
        while True:
            if is_interrupted():
                interrupted_early = True
                if pbar is not None:
                    pbar.close()
                print("\n[!] interrupted, saving completed results...")
                break

            batch_items = []
            end_of_queue = False
            while len(batch_items) < max(1, int(batch_size)):
                try:
                    item = img_queue.get(timeout=1.0)
                except:
                    if is_interrupted():
                        interrupted_early = True
                        if pbar is not None:
                            pbar.close()
                        print("\n[!] interrupted, saving completed results...")
                        break
                    # Preloader exited and the queue is empty: treat as end of stream
                    if (not preload_thread.is_alive()) and img_queue.empty():
                        end_of_queue = True
                        break
                    continue

                if item is None:
                    end_of_queue = True
                    break
                batch_items.append(item)

            if interrupted_early:
                break

            if len(batch_items) == 0:
                if end_of_queue:
                    break
                continue

            processed_count = len(batch_items)
            total_batch_iters += 1
            total_batch_items += processed_count
            batch_hist[processed_count] = batch_hist.get(processed_count, 0) + 1
            if total_batch_iters <= 3:
                tqdm.write(
                    f"[BATCH] iter={total_batch_iters}, size={processed_count}, "
                    f"target_bs={batch_size}, amp_runtime={'on' if amp_runtime_enabled else 'off'}")
            for item in batch_items:
                magnification = item["magnification"]
                processed_mag_counts[magnification] = processed_mag_counts.get(magnification, 0) + 1

            img_ndarrays = [item["img"] for item in batch_items]
            results = None
            valid_items = []
            try:
                results = inference_detector_true_batch(
                    model,
                    img_ndarrays,
                    batch_test_pipeline,
                    use_amp=amp_runtime_enabled,
                    device=device,
                )
                valid_items = batch_items
            except Exception as e:
                err_msg = str(e)
                amp_dtype_err = (
                    amp_runtime_enabled
                    and (
                        "expected scalar type Float but found Half" in err_msg
                        or "found Half" in err_msg
                        or "not implemented for 'Half'" in err_msg
                        or "not implemented for \"Half\"" in err_msg
                    )
                )

                if amp_dtype_err:
                    print("\n    [WARN] AMP not supported by this model/op, falling back to FP32.")
                    amp_runtime_enabled = False
                    try:
                        results = inference_detector_true_batch(
                            model,
                            img_ndarrays,
                            batch_test_pipeline,
                            use_amp=False,
                            device=device,
                        )
                        valid_items = batch_items
                    except Exception as e_retry:
                        e = e_retry

                if is_interrupted():
                    interrupted_early = True
                    break
                if len(valid_items) == 0:
                    print(f"\n    [WARN] batch inference failed, falling back to per-image: {e}")
                    results = []
                    valid_items = []
                    for item in batch_items:
                        try:
                            res = inference_detector_true_batch(
                                model,
                                item["img"],
                                batch_test_pipeline,
                                use_amp=amp_runtime_enabled,
                                device=device,
                            )[0]
                            results.append(res)
                            valid_items.append(item)
                        except Exception as e2:
                            err2 = str(e2)
                            if amp_runtime_enabled and (
                                "expected scalar type Float but found Half" in err2
                                or "found Half" in err2
                                or "not implemented for 'Half'" in err2
                                or "not implemented for \"Half\"" in err2
                            ):
                                amp_runtime_enabled = False
                                try:
                                    res = inference_detector_true_batch(
                                        model,
                                        item["img"],
                                        batch_test_pipeline,
                                        use_amp=False,
                                        device=device,
                                    )[0]
                                    results.append(res)
                                    valid_items.append(item)
                                    continue
                                except Exception as e3:
                                    print(f"\n    [ERROR] inference failed {item['file_name']}: {e3}")
                            else:
                                print(f"\n    [ERROR] inference failed {item['file_name']}: {e2}")

            if len(results) != len(valid_items):
                min_len = min(len(results), len(valid_items))
                results = list(results)[:min_len]
                valid_items = list(valid_items)[:min_len]

            for result, item in zip(results, valid_items):
                image_id = item["image_id"]
                magnification = item["magnification"]
                img_info = item["img_info"]

                # Optionally export the objaware map (distance map)
                if export_objaware_map:
                    pred_instances = getattr(result, "pred_instances", None)
                    if pred_instances is not None:
                        obj_map = getattr(pred_instances, "objaware_map", None)
                        kind = getattr(pred_instances, "objaware_map_kind", None)
                        if obj_map is None and hasattr(pred_instances, "metainfo"):
                            obj_map = pred_instances.metainfo.get("objaware_map", None)
                            kind = pred_instances.metainfo.get("objaware_map_kind", kind)
                    if obj_map is not None:
                        obj_map = obj_map.detach().cpu().numpy()
                        kind = kind or "objaware"
                        safe_name = item["file_name"].replace("/", "_").replace("\\", "_")
                        map_path = map_dir / f"{safe_name}_objaware_{kind}.npy"
                        np.save(map_path, obj_map)

                preds = process_single_result(
                    result,
                    image_id,
                    img_info,
                    magnification,
                    mask_nms_contain_thres=mask_nms_contain_thres,
                )

                for pred in preds:
                    fo.write(json_dumps_bytes(pred))
                    fo.write(b"\n")
                    total_predictions += 1
                    cat_id = int(pred["category_id"])
                    cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
                    pred_mag_counts[magnification] = pred_mag_counts.get(magnification, 0) + 1

            for item in batch_items:
                item["img"] = None

            processed += processed_count
            if pbar is not None:
                pbar.update(processed_count)

            if log_every > 0 and (processed - processed_last_log >= log_every):
                now = time.time()
                dt = max(now - t_last_log, 1e-6)
                total_dt = max(now - t_start, 1e-6)
                inst_ips = (processed - processed_last_log) / dt
                avg_ips = processed / total_dt
                remain = max(len(images) - processed, 0)
                eta_sec = remain / max(avg_ips, 1e-6)
                avg_batch_now = total_batch_items / max(total_batch_iters, 1)
                try:
                    qsize = img_queue.qsize()
                except Exception:
                    qsize = -1

                tqdm.write(
                    f"[LOG] {processed}/{len(images)} "
                    f"({processed / max(len(images), 1) * 100:.1f}%) | "
                    f"inst={inst_ips:.2f} img/s, avg={avg_ips:.2f} img/s | "
                    f"avg_batch={avg_batch_now:.2f}, q={qsize} | "
                    f"ETA={eta_sec/60:.1f} min")

                t_last_log = now
                processed_last_log = processed
        
        if not interrupted_early:
            if pbar is not None:
                pbar.close()
    
    preload_thread.join(timeout=2.0)
    
    if interrupted_early:
        print(f"\n[!] Inference interrupted!")
        print(f"    images processed: {processed}")
        print(f"    predictions saved: {total_predictions}")
        print(f"    processed images by magnification: 10x={processed_mag_counts.get('10x', 0)}, 40x={processed_mag_counts.get('40x', 0)}, unknown={processed_mag_counts.get('unknown', 0)}")
        print(f"    saved predictions by magnification: 10x={pred_mag_counts.get('10x', 0)}, 40x={pred_mag_counts.get('40x', 0)}, unknown={pred_mag_counts.get('unknown', 0)}")
        print(f"    Output file: {output_path}")
        return output_path
    
    print(f"\n[OK] Inference done!")
    print(f"    Output file: {output_path}")
    print(f"    total predictions: {total_predictions}")
    if total_batch_iters > 0:
        avg_batch = total_batch_items / total_batch_iters
        max_batch = max(batch_hist.keys())
        min_batch = min(batch_hist.keys())
        print(f"    batch stats: iters={total_batch_iters}, avg={avg_batch:.2f}, min={min_batch}, max={max_batch}")
        hist_str = ", ".join([f"{k}:{v}" for k, v in sorted(batch_hist.items())])
        print(f"    batch size histogram: {hist_str}")
    print(f"    processed images by magnification: 10x={processed_mag_counts.get('10x', 0)}, 40x={processed_mag_counts.get('40x', 0)}, unknown={processed_mag_counts.get('unknown', 0)}")
    print(f"    saved predictions by magnification: 10x={pred_mag_counts.get('10x', 0)}, 40x={pred_mag_counts.get('40x', 0)}, unknown={pred_mag_counts.get('unknown', 0)}")

    if processed_mag_counts.get('40x', 0) > 0 and pred_mag_counts.get('40x', 0) == 0:
        print("\n    [WARN] 40x images were processed but produced no 40x predictions.")
        print("           Check the checkpoint and whether inference was interrupted.")
    
    print("\n    Predictions per category:")
    for cat_id, count in sorted(cat_counts.items()):
        cat_name = NAMES_4[cat_id - 1] if 1 <= cat_id <= len(NAMES_4) else f"unknown_{cat_id}"
        print(f"      {cat_id}: {cat_name}: {count}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="RTMDet-Ins-L instance segmentation inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # standard run
    python inference.py --config CFG --checkpoint CKPT --gt-json GT.json
    
    # parallel image preloading (4-8 workers)
    python inference.py ... --workers 4
    
    # faster: disable Mask NMS
    python inference.py ... --workers 8 --no-mask-nms
    
    # select a device
    python inference.py ... --device DEVICE
        """
    )
    parser.add_argument("--config", type=str, default=None,
                        help="model config file path (required)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="model checkpoint path (required)")
    parser.add_argument("--gt-json", type=str, default=None,
                        help="test-set GT JSON path, used for the image list (required)")
    parser.add_argument("--img-root", type=str, default=None,
                        help="test-set image root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory (overrides the default)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="device, e.g. cuda:0 (default: cuda:0)")
    parser.add_argument("--no-mask-nms", action="store_true",
                        help="disable Mask NMS post-processing")
    parser.add_argument("--mask-nms-contain-thres", type=float, default=1.01,
                        help="Mask NMS containment threshold inter/min(area_i,area_j); >=1 disables it")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of CPU preload workers (default: 4)")
    parser.add_argument("--preload-mode", type=str, default="thread",
                        choices=["thread", "process"],
                        help="preload parallelism: thread or process (default: thread)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="inference batch size (default: 1)")
    parser.add_argument("--amp", action="store_true",
                        help="enable AMP mixed-precision inference (CUDA)")
    parser.add_argument("--log-every", type=int, default=200,
                        help="log every N images (default: 200); <=0 disables")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total number of shards (default: 1)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="shard index (0-based, default: 0)")
    parser.add_argument("--subset-image-list", type=str, default=None,
                        help="subset image-ID list JSON (list[int] or {image_ids:[...] })")
    parser.add_argument("--subset-size", type=int, default=0,
                        help="subset sample size (0 disables)")
    parser.add_argument("--subset-ratio", type=float, default=1.0,
                        help="sampling ratio in (0,1]; used only when subset-size=0")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="subset sampling random seed")
    parser.add_argument("--subset-save-list", type=str, default=None,
                        help="save the subset IDs used for this run to a JSON file")
    parser.add_argument("--export-objaware-map", action="store_true",
                        help="export the objaware map (distance map) as .npy")
    parser.add_argument("--objaware-map-dir", type=str, default=None,
                        help="objaware map output directory (default: <output-dir>/objaware_maps)")
    
    args = parser.parse_args()

    # Test-set paths
    global GT_JSON, IMG_ROOT
    if args.gt_json is not None:
        GT_JSON = Path(args.gt_json)
    if GT_JSON is None:
        parser.error("--gt-json is required")
    if args.img_root is not None:
        IMG_ROOT = Path(args.img_root)
    else:
        IMG_ROOT = infer_default_image_root(GT_JSON)
    
    # Command-line config/checkpoint override the defaults
    if args.config is not None:
        MODEL_CONFIG["config"] = args.config
        # Output files are named after the config file
        output_name = Path(args.config).stem
        MODEL_CONFIG["output_name"] = output_name
        print(f"[INFO] output name derived from config: {output_name}")

    if args.checkpoint is not None:
        MODEL_CONFIG["checkpoint"] = args.checkpoint

    if not MODEL_CONFIG.get("config"):
        parser.error("--config is required")
    if not MODEL_CONFIG.get("checkpoint"):
        parser.error("--checkpoint is required")

    run_inference(args.device,
                  enable_mask_nms=not args.no_mask_nms,
                  mask_nms_contain_thres=args.mask_nms_contain_thres,
                  num_workers=args.workers,
                  preload_mode=args.preload_mode,
                  batch_size=args.batch_size,
                  use_amp=args.amp,
                  log_every=args.log_every,
                  num_shards=args.num_shards,
                  shard_id=args.shard_id,
                  subset_image_list=Path(args.subset_image_list) if args.subset_image_list else None,
                  subset_size=args.subset_size,
                  subset_ratio=args.subset_ratio,
                  subset_seed=args.subset_seed,
                  subset_save_list=Path(args.subset_save_list) if args.subset_save_list else None,
                  output_dir=args.output_dir,
                  export_objaware_map=args.export_objaware_map,
                  objaware_map_dir=Path(args.objaware_map_dir) if args.objaware_map_dir else None,
                  )


if __name__ == "__main__":
    main()
