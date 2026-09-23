#!/usr/bin/env bash
# 4-class mapping evaluation (kpmp_test_core4): sharded inference -> merge -> evaluate.
#
# Usage (run from anywhere; relative paths resolve against the repository root):
#   bash scripts/eval_core4.sh [--checkpoint CKPT] [--config CFG] [--out-dir DIR] \
#        [--devices cuda:0,cuda:0,cuda:0,cuda:1] [--amp] [--contain-thres VAL]
#
#   --checkpoint    default checkpoint/Mori_seg.pth
#   --config        default configs/stage1_objaware_boundary_distexp3_100e.py
#   --out-dir       default work_dirs/eval_core4/<checkpoint name>
#   --devices       one shard process per entry, a GPU may repeat; ~2.1GB VRAM each
#   --amp           enable mixed-precision inference (off by default)
#   --contain-thres Mask NMS containment-dedup threshold; 1.01 disables the rule
#
# Required environment variable:
#   KPMP_TEST_ROOT  test set root, containing annotations/{test,test_instance}.json and images/test
# Optional:
#   PY              directory holding the python interpreter to use
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CKPT="checkpoint/Mori_seg.pth"
CFG="configs/stage1_objaware_boundary_distexp3_100e.py"
OUT=""
DEVICES="cuda:0"
AMP=""
CONTAIN_THRES="1.01"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --config) CFG="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --amp) AMP="--amp"; shift ;;
        --no-amp) AMP=""; shift ;;
        --contain-thres) CONTAIN_THRES="$2"; shift 2 ;;
        -h|--help) sed -n 2,19p "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Relative paths resolve against the repository root
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }
CKPT=$(abspath "$CKPT")
CFG=$(abspath "$CFG")
[[ -f "$CKPT" ]] || { echo "[ERROR] checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "[ERROR] config not found: $CFG" >&2; exit 1; }

CKPT_STEM=$(basename "$CKPT" .pth)
OUT=$(abspath "${OUT:-work_dirs/eval_core4/$CKPT_STEM}")
# The inference script names its outputs after the config file
NAME=$(basename "$CFG" .py | sed -E 's/_ki_split_[0-9]+$//')

[[ -n "${KPMP_TEST_ROOT:-}" ]] || { echo "[ERROR] KPMP_TEST_ROOT is not set (test set root)" >&2; exit 1; }
INFER_PY="$ROOT/mori_seg/eval/inference_core4.py"
GT_TEST="$KPMP_TEST_ROOT/annotations/test.json"
GT_INSTANCE="$KPMP_TEST_ROOT/annotations/test_instance.json"
IMG_ROOT="$KPMP_TEST_ROOT/images/test"
for f in "$INFER_PY" "$ROOT/mori_seg/eval/eval_ndjson_gpu.py" "$GT_TEST" "$GT_INSTANCE" "$IMG_ROOT"; do
    [[ -e "$f" ]] || { echo "[ERROR] missing dependency: $f" >&2; exit 1; }
done

[[ -n "${PY:-}" ]] && export PATH="$PY:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# The config uses custom_imports=['mori_seg'] and the evaluator is called as a
# package module, so the repository root must be on PYTHONPATH
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a DEVS <<< "$DEVICES"
N=${#DEVS[@]}
mkdir -p "$OUT/logs"
echo "[INFO] root=$ROOT"
echo "[INFO] ckpt=$CKPT"
echo "[INFO] config=$CFG"
echo "[INFO] out=$OUT  shards=$N  devices=$DEVICES  contain-thres=$CONTAIN_THRES  amp=${AMP:-off}"

cd "$ROOT"
pids=()
for i in $(seq 0 $((N-1))); do
    shard_args=()
    [[ $N -gt 1 ]] && shard_args=(--num-shards "$N" --shard-id "$i")
    python3 "$INFER_PY" --device "${DEVS[$i]}" --workers 4 --preload-mode thread \
        --batch-size 1 --log-every 200 $AMP "${shard_args[@]}" \
        ${CONTAIN_THRES:+--mask-nms-contain-thres $CONTAIN_THRES} \
        --output-dir "$OUT" --config "$CFG" --checkpoint "$CKPT" \
        --gt-json "$GT_TEST" --img-root "$IMG_ROOT" \
        > "$OUT/logs/infer_shard$i.log" 2>&1 &
    pids+=($!)
    sleep 20
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { echo "[ERROR] a shard failed, see $OUT/logs/infer_shard*.log" >&2; exit 1; }

PRED="$OUT/${NAME}_merge_v2.ndjson"
if [[ $N -gt 1 ]]; then
    cat $(for i in $(seq 0 $((N-1))); do echo "$OUT/${NAME}_shard${i}_merge_v2.ndjson"; done) > "$PRED"
fi
echo "[INFO] predictions: $(wc -l < "$PRED") lines"

python3 -m mori_seg.eval.eval_ndjson_gpu \
    --pred "$PRED" --final-pred "$PRED" \
    --model-name "$NAME" --output-dir "$OUT" \
    --gt-json "$GT_INSTANCE" --category-space kpmp_test_core4 \
    --device "${DEVS[0]}" --config "$CFG" --checkpoint "$CKPT" \
    2>&1 | tee "$OUT/logs/eval.log"
echo "[DONE] results: $OUT/eval_results_${NAME}.json"
