#!/usr/bin/env bash
set -u

BIN=""
M=1
N=2048
K=2048
GROUP=128
WARMUP=10
ITERS=100
OUT=""
SKIP_CUDA=0

usage() {
  cat <<USAGE
Usage: $0 --bin <test_fpA_intB_gemm> [--m N] [--n N] [--k N] [--group N] [--warmup N] [--iters N] [--out FILE] [--skip-cuda]

Example:
  $0 --bin ./build_sm80_ptx_w4a16/test_fpA_intB_gemm --m 1 --n 2048 --k 2048 --group 128 --out results.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --m) M="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --group|--group_size) GROUP="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --skip-cuda) SKIP_CUDA=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
 done

if [[ -z "$BIN" ]]; then
  echo "--bin is required" >&2
  usage
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  : > "$OUT"
fi

map_tile_enum() {
  case "$1" in
    3) echo "16,128,64";;
    4) echo "32,128,64";;
    7) echo "64,128,64";;
    11) echo "128,128,64";;
    15) echo "16,256,64";;
    *) return 1;;
  esac
}

configs=$($BIN --list_configs | sed -n 's/^ *[0-9]\+: //p')

fail=0
idx=0
while IFS= read -r cfg; do
  if [[ -z "$cfg" ]]; then
    continue
  fi

  if [[ "$cfg" == "cuda_kernel" ]]; then
    if [[ "$SKIP_CUDA" -eq 1 ]]; then
      continue
    fi
    config_arg="cuda"
  else
    if [[ "$cfg" =~ tile_enum=([0-9]+)[[:space:]]+stages=([0-9]+)[[:space:]]+split_k=([0-9]+) ]]; then
      tile_enum="${BASH_REMATCH[1]}"
      stages="${BASH_REMATCH[2]}"
      split_k="${BASH_REMATCH[3]}"
      tile_shape=$(map_tile_enum "$tile_enum") || { echo "Unknown tile_enum=$tile_enum" >&2; fail=$((fail+1)); continue; }
      config_arg="${tile_shape},${stages},${split_k}"
    else
      echo "Unrecognized config line: $cfg" >&2
      fail=$((fail+1))
      continue
    fi
  fi

  cmd=("$BIN" "--m=$M" "--n=$N" "--k=$K" "--group_size=$GROUP" "--warmup=$WARMUP" "--iters=$ITERS" "--config=$config_arg")
  echo "[${idx}] ${cmd[*]}"
  if [[ -n "$OUT" ]]; then
    {
      echo "[${idx}] ${cmd[*]}"
      "${cmd[@]}"
      echo
    } >> "$OUT" 2>&1 || fail=$((fail+1))
  else
    "${cmd[@]}" || fail=$((fail+1))
  fi

  idx=$((idx+1))
 done <<< "$configs"

if [[ $fail -ne 0 ]]; then
  echo "Done with $fail failures." >&2
  exit 1
fi

echo "Done."
