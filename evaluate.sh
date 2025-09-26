#!/usr/bin/env bash

# Usage: ./run_eval.sh <task> <dataset>
# Example: ./run_eval.sh ds mimic

TASK=$1
DATASET=$2

if [[ -z "$TASK" || -z "$DATASET" ]]; then
  exit 1
fi

if [[ "$DATASET" == "ehrshot" ]]; then
  python evaluation/evaluate_ehrshot.py --task "$TASK"
elif [[ "$DATASET" == "mimic" ]]; then
  case "$TASK" in
    ds)
      python evaluation/evaluate_ds.py
      ;;
    ap)
      python evaluation/evaluate_ap.py
      ;;
    *)
      exit 1
      ;;
  esac
else
  exit 1
fi
