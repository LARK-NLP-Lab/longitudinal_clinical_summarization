#!/usr/bin/env bash

# Usage: ./run.sh <task>
# task: ds | ap | rag_ds | rag_ap

TASK=$1

if [[ -z "$TASK" ]]; then
  echo "Usage: $0 <task>"
  echo "  task: ds | ap | rag_ds | rag_ap"
  exit 1
fi

case "$TASK" in
  ds)
    base_dir="results/directgen_discharge_sum"
    mkdir -p "$base_dir"
    python ds_gen.py --outputdir "$base_dir"
    ;;
  
  ap)
    base_dir="results/directgen_assessment_plan"
    mkdir -p "$base_dir/method-1" "$base_dir/method1" "$base_dir/method2"
    python ap_gen.py --method -1 --outputdir "$base_dir/method-1"
    python ap_gen.py --method 1  --outputdir "$base_dir/method1"
    python ap_gen.py --method 2  --outputdir "$base_dir/method2"
    ;;
  
  rag_ds)
    base_dir="results/rag_discharge_sum"
    mkdir -p "$base_dir"
    python RAG_script_ds.py --outputdir "$base_dir"
    ;;
  
  rag_ap)
    base_dir="results/rag_assessment_plan"
    mkdir -p "$base_dir/method-1" "$base_dir/method1" "$base_dir/method2"
    python RAG_script_ap.py --method -1 --outputdir "$base_dir/method-1"
    python RAG_script_ap.py --method 1  --outputdir "$base_dir/method1"
    python RAG_script_ap.py --method 2  --outputdir "$base_dir/method2"
    ;;
  
  *)
    echo "Error: Unknown task '$TASK'. Valid options: ds, ap, rag_ds, rag_ap"
    exit 1
    ;;
esac
