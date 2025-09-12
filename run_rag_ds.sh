#!/usr/bin/env bash
base_dir="results/rag_discharge_sum"

mkdir -p "$base_dir"

python RAG_script.py --outputdir "$base_dir"
