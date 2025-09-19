#!/usr/bin/env bash
base_dir="results/directgen_discharge_sum"

mkdir -p "$base_dir"

python ds_gen.py --outputdir "$base_dir"
