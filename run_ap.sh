#!/usr/bin/env bash
base_dir="results/directgen_assessment_plan"

mkdir -p "$base_dir/method-1"
mkdir -p "$base_dir/method1"
mkdir -p "$base_dir/method2"

python ap_gen.py --method -1 --outputdir "$base_dir/method-1"
echo "-----------------------method -1 done -------------------------------"

python ap_gen.py --method 1 --outputdir "$base_dir/method1"
echo "-----------------------method 1 done --------------------------------"

python ap_gen.py --method 2 --outputdir "$base_dir/method2"
echo "-----------------------method 2 done --------------------------------"