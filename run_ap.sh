#!/usr/bin/env bash
base_dir="results/directgen_assessment_plan"

mkdir -p "$base_dir/method-1"
mkdir -p "$base_dir/method1"
mkdir -p "$base_dir/method2"

python args_pn_llm.py --method -1 --outputdir "$base_dir/method-1"
echo "-----------------------method -1 done -------------------------------"

python args_pn_llm.py --method 1 --outputdir "$base_dir/method1"
echo "-----------------------method 1 done --------------------------------"

python args_pn_llm.py --method 2 --outputdir "$base_dir/method2"
echo "-----------------------method 2 done --------------------------------"

# move generated folder to google storage
gsutil -m cp -r "$base_dir" gs://lark_projects/workspace_maya/patient_summarization/data/pn_generation/generated/ 