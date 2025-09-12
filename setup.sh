#!/usr/bin/env bash

python get_target_population.py --sample_size 2000

mkdir -p data/discharge_sum/input
mkdir -p data/discharge_sum/gold

python get_chronologies_DS.py

mkdir -p data/assessment_plan/input
mkdir -p data/assessment_plan/gold

python get_chronologies_AP.py
