#!/usr/bin/env bash

python get_target_population.py --sample_size 2000

mkdir -p data/DS/input
mkdir -p data/DS/gold

python get_chronologies_DS.py

mkdir -p data/AP/input
mkdir -p data/AP/gold

python get_chronologies_AP.py
