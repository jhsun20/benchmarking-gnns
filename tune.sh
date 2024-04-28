#!/bin/bash

models=("GatedGCN" "GCN" "GIN" "GMM" "GraphSage")
for model in "${models[@]}"; do
    python tune.py -m "$model" -d MC -l --device 2
done