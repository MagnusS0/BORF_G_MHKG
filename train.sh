#!/bin/bash

# Datasets to test
datasets=("cora")

# Rewiring methods
rewiring_methods=("none" "borf")

# Layer types
layer_types=("GIN" "G_MHKG")

# Common parameters
NUM_TRIALS=10
NUM_SPLITS=3
HIDDEN_DIM=64
NUM_LAYERS=3
DROPOUT=0.5
NUM_ITERATIONS=12
BATCH_SIZE=3
BORF_BATCH_ADD=20
BORF_BATCH_REMOVE=10
DATASET=Cora
DEVICE=cuda:0


# Create results directory if not exists
mkdir -p results

# Log file
LOG_FILE="results/training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting training runs at $(date)" | tee -a "$LOG_FILE"

for dataset in "${datasets[@]}"; do
    for rewiring in "${rewiring_methods[@]}"; do
        for layer in "${layer_types[@]}"; do
            echo "Running with dataset=$dataset, rewiring=$rewiring, layer=$layer" | tee -a "$LOG_FILE"
            
            # Base command
            CMD="python run_node_classification.py \
                --dataset $dataset \
                --rewiring $rewiring \
                --layer_type $layer \
                --num_trials $NUM_TRIALS \
                --num_splits $NUM_SPLITS \
                --hidden_dim $HIDDEN_DIM \
                --num_layers $NUM_LAYERS \
                --num_iterations $NUM_ITERATIONS \
                --batch_size $BATCH_SIZE \
                --borf_batch_add $BORF_BATCH_ADD \
                --borf_batch_remove $BORF_BATCH_REMOVE \
                --device $DEVICE \
                --dropout $DROPOUT"
            
            # Run the command and log output
            echo "Command: $CMD" | tee -a "$LOG_FILE"
            $CMD 2>&1 | tee -a "$LOG_FILE"
            
            # Add separator in log
            echo "----------------------------------------" | tee -a "$LOG_FILE"
        done
    done
done

echo "All training runs completed at $(date)" | tee -a "$LOG_FILE"