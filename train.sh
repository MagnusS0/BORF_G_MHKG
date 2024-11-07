#!/bin/bash

# Datasets to test
datasets=("cora" "citeseer")

# Rewiring methods
rewiring_methods=("none" "borf")

# Layer types
layer_types=("G_MHKG" "GCN" "GIN")

# Number of layers
num_layers_list=(2 3 5 7 9)

# Common parameters
NUM_TRIALS=10
NUM_SPLITS=3
HIDDEN_DIM=64
DROPOUT=0.3
NUM_ITERATIONS=3
BATCH_SIZE=32
BORF_BATCH_ADD=20
BORF_BATCH_REMOVE=10
DEVICE=cuda:0

# Optimal settings for different datasets
declare -A optimal_settings
optimal_settings["cora_GCN"]="3 20 10"
optimal_settings["cora_GIN"]="3 20 30"
optimal_settings["cora_G_MHKG"]="3 20 10"
optimal_settings["citeseer_GCN"]="3 20 10"
optimal_settings["citeseer_GIN"]="3 10 20"
optimal_settings["citeseer_G_MHKG"]="3 10 20"
optimal_settings["texas_GCN"]="3 30 10"
optimal_settings["texas_GIN"]="1 20 10"
optimal_settings["texas_G_MHKG"]="1 20 10"
optimal_settings["cornell_GCN"]="2 20 30"
optimal_settings["cornell_GIN"]="3 10 20"
optimal_settings["cornell_G_MHKG"]="3 10 20"
optimal_settings["wisconsin_GCN"]="2 30 20"
optimal_settings["wisconsin_GIN"]="2 50 30"
optimal_settings["wisconsin_G_MHKG"]="2 50 30"
optimal_settings["chameleon_GCN"]="3 20 20"
optimal_settings["chameleon_GIN"]="3 30 30"
optimal_settings["chameleon_G_MHKG"]="3 30 30"

# Create results directory if not exists
mkdir -p results

# Log file
LOG_FILE="results/training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting training runs at $(date)" | tee -a "$LOG_FILE"

for dataset in "${datasets[@]}"; do
    for rewiring in "${rewiring_methods[@]}"; do
        for layer in "${layer_types[@]}"; do
            for num_layers in "${num_layers_list[@]}"; do
                echo "Running with dataset=$dataset, rewiring=$rewiring, layer=$layer" | tee -a "$LOG_FILE"
                
                # Apply optimal settings for the current layer type
                key="${dataset}_${layer}"
                if [[ -n "${optimal_settings[$key]}" ]]; then
                    IFS=' ' read -r NUM_ITERATIONS BORF_BATCH_ADD BORF_BATCH_REMOVE <<< "${optimal_settings[$key]}"
                fi

                # Base command for the current layer type
                CMD="python run_node_classification.py \
                    --dataset $dataset \
                    --rewiring $rewiring \
                    --layer_type $layer \
                    --num_trials $NUM_TRIALS \
                    --num_splits $NUM_SPLITS \
                    --hidden_dim $HIDDEN_DIM \
                    --dropout $DROPOUT \
                    --num_iterations $NUM_ITERATIONS \
                    --batch_size $BATCH_SIZE \
                    --borf_batch_add $BORF_BATCH_ADD \
                    --borf_batch_remove $BORF_BATCH_REMOVE \
                    --device $DEVICE \
                    --num_layers $num_layers"

                echo "Executing: $CMD" | tee -a "$LOG_FILE"
                eval $CMD | tee -a "$LOG_FILE"
            done
        done
    done
done

echo "Training runs completed at $(date)" | tee -a "$LOG_FILE"