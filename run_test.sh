#!/bin/bash
# run_test_all_models.sh - Test fine-tuning for all 3 models (3 runs × 10 epochs each)

echo "======================================================================"
echo "STARTING TEST: All 3 models × 3 runs × 10 epochs with per-epoch logging"
echo "======================================================================"

# Configuration
MODELS=("runs/roberta-base/best" "runs/microsoft__deberta-v3-large/best" "runs/answerdotai__ModernBERT-large/best")
MODEL_NAMES=("roberta-base" "microsoft__deberta-v3-large" "answerdotai__ModernBERT-large")
DATA_FILE="CR_ECSS_dataset.json"
EPOCHS=10
BATCH_SIZE=8
OUTPUT_DIR="results_5epoch_test"
NUM_RUNS=3

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Models: ${MODEL_NAMES[@]}"
echo "  Data: $DATA_FILE"
echo "  Epochs per run: $EPOCHS"
echo "  Number of runs per model: $NUM_RUNS"
echo "  Output: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Loop through each model
for i in {0..2}; do
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    
    echo ""
    echo "======================================================================"
    echo "MODEL: $MODEL_NAME"
    echo "======================================================================"
    echo "Path: $MODEL_PATH"
    echo ""
    
    # Run 3 experiments for this model
    for run in {1..3}; do
        SEED=$((42 + run - 1))
        
        echo "----------------------------------------------------------------------"
        echo "Run $run/$NUM_RUNS | Seed: $SEED | Model: $MODEL_NAME"
        echo "----------------------------------------------------------------------"
        
        python fine_tune_with_summary.py \
            --model-path $MODEL_PATH \
            --data-file $DATA_FILE \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --seed $SEED \
            --run-number $run \
            --output-dir $OUTPUT_DIR
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ SUCCESS: Run $run completed for $MODEL_NAME"
        else
            echo "✗ ERROR: Run $run failed for $MODEL_NAME (exit code: $EXIT_CODE)"
        fi
        
        echo ""
    done
    
    echo "======================================================================"
    echo "COMPLETED ALL RUNS FOR: $MODEL_NAME"
    echo "======================================================================"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""

# Check if per-epoch logs were created for all models
echo "Checking per-epoch logs:"
echo "----------------------------------------------------------------------"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    EPOCH_LOG="$OUTPUT_DIR/${MODEL_NAME}_per_epoch_logs.csv"
    if [ -f "$EPOCH_LOG" ]; then
        NUM_ROWS=$(wc -l < "$EPOCH_LOG")
        echo "✓ $MODEL_NAME: $((NUM_ROWS - 1)) rows (should be $((NUM_RUNS * EPOCHS)) = 30)"
    else
        echo "✗ $MODEL_NAME: Log not found"
    fi
done

echo ""
echo "======================================================================"
echo "Next step: Generate plot"
echo "======================================================================"
echo "Run: python plot_finetuning_validation_loss.py"
echo ""
