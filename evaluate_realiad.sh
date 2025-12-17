#!/bin/bash
# Real-IAD Evaluation Script
# Computes I-AUROC and P-AUROC metrics between predicted and ground truth datasets

# Default paths
GT_ROOT="./data/Real-IAD/realiad_variety_subset_extracted"
PRED_ROOT="./data/Real-IAD/realiad_variety_subset_predict_zero_shot"
OUTPUT_DIR="./evaluation_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gt_root)
            GT_ROOT="$2"
            shift 2
            ;;
        --pred_root)
            PRED_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Real-IAD Evaluation"
echo "=========================================="
echo "Ground truth root: ${GT_ROOT}"
echo "Prediction root: ${PRED_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run evaluation
python evaluate_realiad.py \
    --gt_root ${GT_ROOT} \
    --pred_root ${PRED_ROOT} \
    --output_file ${OUTPUT_DIR}/evaluation_results.json

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}/evaluation_results.json"
echo "=========================================="
