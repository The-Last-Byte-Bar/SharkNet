#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pool

# Default values
CONFIG_PATH="config/evaluation_config.yml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run evaluation only mode
CMD="python run_pipeline.py --config ${CONFIG_PATH} --eval-only"

# Run the evaluation
echo "Running evaluation command: ${CMD}"
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    
    # Open the latest evaluation report
    if [ -f "artifacts/latest/evaluation_report.md" ]; then
        echo "Opening evaluation report..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "artifacts/latest/evaluation_report.md"
        else
            echo "Evaluation report available at: artifacts/latest/evaluation_report.md"
        fi
    fi
else
    echo "Evaluation failed!"
    exit 1
fi 