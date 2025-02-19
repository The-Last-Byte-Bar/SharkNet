#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pool

# Default port
PORT=7860

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the web UI
echo "Starting web UI on port ${PORT}..."
python -m webui.app --port ${PORT}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Web UI started successfully!"
else
    echo "Failed to start Web UI!"
    exit 1
fi 