# SharkNet LLM Training & Evaluation Pipeline

A comprehensive pipeline for training, evaluating, and comparing LLM models.

## Setup

### Environment Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate pool
```

## Usage

### End-to-End Training and Evaluation

Run the complete pipeline (training + evaluation):
```bash
./run_e2e.sh --mode [standard|grpo]
```

Options:
- `--config`: Path to config file (default: config/evaluation_config.yml)
- `--mode`: Training mode (standard or grpo)
- `--skip-training`: Skip training and run evaluation only
- `--eval-only`: Run evaluation on existing models

### Evaluation Only

To evaluate and compare existing models:
```bash
./run_evaluation.sh
```

Options:
- `--config`: Path to config file (default: config/evaluation_config.yml)

## Directory Structure

```
.
├── pipeline/           # Core pipeline code
├── config/            # Configuration files
├── checkpoints/       # Model checkpoints
├── artifacts/         # Training and evaluation artifacts
│   └── latest/       # Symlink to latest run
└── logs/             # Log files
```

## Configuration

The pipeline is configured through YAML files in the `config/` directory:

- `evaluation_config.yml`: Controls evaluation metrics, model paths, and output formats

## Outputs

### Training Artifacts

- Model checkpoints saved in `checkpoints/`
- Training logs and metrics in `artifacts/run_TIMESTAMP/`

### Evaluation Results

- Individual model metrics
- Model comparison report (markdown format)
- Performance benchmarks
- Detailed logs

## Metrics

The evaluation system tracks:

1. Format Validation
   - XML tag presence and structure
   - Proper nesting
   - Format score

2. Content Quality
   - Response similarity
   - Length ratio
   - Reasoning presence

3. Performance
   - Generation time
   - Memory usage
   - GPU utilization
