#!/usr/bin/env python3
import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
import json
import shutil
import torch

from pipeline.main import main as train_main
from pipeline.metrics import calculate_metrics, compare_models, ModelMetrics
from pipeline.data_loader import create_dataloaders
from pipeline.model import create_model
from pipeline.trainer import evaluate_model
from pipeline.model import FastLanguageModel

def setup_logging(config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["file"]),
            logging.StreamHandler() if config["logging"]["console"] else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(config: dict) -> None:
    """Create necessary directories."""
    dirs = [
        config["output"]["results_dir"],
        "checkpoints",
        "artifacts",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_training(args, config: dict, logger: logging.Logger) -> str:
    """Run the training pipeline and return the output directory."""
    logger.info("Starting training pipeline...")
    
    # Create unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("artifacts", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Run training
    train_args = argparse.Namespace(
        mode=args.training_mode,
        output_dir=run_dir,
        config=args.config
    )
    
    try:
        train_main(train_args)
        logger.info(f"Training completed successfully. Artifacts saved to {run_dir}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    return run_dir

def run_evaluation(run_dir: str, config: dict, logger: logging.Logger) -> dict:
    """Run model evaluation and return metrics."""
    logger.info("Starting model evaluation...")
    
    results = {}
    
    # Get all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir("checkpoints"):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join("checkpoints", item)):
            checkpoint_dirs.append(item)
    
    checkpoint_dirs.sort()  # Sort to evaluate in order
    
    for checkpoint in checkpoint_dirs:
        model_name = checkpoint
        model_path = os.path.join("checkpoints", checkpoint)
        
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Create test dataloader
            _, _, test_loader = create_dataloaders()
            
            # Load model and tokenizer directly from checkpoint
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=512,  # Use fixed value instead of config
                load_in_4bit=True,
                device_map="auto",
                max_memory={0: "16GiB"},
            )
            
            # Prepare model for inference
            model = FastLanguageModel.for_inference(model)
            
            # Run evaluation
            test_results = evaluate_model(model, tokenizer, test_loader, run_dir)
            
            # Save test results for this checkpoint
            test_results_path = os.path.join(run_dir, f"{model_name}_test_results.json")
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            # Calculate metrics
            metrics = calculate_metrics(
                results_dir=run_dir,
                model_name=model_name,
                reference_outputs=None  # We don't have reference outputs yet
            )
            results[model_name] = metrics
            
            # Save individual model metrics
            metrics_path = os.path.join(run_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
                
            logger.info(f"Saved metrics for {model_name} to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {str(e)}")
            continue
    
    return results

def generate_comparison_report(results: dict, run_dir: str, config: dict, logger: logging.Logger) -> None:
    """Generate comparison report for all evaluated models."""
    logger.info("Generating comparison report...")
    
    try:
        # Compare models
        comparison = compare_models(list(results.values()))
        
        # Save comparison results
        comparison_path = os.path.join(run_dir, "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate markdown report if configured
        if config["output"]["report_format"] == "markdown":
            report_path = os.path.join(run_dir, "evaluation_report.md")
            with open(report_path, 'w') as f:
                f.write("# Model Evaluation Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Model Summaries
                f.write("## Model Summaries\n\n")
                for model_name, metrics in results.items():
                    f.write(f"### {model_name}\n\n")
                    f.write(f"- Total samples: {metrics.total_samples}\n")
                    f.write(f"- Format score: {metrics.format_metrics['format_score']:.2%}\n")
                    f.write(f"- Average generation time: {metrics.performance_metrics['average_generation_time']:.2f}s\n")
                    if metrics.performance_metrics.get('average_gpu_memory_mb'):
                        f.write(f"- GPU memory usage: {metrics.performance_metrics['average_gpu_memory_mb']:.2f}MB\n")
                    f.write("\n")
                
                # Comparative Analysis
                f.write("## Comparative Analysis\n\n")
                for category, metrics in comparison.items():
                    if category != "models":
                        f.write(f"### {category.replace('_', ' ').title()}\n\n")
                        f.write("| Metric | " + " | ".join(comparison["models"]) + " |\n")
                        f.write("|--------|" + "|".join(["-" * len(model) for model in comparison["models"]]) + "|\n")
                        
                        for metric, values in metrics.items():
                            row = f"| {metric} |"
                            for model in comparison["models"]:
                                value = values.get(model, "N/A")
                                row += f" {value:.2f} |" if isinstance(value, float) else f" {value} |"
                            f.write(row + "\n")
                        f.write("\n")
        
        logger.info(f"Comparison report saved to {run_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate comparison report: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="End-to-end training and evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--training-mode", default="standard", choices=["standard", "grpo"],
                      help="Training mode to use")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip training and run evaluation only")
    parser.add_argument("--eval-only", action="store_true",
                      help="Run evaluation only on existing models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    try:
        # Create necessary directories
        setup_directories(config)
        
        # Run pipeline
        if not args.eval_only:
            if not args.skip_training:
                run_dir = run_training(args, config, logger)
            else:
                run_dir = os.path.join("artifacts", "latest")
                if not os.path.exists(run_dir):
                    logger.error("No existing run directory found for evaluation")
                    sys.exit(1)
        else:
            # For eval-only mode, create a new run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join("artifacts", f"eval_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Update latest symlink
            latest_link = os.path.join("artifacts", "latest")
            if os.path.lexists(latest_link):  # Use lexists to check for broken symlinks
                os.remove(latest_link)
            try:
                os.symlink(os.path.basename(run_dir), latest_link)  # Use relative path
            except Exception as e:
                logger.warning(f"Failed to create symlink: {str(e)}")
                # Continue anyway as this is not critical
        
        # Run evaluation
        results = run_evaluation(run_dir, config, logger)
        
        # Generate comparison report
        generate_comparison_report(results, run_dir, config, logger)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 