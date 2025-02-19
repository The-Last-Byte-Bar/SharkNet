import torch
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
from dataclasses import dataclass
from pipeline import eval_config, monitoring
from pipeline.metrics import calculate_script_similarity
from pipeline.data_loader import create_dataloaders, ErgoDataset
from pipeline.model import create_model
import mlflow
from mlflow.tracking import MlflowClient
from tqdm import tqdm
from datetime import datetime

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    sample_id: int
    input_text: str
    generated_text: str
    reference_text: Optional[str]
    metrics: Dict[str, float]
    generation_time: float
    memory_usage: float

class ModelEvaluator:
    def __init__(self, model_config: Union[Dict, str], test_cases_path=None, output_dir=None, run_name=None, is_dry_run=False):
        """Initialize the evaluation pipeline.
        
        Args:
            model_config: Either a dictionary containing model configuration or a model name string
            test_cases_path: Path to test cases directory (optional)
            output_dir: Path to output directory (optional)
            run_name: Name for this evaluation run (optional)
            is_dry_run: Whether this is a dry run (optional)
        """
        if isinstance(model_config, dict):
            self.model_config = model_config
            self.model_name = model_config['name']
            self.is_dry_run = is_dry_run
        else:
            self.model_name = model_config
            self.model_config = {'name': model_config}
            self.is_dry_run = is_dry_run
            
        self.test_cases_path = test_cases_path
        self.output_dir = output_dir
        self.run_name = run_name or f"eval_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up MLflow tracking
        tracking_uri = f"file://{os.path.abspath(eval_config.MLFLOW_TRACKING_URI)}"
        self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        
        # Create or get experiment
        experiment = self.mlflow_client.get_experiment_by_name(eval_config.EXPERIMENT_NAME)
        if experiment is None:
            self.experiment_id = self.mlflow_client.create_experiment(eval_config.EXPERIMENT_NAME)
        else:
            self.experiment_id = experiment.experiment_id
        
        # Start a new run
        self.run = self.mlflow_client.create_run(self.experiment_id, run_name=self.run_name)
        self.run_id = self.run.info.run_id
        
        # Load model
        self.model, self.tokenizer = self._load_model()
        
        # Setup output directory
        self.output_dir = os.path.join(
            eval_config.EVAL_OUTPUT_DIR,
            self.run_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_model(self) -> Tuple[torch.nn.Module, Any]:
        """Load the model and tokenizer."""
        model, tokenizer = create_model()
        return model, tokenizer
        
    def evaluate_sample(self, sample: Dict) -> EvaluationResult:
        """Evaluate a single sample."""
        start_time = time.time()
        
        # Prepare input
        inputs = self.tokenizer(
            sample["text"],
            padding=True,
            truncation=True,
            max_length=eval_config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(eval_config.DEVICE)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=eval_config.MAX_SEQ_LENGTH,
                num_return_sequences=1
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "script_similarity": calculate_script_similarity(
                generated_text,
                sample["response"]
            ) if "response" in sample else 0.0,
            "response_time": generation_time
        }
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            memory_usage = 0
            
        return EvaluationResult(
            model_name=self.model_name,
            sample_id=sample.get("id", 0),
            input_text=sample["text"],
            generated_text=generated_text,
            reference_text=sample.get("response"),
            metrics=metrics,
            generation_time=generation_time,
            memory_usage=memory_usage
        )
        
    def run_evaluation(self, eval_dataset: ErgoDataset) -> List[EvaluationResult]:
        """Run evaluation on the dataset."""
        results = []
        self.monitor.start_monitoring()
        
        try:
            # Log model configuration
            for key, value in self.model_config.items():
                mlflow.log_param(self.run_id, key, value)
                
            # Evaluate samples
            for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
                if self.is_dry_run and idx >= eval_config.DRY_RUN_SAMPLES:
                    break
                    
                result = self.evaluate_sample(sample)
                results.append(result)
                
                # Log metrics
                self.monitor.log_batch_metrics(
                    {
                        **result.metrics,
                        "memory_usage": result.memory_usage
                    },
                    idx
                )
                
            # Calculate and log aggregate metrics
            agg_metrics = self._calculate_aggregate_metrics(results)
            mlflow.log_metrics(self.run_id, agg_metrics)
            
            # Save detailed results
            self._save_results(results)
            
        finally:
            self.monitor.close()
            mlflow.end_run()
            
        return results
        
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        metrics = {}
        
        # Calculate averages for all numeric metrics
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
                
        return {
            f"avg_{name}": sum(values) / len(values)
            for name, values in metrics.items()
        }
        
    def _save_results(self, results: List[EvaluationResult]):
        """Save evaluation results to file."""
        output_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(
                [
                    {
                        "model_name": r.model_name,
                        "sample_id": r.sample_id,
                        "input_text": r.input_text,
                        "generated_text": r.generated_text,
                        "reference_text": r.reference_text,
                        "metrics": r.metrics,
                        "generation_time": r.generation_time,
                        "memory_usage": r.memory_usage
                    }
                    for r in results
                ],
                f,
                indent=2
            )
            
def run_evaluation_pipeline(
    model_config: Union[Dict, str],
    test_cases_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    is_dry_run: bool = False
) -> List[EvaluationResult]:
    """Run the evaluation pipeline.
    
    Args:
        model_config: Either a dictionary containing model configuration or a model name string
        test_cases_path: Path to test cases directory (optional)
        output_dir: Path to output directory (optional)
        run_name: Name for this evaluation run (optional)
        is_dry_run: Whether this is a dry run (default: False)
        
    Returns:
        List of evaluation results
    """
    print(f"\nEvaluating model: {model_config['name'] if isinstance(model_config, dict) else model_config}")
    
    evaluator = ModelEvaluator(
        model_config=model_config,
        test_cases_path=test_cases_path,
        output_dir=output_dir,
        run_name=run_name,
        is_dry_run=is_dry_run
    )
    
    # Create evaluation dataloader using the data_loader module
    from pipeline.data_loader import create_dataloaders
    _, eval_loader = create_dataloaders()
    
    return evaluator.run_evaluation(eval_loader.dataset)

def _compare_models(results_by_model: Dict[str, List[EvaluationResult]]):
    """Compare results across different models."""
    comparison_file = os.path.join(eval_config.EVAL_OUTPUT_DIR, "model_comparison.json")
    
    comparison = {}
    for model_name, results in results_by_model.items():
        agg_metrics = ModelEvaluator._calculate_aggregate_metrics(results)
        comparison[model_name] = agg_metrics
        
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
        
    print("\nModel Comparison:")
    for model_name, metrics in comparison.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run with limited samples')
    args = parser.parse_args()
    
    run_evaluation_pipeline(args.dry_run) 