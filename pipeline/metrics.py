import os
import json
import time
import argparse
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score
import difflib
import re
from dataclasses import dataclass
from datetime import datetime
import torch
import psutil
import GPUtil

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    timestamp: str
    total_samples: int
    format_metrics: Dict
    content_metrics: Dict
    performance_metrics: Dict
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "format_metrics": self.format_metrics,
            "content_metrics": self.content_metrics,
            "performance_metrics": self.performance_metrics
        }

def validate_xml_format(text: str) -> Dict:
    """Validate XML format of the response."""
    format_metrics = {
        "has_reasoning_tag": False,
        "has_answer_tag": False,
        "proper_nesting": False,
        "format_score": 0.0
    }
    
    # Check for required tags
    format_metrics["has_reasoning_tag"] = "<reasoning>" in text and "</reasoning>" in text
    format_metrics["has_answer_tag"] = "<answer>" in text and "</answer>" in text
    
    # Check proper nesting
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    format_metrics["proper_nesting"] = bool(re.search(pattern, text, re.DOTALL))
    
    # Calculate format score
    format_metrics["format_score"] = (
        format_metrics["has_reasoning_tag"] +
        format_metrics["has_answer_tag"] +
        format_metrics["proper_nesting"]
    ) / 3.0
    
    return format_metrics

def calculate_content_metrics(generated: str, reference: str) -> Dict:
    """Calculate content quality metrics."""
    content_metrics = {
        "similarity_score": 0.0,
        "length_ratio": 0.0,
        "reasoning_present": False
    }
    
    # Calculate similarity
    content_metrics["similarity_score"] = difflib.SequenceMatcher(None, generated, reference).ratio()
    
    # Calculate length ratio (normalized)
    max_len = max(len(generated), len(reference))
    min_len = min(len(generated), len(reference))
    content_metrics["length_ratio"] = min_len / max_len if max_len > 0 else 0.0
    
    # Check for reasoning
    content_metrics["reasoning_present"] = bool(re.search(r"<reasoning>.*?</reasoning>", generated, re.DOTALL))
    
    return content_metrics

def measure_performance(func):
    """Decorator to measure performance metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            start_gpu = GPUtil.getGPUs()[0].memoryUsed
            torch.cuda.reset_peak_memory_stats()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        performance_metrics = {
            "execution_time": end_time - start_time,
            "memory_usage_mb": end_memory - start_memory,
        }
        
        if torch.cuda.is_available():
            end_gpu = GPUtil.getGPUs()[0].memoryUsed
            performance_metrics["gpu_memory_mb"] = end_gpu - start_gpu
            performance_metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return result, performance_metrics
    
    return wrapper

@measure_performance
def run_model_inference(model, input_text: str) -> str:
    """Run model inference with performance tracking."""
    # This is a placeholder - actual implementation would use the model
    return "Generated response"

def calculate_metrics(
    results_dir: str,
    model_name: str,
    reference_outputs: Optional[Dict] = None
) -> ModelMetrics:
    """Calculate comprehensive metrics for model outputs."""
    
    # Load test results
    results_file = os.path.join(results_dir, "test_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize metric accumulators
    format_metrics_total = {
        "format_score": 0.0,
        "proper_format_rate": 0.0
    }
    
    content_metrics_total = {
        "average_similarity": 0.0,
        "average_length_ratio": 0.0,
        "reasoning_rate": 0.0
    }
    
    performance_metrics_total = {
        "average_generation_time": 0.0,
        "average_memory_usage_mb": 0.0,
        "average_gpu_memory_mb": 0.0 if torch.cuda.is_available() else None
    }
    
    total_samples = len(results)
    
    # Process each result
    for result in results:
        # Format validation
        format_metrics = validate_xml_format(result["generated_output"])
        format_metrics_total["format_score"] += format_metrics["format_score"]
        format_metrics_total["proper_format_rate"] += float(format_metrics["proper_nesting"])
        
        # Content quality
        if reference_outputs and result["id"] in reference_outputs:
            content_metrics = calculate_content_metrics(
                result["generated_output"],
                reference_outputs[result["id"]]
            )
            content_metrics_total["average_similarity"] += content_metrics["similarity_score"]
            content_metrics_total["average_length_ratio"] += content_metrics["length_ratio"]
            content_metrics_total["reasoning_rate"] += float(content_metrics["reasoning_present"])
        
        # Performance metrics
        if "performance" in result:
            perf = result["performance"]
            performance_metrics_total["average_generation_time"] += perf["execution_time"]
            performance_metrics_total["average_memory_usage_mb"] += perf["memory_usage_mb"]
            if torch.cuda.is_available() and "gpu_memory_mb" in perf:
                performance_metrics_total["average_gpu_memory_mb"] += perf["gpu_memory_mb"]
    
    # Calculate averages
    for metric in format_metrics_total:
        format_metrics_total[metric] /= total_samples
    
    if reference_outputs:
        for metric in content_metrics_total:
            content_metrics_total[metric] /= total_samples
    
    for metric in performance_metrics_total:
        if performance_metrics_total[metric] is not None:
            performance_metrics_total[metric] /= total_samples
    
    return ModelMetrics(
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        total_samples=total_samples,
        format_metrics=format_metrics_total,
        content_metrics=content_metrics_total,
        performance_metrics=performance_metrics_total
    )

def compare_models(metrics_list: List[ModelMetrics]) -> Dict:
    """Compare metrics across different models."""
    comparison = {
        "models": [],
        "format_comparison": {},
        "content_comparison": {},
        "performance_comparison": {}
    }
    
    for metrics in metrics_list:
        model_data = metrics.to_dict()
        comparison["models"].append(model_data["model_name"])
        
        # Format metrics comparison
        for metric, value in model_data["format_metrics"].items():
            if metric not in comparison["format_comparison"]:
                comparison["format_comparison"][metric] = {}
            comparison["format_comparison"][metric][model_data["model_name"]] = value
        
        # Content metrics comparison
        for metric, value in model_data["content_metrics"].items():
            if metric not in comparison["content_comparison"]:
                comparison["content_comparison"][metric] = {}
            comparison["content_comparison"][metric][model_data["model_name"]] = value
        
        # Performance metrics comparison
        for metric, value in model_data["performance_metrics"].items():
            if metric not in comparison["performance_comparison"]:
                comparison["performance_comparison"][metric] = {}
            comparison["performance_comparison"][metric][model_data["model_name"]] = value
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Calculate and compare model metrics')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--model-name', required=True, help='Name of the model being evaluated')
    parser.add_argument('--reference-file', help='Path to reference outputs JSON')
    parser.add_argument('--output-file', required=True, help='Path to save metrics JSON')
    
    args = parser.parse_args()
    
    # Load reference outputs if provided
    reference_outputs = None
    if args.reference_file and os.path.exists(args.reference_file):
        with open(args.reference_file, 'r') as f:
            reference_outputs = json.load(f)
    
    # Calculate metrics
    metrics = calculate_metrics(
        args.results_dir,
        args.model_name,
        reference_outputs
    )
    
    # Save metrics
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"Metrics saved to {args.output_file}")
    print("\nSummary:")
    print(f"Model: {metrics.model_name}")
    print(f"Total samples: {metrics.total_samples}")
    print(f"Format score: {metrics.format_metrics['format_score']:.2%}")
    if reference_outputs:
        print(f"Average similarity: {metrics.content_metrics['average_similarity']:.2%}")
    print(f"Average generation time: {metrics.performance_metrics['average_generation_time']:.2f}s")
    if metrics.performance_metrics.get('average_gpu_memory_mb'):
        print(f"Average GPU memory usage: {metrics.performance_metrics['average_gpu_memory_mb']:.2f}MB")

if __name__ == "__main__":
    main() 