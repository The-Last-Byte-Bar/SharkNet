import os
import json
import argparse
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score
import difflib

def calculate_script_similarity(generated: str, reference: str) -> float:
    """Calculate similarity between generated and reference scripts."""
    return difflib.SequenceMatcher(None, generated, reference).ratio()

def calculate_metrics(results_dir: str) -> Dict:
    """Calculate metrics for the model outputs."""
    metrics = {
        "script_similarities": [],
        "execution_success_rate": 0,
        "average_generation_time": 0.0,
        "total_samples": 0
    }
    
    # Load test results
    results_file = os.path.join(results_dir, "test_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_time = 0
    successful_executions = 0
    
    for result in results:
        # Calculate script similarity if reference exists
        if "reference_script" in result and "generated_script" in result:
            similarity = calculate_script_similarity(
                result["generated_script"],
                result["reference_script"]
            )
            metrics["script_similarities"].append(similarity)
        
        # Track execution success
        if result.get("execution_success", False):
            successful_executions += 1
            
        # Track generation time
        if "generation_time" in result:
            total_time += result["generation_time"]
    
    total_samples = len(results)
    metrics["total_samples"] = total_samples
    
    if total_samples > 0:
        metrics["execution_success_rate"] = successful_executions / total_samples
        metrics["average_generation_time"] = total_time / total_samples
        
    if metrics["script_similarities"]:
        metrics["average_similarity"] = np.mean(metrics["script_similarities"])
        metrics["similarity_std"] = np.std(metrics["script_similarities"])
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for model outputs')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--output-file', required=True, help='Path to save metrics JSON')
    
    args = parser.parse_args()
    
    # Calculate metrics
    metrics = calculate_metrics(args.results_dir)
    
    # Save metrics
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {args.output_file}")
    print("\nSummary:")
    print(f"Total samples: {metrics['total_samples']}")
    if 'average_similarity' in metrics:
        print(f"Average script similarity: {metrics['average_similarity']:.2%}")
    print(f"Execution success rate: {metrics['execution_success_rate']:.2%}")
    print(f"Average generation time: {metrics['average_generation_time']:.2f}s")

if __name__ == "__main__":
    main() 