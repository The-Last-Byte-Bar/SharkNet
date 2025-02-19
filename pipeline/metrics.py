import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import difflib
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QualityMetrics:
    """Container for quality-related metrics."""
    format_compliance: float  # How well the output follows expected format
    reasoning_quality: float  # Quality of reasoning (if applicable)
    code_quality: float  # Quality of generated code
    response_completeness: float  # Whether all parts of the query were addressed

def calculate_script_similarity(generated: str, reference: str) -> float:
    """Calculate similarity between generated and reference scripts."""
    return difflib.SequenceMatcher(None, generated, reference).ratio()

def assess_format_compliance(text: str, expected_format: Optional[str] = None) -> float:
    """Assess how well the output follows expected format."""
    if expected_format == "xml":
        # Check for XML-style format
        has_reasoning = bool(re.search(r"<reasoning>.*</reasoning>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*</answer>", text, re.DOTALL))
        proper_nesting = text.count("<reasoning>") == text.count("</reasoning>") == 1 and \
                        text.count("<answer>") == text.count("</answer>") == 1
        return sum([has_reasoning, has_answer, proper_nesting]) / 3
    return 1.0  # Default to perfect score if no format specified

def assess_reasoning_quality(text: str) -> float:
    """Assess the quality of reasoning in the response."""
    # Extract reasoning section if present
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not reasoning_match:
        return 0.0
        
    reasoning = reasoning_match.group(1).strip()
    
    # Quality indicators
    has_multiple_steps = len(re.findall(r"\n\s*[-•*]\s+", reasoning)) > 1
    has_explanations = "because" in reasoning.lower() or "since" in reasoning.lower()
    has_structure = bool(re.search(r"(?:first|second|third|finally|lastly)", reasoning.lower()))
    
    return sum([
        has_multiple_steps,
        has_explanations,
        has_structure
    ]) / 3

def assess_code_quality(code: str) -> float:
    """Assess the quality of generated code."""
    # Basic code quality indicators
    has_comments = bool(re.search(r"(?://|#|/\*|\*).*\w+", code))
    has_error_handling = "try" in code and "catch" in code
    has_functions = bool(re.search(r"(?:function|def)\s+\w+\s*\(", code))
    proper_indentation = all(line.startswith((" " * 4 * i for i in range(10))) 
                           for line in code.splitlines() if line.strip())
    
    return sum([
        has_comments,
        has_error_handling,
        has_functions,
        proper_indentation
    ]) / 4

def assess_response_completeness(response: str, query: str) -> float:
    """Assess whether all parts of the query were addressed."""
    # Extract key elements from query
    query_elements = set(re.findall(r'\b\w+(?:\s+\w+){0,2}\b', query.lower()))
    response_elements = set(re.findall(r'\b\w+(?:\s+\w+){0,2}\b', response.lower()))
    
    # Calculate overlap
    if not query_elements:
        return 1.0
    return len(query_elements.intersection(response_elements)) / len(query_elements)

def calculate_quality_metrics(text: str, query: str = "", 
                           expected_format: Optional[str] = None) -> QualityMetrics:
    """Calculate comprehensive quality metrics for a response."""
    return QualityMetrics(
        format_compliance=assess_format_compliance(text, expected_format),
        reasoning_quality=assess_reasoning_quality(text),
        code_quality=assess_code_quality(text),
        response_completeness=assess_response_completeness(text, query)
    )

def calculate_metrics(results_dir: str) -> Dict:
    """Calculate metrics for the model outputs."""
    metrics = {
        "script_similarities": [],
        "execution_success_rate": 0,
        "average_generation_time": 0.0,
        "total_samples": 0,
        "quality_metrics": {
            "format_compliance": [],
            "reasoning_quality": [],
            "code_quality": [],
            "response_completeness": []
        }
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
        
        # Calculate quality metrics
        quality = calculate_quality_metrics(
            result.get("generated_text", ""),
            result.get("input_text", ""),
            result.get("expected_format")
        )
        
        metrics["quality_metrics"]["format_compliance"].append(quality.format_compliance)
        metrics["quality_metrics"]["reasoning_quality"].append(quality.reasoning_quality)
        metrics["quality_metrics"]["code_quality"].append(quality.code_quality)
        metrics["quality_metrics"]["response_completeness"].append(quality.response_completeness)
        
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
        
        # Calculate average quality metrics
        for metric_name in metrics["quality_metrics"]:
            values = metrics["quality_metrics"][metric_name]
            if values:
                metrics[f"average_{metric_name}"] = np.mean(values)
                metrics[f"{metric_name}_std"] = np.std(values)
        
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
    
    # Print averages for all available metrics
    for key, value in metrics.items():
        if key.startswith("average_"):
            print(f"{key}: {value:.2%}")
    
    print(f"Execution success rate: {metrics['execution_success_rate']:.2%}")
    print(f"Average generation time: {metrics['average_generation_time']:.2f}s")

if __name__ == "__main__":
    main() 