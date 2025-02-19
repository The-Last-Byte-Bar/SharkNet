import os
import json
import shutil
from typing import Dict, List
import torch
from pipeline import eval_config
from pipeline.test_cases import TestCaseManager, TestCase
from pipeline.evaluator import run_evaluation_pipeline
from pipeline import monitoring
import mlflow
from datetime import datetime
import logging
import argparse

# Set up logging
logger = logging.getLogger(__name__)

class DryRunManager:
    """Manages the dry run process for testing the pipeline."""
    
    def __init__(self, cleanup=False, debug=False):
        """Initialize the dry run manager.
        
        Args:
            cleanup: Whether to clean up artifacts after completion
            debug: Whether to enable debug logging
        """
        self.should_cleanup = cleanup
        self.debug = debug
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"dry_run_{self.timestamp}"
        self.run_dir = self.run_name
        
        # Set up logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
            
        logger.debug("Initializing DryRunManager")
        
        # Create directories
        logger.debug(f"Setting up directories in {self.run_dir}")
        os.makedirs(os.path.join(self.run_dir, "test_cases"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "model_outputs"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        
        logger.debug(f"Created directory: {self.run_dir}/test_cases")
        logger.debug(f"Created directory: {self.run_dir}/model_outputs")
        logger.debug(f"Created directory: {self.run_dir}/metrics")
        logger.debug(f"Created directory: {self.run_dir}/logs")
        
        self.test_case_manager = TestCaseManager()
        self.model_config = None
        
    def setup_directories(self):
        """Set up directories for dry run."""
        logger.debug(f"Setting up directories in {self.run_dir}")
        # Create dry run directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectories
        dirs = [
            "test_cases",
            "model_outputs",
            "metrics",
            "logs"
        ]
        for dir_name in dirs:
            dir_path = os.path.join(self.run_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
            
    def create_sample_test_cases(self):
        """Create sample test cases for dry run."""
        logger.debug("Creating sample test cases")
        sample_cases = [
            {
                "id": "test_001",
                "input_text": "Write a simple ErgoScript contract that implements a basic token.",
                "expected_output": None,
                "category": "smart_contracts",
                "difficulty": "easy",
                "tags": ["token", "basic"]
            },
            {
                "id": "test_002",
                "input_text": "Explain how to handle errors in ErgoScript.",
                "expected_output": None,
                "category": "error_handling",
                "difficulty": "medium",
                "tags": ["explanation", "errors"]
            }
        ]
        
        for case in sample_cases:
            try:
                logger.debug(f"Creating test case: {case['id']}")
                test_case = TestCase(**case)
                self.test_case_manager.save_test_case(test_case, is_predefined=True)
            except Exception as e:
                logger.error(f"Error creating test case {case['id']}: {str(e)}", exc_info=True)
                raise
            
    def setup_mock_model(self):
        """Set up a mock model for dry run."""
        logger.debug("Setting up mock model")
        self.model_config = {
            "name": "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
            "type": "mock",
            "parameters": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        config_file = os.path.join(self.run_dir, "mock_model_config.json")
        try:
            with open(config_file, 'w') as f:
                json.dump(self.model_config, f, indent=4)
            logger.debug(f"Saved mock model config to {config_file}")
        except Exception as e:
            logger.error(f"Error saving mock model config: {str(e)}")
            raise
            
    def cleanup_artifacts(self):
        """Clean up dry run artifacts."""
        logger.debug(f"Cleaning up dry run directory: {self.run_dir}")
        if os.path.exists(self.run_dir):
            try:
                shutil.rmtree(self.run_dir)
                logger.info(f"Cleaned up dry run directory: {self.run_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up dry run directory {self.run_dir}: {str(e)}")
        else:
            logger.warning(f"Dry run directory {self.run_dir} not found. Skipping cleanup.")
            
    def run(self):
        """Run the dry run process."""
        try:
            logger.info("Starting dry run...")
            
            # Initialize metrics monitor
            logger.debug("Initializing metrics monitor")
            self.monitor = monitoring.MetricsMonitor(self.run_name, debug=True)
            self.monitor.start_monitoring()
            
            # Create sample test cases
            logger.info("Creating sample test cases...")
            self.create_sample_test_cases()
            
            # Set up mock model
            logger.info("Setting up mock model...")
            self.setup_mock_model()
            
            # Run evaluation pipeline
            logger.info("Running evaluation pipeline...")
            run_evaluation_pipeline(
                model_config=self.model_config,
                test_cases_path=os.path.join(self.run_dir, "test_cases"),
                output_dir=os.path.join(self.run_dir, "model_outputs"),
                is_dry_run=True
            )
            
            logger.info("Dry run completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during dry run: {str(e)}", exc_info=True)
            return False
            
        finally:
            logger.debug("Cleaning up monitor")
            if hasattr(self, 'monitor'):
                self.monitor.close()
            if self.should_cleanup:
                self.cleanup_artifacts()
            
    def verify_outputs(self):
        """Verify that all expected outputs were created."""
        logger.debug("Starting output verification")
        expected_files = [
            os.path.join(eval_config.EVAL_OUTPUT_DIR, "model_comparison.json"),
            os.path.join(eval_config.TENSORBOARD_LOG_DIR),
            os.path.join(eval_config.MLFLOW_TRACKING_URI)
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                logger.warning(f"Missing expected output: {file_path}")
                missing_files.append(file_path)
                
        if missing_files:
            logger.warning("The following expected outputs are missing:")
            for file_path in missing_files:
                logger.warning(f"  - {file_path}")
        else:
            logger.info("All expected outputs were created successfully.")
            
def main():
    parser = argparse.ArgumentParser(description='Run a dry run of the evaluation pipeline')
    parser.add_argument('--cleanup', action='store_true', help='Clean up dry run artifacts after completion')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dry_run_debug.log')
        ]
    )
    
    dry_run = DryRunManager(cleanup=args.cleanup, debug=args.debug)
    try:
        dry_run.run()
        logger.info("Dry run completed successfully!")
    except Exception as e:
        logger.error("Dry run failed!", exc_info=True)
        raise
    finally:
        if args.cleanup:
            dry_run.cleanup_artifacts()
            
if __name__ == "__main__":
    main() 