import time
import psutil
import torch
from typing import Dict, Any, Optional
import logging
from torch.utils.tensorboard import SummaryWriter
import threading
from dataclasses import dataclass, asdict
import numpy as np
from pipeline import eval_config
import GPUtil
import os
import mlflow
try:
    from mlflow import MlflowClient
except ImportError as e:
    try:
         from mlflow.tracking import MlflowClient
    except ImportError as e2:
         print("ERROR: mlflow package does not have MlflowClient. Please install or update mlflow with: pip install mlflow")
         raise e2

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_debug.log')
    ]
)

logger = logging.getLogger(__name__)

# Try to import MLflow, but make it optional
try:
    MLFLOW_AVAILABLE = True
    logger.info("MLflow is available and will be used for tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow is not available. Metrics will only be logged to TensorBoard")

# Added debug logging block for mlflow module version if DEBUG mode is enabled
if os.getenv('DEBUG', 'false').lower() in ['1', 'true']:
    try:
        import mlflow
        print("DEBUG: mlflow package version:", mlflow.__version__)
    except Exception as ex:
        print("DEBUG: Failed to retrieve mlflow version:", ex)

@dataclass
class ResourceMetrics:
    """Container for resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]
    timestamp: float

class MetricsMonitor:
    def __init__(self, run_name: str, debug: bool = True):
        logger.debug(f"Initializing MetricsMonitor with run_name: {run_name}")
        self.run_name = run_name
        self.debug = debug
        self.tensorboard = None
        self.monitoring = False
        self.metrics_buffer = []
        self.mlflow_active = False
        self.mlflow_client = None
        self.experiment_id = None
        self.run_id = None
        
        try:
            logger.debug("Setting up TensorBoard...")
            os.makedirs(eval_config.TENSORBOARD_LOG_DIR, exist_ok=True)
            self.tensorboard = SummaryWriter(f"{eval_config.TENSORBOARD_LOG_DIR}/{run_name}")
            logger.debug("TensorBoard setup complete")
        except Exception as e:
            logger.error(f"Error setting up TensorBoard: {str(e)}")
            raise
            
        if MLFLOW_AVAILABLE:
            try:
                logger.debug("Setting up MLflow...")
                self._setup_mlflow()
                self.mlflow_active = True
                logger.debug("MLflow setup complete")
            except Exception as e:
                logger.error(f"Error setting up MLflow: {str(e)}")
                logger.warning("Continuing without MLflow tracking")
                self.mlflow_active = False
        
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            # Create MLflow directory if it doesn't exist
            os.makedirs(eval_config.MLFLOW_TRACKING_URI, exist_ok=True)
            
            # Set up MLflow client
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
            self.mlflow_active = True
            logger.info("MLflow tracking initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking: {str(e)}")
            self.mlflow_active = False
        
    def start_monitoring(self):
        """Start the monitoring thread."""
        logger.debug("Starting monitoring thread...")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        logger.debug("Monitoring thread started")
        
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        logger.debug("Stopping monitoring thread...")
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logger.debug("Monitoring thread stopped")
            
    def _monitor_resources(self):
        """Continuously monitor system resources."""
        logger.debug("Resource monitoring started")
        while self.monitoring:
            try:
                metrics = self._collect_resource_metrics()
                self._log_metrics(metrics)
                time.sleep(eval_config.MONITORING_INTERVAL)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
            
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource usage metrics."""
        gpu_util = None
        gpu_memory = None
        
        if eval_config.MONITOR_GPU and torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
                gpu_util = gpu.load * 100
                gpu_memory = gpu.memoryUsed
                logger.debug(f"GPU metrics collected - Utilization: {gpu_util:.2f}%, Memory: {gpu_memory:.2f}MB")
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {str(e)}")
                
        metrics = ResourceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_memory,
            timestamp=time.time()
        )
        
        if self.debug:
            logger.debug(f"Collected metrics: {asdict(metrics)}")
        
        return metrics
        
    def _log_metrics(self, metrics: ResourceMetrics):
        """Log metrics to both TensorBoard and MLflow."""
        metrics_dict = asdict(metrics)
        self.metrics_buffer.append(metrics_dict)
        
        try:
            # Log to TensorBoard
            if self.tensorboard:
                for name, value in metrics_dict.items():
                    if value is not None and name != 'timestamp':
                        self.tensorboard.add_scalar(f"resources/{name}", value, metrics_dict['timestamp'])
                        
            # Periodically log to MLflow (every 10 records)
            if self.mlflow_active and len(self.metrics_buffer) >= 10:
                # Calculate and log average metrics
                avg_metrics = {
                    name: np.mean([m[name] for m in self.metrics_buffer if m[name] is not None])
                    for name in metrics_dict.keys()
                    if name != 'timestamp'
                }
                self.mlflow_client.log_metrics(self.run_id, avg_metrics, step=metrics_dict['timestamp'])
                self.metrics_buffer = []
                
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            
    def log_evaluation_metric(self, metric_name: str, value: float, step: int):
        """Log an evaluation metric."""
        logger.debug(f"Logging evaluation metric: {metric_name}={value} at step {step}")
        try:
            if self.tensorboard:
                self.tensorboard.add_scalar(f"eval/{metric_name}", value, step)
            if self.mlflow_active:
                mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            logger.error(f"Error logging evaluation metric: {str(e)}")
            
    def log_batch_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics for a batch of evaluations."""
        logger.debug(f"Logging batch metrics at step {step}: {metrics}")
        try:
            # Log to TensorBoard
            if self.tensorboard:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard.add_scalar(f"batch/{name}", value, step)
                
            # Log to MLflow
            if self.mlflow_active:
                batch_metrics = {
                    f"batch_{k}": v for k, v in metrics.items() 
                    if isinstance(v, (int, float))
                }
                for k, v in batch_metrics.items():
                    mlflow.log_metric(k, v, step=step)
        except Exception as e:
            logger.error(f"Error logging batch metrics: {str(e)}")
            
    def close(self):
        """Clean up resources."""
        logger.debug("Closing MetricsMonitor...")
        self.stop_monitoring()
        if self.tensorboard:
            self.tensorboard.close()
        if self.mlflow_active and self.run_id:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.error(f"Error terminating MLflow run: {str(e)}")
        logger.debug("MetricsMonitor closed") 