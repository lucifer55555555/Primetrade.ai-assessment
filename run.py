"""
MLOps Pipeline for Cryptocurrency Signal Generation
Computes rolling mean signals from OHLCV data with comprehensive logging and metrics.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger('mlops_pipeline')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If required fields are missing
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
    
    # Validate required fields
    required_fields = ['seed', 'window', 'version']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Validate field types
    if not isinstance(config['seed'], int):
        raise ValueError("Config 'seed' must be an integer")
    if not isinstance(config['window'], int) or config['window'] <= 0:
        raise ValueError("Config 'window' must be a positive integer")
    if not isinstance(config['version'], str):
        raise ValueError("Config 'version' must be a string")
    
    return config


def load_data(data_path: str, required_columns: list) -> pd.DataFrame:
    """
    Load and validate CSV data.
    
    Args:
        data_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        pd.errors.ParserError: If CSV is invalid
        ValueError: If required columns are missing
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if Path(data_path).stat().st_size == 0:
        raise pd.errors.EmptyDataError(f"Data file is empty: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Invalid CSV format: {e}")
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean and generate signals.
    
    Args:
        df: Input DataFrame with 'close' column
        window: Rolling window size
        logger: Logger instance
        
    Returns:
        DataFrame with added 'rolling_mean' and 'signal' columns
    """
    logger.info(f"Calculating rolling mean with window={window}")
    df['rolling_mean'] = df['close'].rolling(window=window, min_periods=1).mean()
    
    logger.info("Generating signals")
    df['signal'] = (df['close'] > df['rolling_mean']).astype(int)
    
    return df


def calculate_metrics(df: pd.DataFrame, start_time: float) -> Dict[str, Any]:
    """
    Calculate pipeline metrics.
    
    Args:
        df: Processed DataFrame
        start_time: Pipeline start timestamp
        
    Returns:
        Dictionary with metrics
    """
    rows_processed = len(df)
    signal_rate = float(df['signal'].mean())
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        'rows_processed': rows_processed,
        'signal_rate': signal_rate,
        'latency_ms': latency_ms
    }


def write_output(metrics: Dict[str, Any], config: Dict[str, Any], 
                 output_path: str, status: str = "success", 
                 error_message: Optional[str] = None) -> None:
    """
    Write metrics to JSON file.
    
    Args:
        metrics: Computed metrics
        config: Configuration dictionary
        output_path: Output file path
        status: Job status ("success" or "error")
        error_message: Optional error message
    """
    if status == "error":
        output = {
            "version": config.get('version', 'unknown'),
            "status": "error",
            "error_message": error_message
        }
    else:
        output = {
            "version": config['version'],
            "rows_processed": metrics['rows_processed'],
            "metric": "signal_rate",
            "value": round(metrics['signal_rate'], 4),
            "latency_ms": metrics['latency_ms'],
            "seed": config['seed'],
            "status": "success"
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also print to stdout for Docker
    print(f"\nFinal metrics: {json.dumps(output, indent=2)}")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='MLOps Signal Generation Pipeline')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--config', required=True, help='Configuration YAML file path')
    parser.add_argument('--output', required=True, help='Output metrics JSON file path')
    parser.add_argument('--log-file', required=True, help='Log file path')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    logger = None
    
    try:
        # Setup logging
        logger = setup_logging(args.log_file)
        logger.info("Job started")
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Config loaded: seed={config['seed']}, window={config['window']}, version={config['version']}")
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = load_data(args.input, required_columns=['close'])
        logger.info(f"Data loaded: {len(df)} rows")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty after loading")
        
        # Process data
        df = compute_signals(df, config['window'], logger)
        
        # Calculate metrics
        metrics = calculate_metrics(df, start_time)
        logger.info(f"Metrics: signal_rate={metrics['signal_rate']:.4f}, rows_processed={metrics['rows_processed']}")
        
        # Write output
        write_output(metrics, config, args.output)
        
        # Completion log
        logger.info(f"Job completed successfully in {metrics['latency_ms']}ms")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = str(e)
        
        if logger:
            logger.error(f"Job failed: {error_msg}")
        else:
            print(f"Fatal error before logging setup: {error_msg}", file=sys.stderr)
        
        # Write error output if possible
        try:
            config = load_config(args.config) if 'args' in locals() else {'version': 'unknown'}
            write_output({}, config, args.output, status="error", error_message=error_msg)
        except:
            pass
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
