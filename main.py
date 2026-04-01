#!/usr/bin/env python3
"""
Predictive Maintenance with PCA, Statistics, and LSTMs

Main entry point for running predictive maintenance analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import (
    load_cmapss_data,
    calculate_rul,
    fit_exponential_regression,
    fit_weibull_aft,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Predictive Maintenance with PCA, Statistics, and LSTMs')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else Path(config['data']['source'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
        df = load_cmapss_data(data_path, config['data']['separator'])
    
        df = calculate_rul(df)
    
    df_model = df[["unit_number", "time_in_cycles", "RUL"] + config['model']['sensor_columns']]
    
    if config['model']['exponential_regression']:
                glm_results, X = fit_exponential_regression(df_model, config['model']['sensor_columns'])
        df_model["predicted_rul_exp"] = glm_results.predict(X)
        
                logging.info(glm_results.summary())
        
        plot_rul_predictions(
            df_model["RUL"], df_model["predicted_rul_exp"],
            "Exponential Regression: Actual vs Predicted RUL",
            output_dir / 'exp_regression_rul.png'
        )
    
    if config['model']['weibull_aft']:
                aft = fit_weibull_aft(df_model, config['model']['sensor_columns'])
                logging.info(aft.summary)
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

