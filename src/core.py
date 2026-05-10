"""Core functions for predictive maintenance with PCA, statistics, and LSTMs."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import statsmodels.api as sm
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_cmapss_data(data_path: Path, sep: str = r'\s+') -> pd.DataFrame:
    """Load CMAPSS dataset."""
    column_names = ["unit_number", "time_in_cycles"] + \
                   [f"op_setting_{i}" for i in range(1, 4)] + \
                   [f"sensor_measurement_{i}" for i in range(1, 22)]
    df = pd.read_csv(data_path, sep=sep, header=None, names=column_names)
    return df

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life (RUL)."""
    df = df.copy()
    df["max_cycle"] = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df["RUL"] = df["RUL"].apply(lambda x: x if x > 0 else 1e-3)
    return df

def fit_exponential_regression(df: pd.DataFrame, sensor_cols: list) -> Tuple:
    """Fit exponential regression using Poisson GLM with log link."""
    X = df[sensor_cols]
    X = sm.add_constant(X)
    y = df["RUL"]
    glm_model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.log()))
    glm_results = glm_model.fit()
    return glm_results, X

def fit_weibull_aft(df: pd.DataFrame, sensor_cols: list) -> WeibullAFTFitter:
    """Fit Weibull Accelerated Failure Time model."""
    aft_df = df[["RUL"] + sensor_cols].copy()
    aft_df["event_observed"] = 1
    
    aft = WeibullAFTFitter()
    aft.fit(aft_df, duration_col="RUL", event_col="event_observed")
    return aft

def plot_rul_predictions(actual: pd.Series, predicted: pd.Series, title: str, output_path: Path, plot: bool = False):
    """Plot RUL predictions vs actual """
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
    
        ax.scatter(actual, predicted, alpha=0.3, s=20, color="#4A90A4", edgecolors='none')
        max_val = max(actual.max(), predicted.max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2, label="Perfect Prediction")
    
        ax.set_xlabel("Actual RUL")
        ax.set_ylabel("Predicted RUL")
        ax.legend(loc='best')
    
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()

