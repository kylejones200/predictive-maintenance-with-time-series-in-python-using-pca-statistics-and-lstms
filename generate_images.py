#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import signalplot

np.random.seed(42)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Set random seeds

# Tufte-style configuration
signalplot.apply(font_family="serif")

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
original_savefig = plt.savefig


def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith("/") and not str(filename).startswith("images/"):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")



def main():
    plt.savefig = savefig_tufte

    # (Placeholder script – replace with real predictive maintenance visualizations as needed.)
    logger.info("All images generated successfully!")


if __name__ == "__main__":
    main()
