# Predictive Maintenance with PCA, Statistics, and LSTMs

This project demonstrates predictive maintenance using PCA, statistical models, and LSTM neural networks.

## Article

Medium article: [Predictive Maintenance with Time Series in Python Using PCA, Statistics, and LSTMs](https://medium.com/@kylejones_47003/predictive-maintenance-with-time-series-in-python-using-pca-statistics-and-lstms-430aa8e79fc5)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Predictive maintenance functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and separator
- Sensor columns for modeling
- Model selection (exponential regression, Weibull AFT)
- Output settings

## Methods

### Exponential Regression
- Poisson GLM with log link
- Models RUL as exponential function of sensors
- Fast and interpretable

### Weibull AFT Model
- Accelerated Failure Time model
- Handles censored data
- Provides survival probabilities

## Caveats

- Requires CMAPSS dataset or similar format.
- RUL values must be positive (zero values are replaced with small epsilon).
- Model performance depends on sensor selection.
