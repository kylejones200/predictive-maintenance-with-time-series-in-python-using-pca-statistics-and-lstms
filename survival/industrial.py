"""Industrial PM dataset: GLM, Weibull AFT, and Cox PH (RUL or log-RUL targets)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def prepare_data_rul(filepath: Path | str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["RUL"] = df.groupby("Fault_Diagnosis").cumcount(ascending=False) + 1
    df["event_observed"] = np.where(df["Fault_Diagnosis"] > 0, 1, 0)
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])
    if "Operator_Shift_Data" in df.columns:
        df = pd.get_dummies(df, columns=["Operator_Shift_Data"], drop_first=True)
    return df


def prepare_data_log_rul(filepath: Path | str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "Fault_Trigger_Timestamps" in df.columns:
        df["event_observed"] = df["Fault_Trigger_Timestamps"]
    elif "Failure_Mode_Indicators" in df.columns:
        df["event_observed"] = df["Failure_Mode_Indicators"]
    else:
        df["event_observed"] = np.where(df["Fault_Diagnosis"] > 0, 1, 0)

    if "Machine_ID" in df.columns:
        df["RUL"] = df.groupby("Machine_ID").cumcount(ascending=False) + 1
    else:
        df["RUL"] = df.groupby("Fault_Diagnosis").cumcount(ascending=False) + 1

    df["log_RUL"] = np.log1p(df["RUL"])

    drop_cols = (
        ["Datetime", "Machine_ID"] if "Machine_ID" in df.columns else ["Datetime"]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "Operator_Shift_Data" in df.columns:
        df = pd.get_dummies(df, columns=["Operator_Shift_Data"], drop_first=True)

    return df


def scale_features(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns.difference(exclude_cols)
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
    return out


def fit_exponential(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.Series, object]:
    x = sm.add_constant(df[features].apply(pd.to_numeric, errors="coerce"))
    y = pd.to_numeric(df[target], errors="coerce")
    model = sm.GLM(y, x, family=sm.families.Poisson(link=sm.families.links.Log())).fit()
    return model.predict(x), model


def fit_weibull(
    df: pd.DataFrame, features: list[str], duration_col: str
) -> tuple[pd.Series, WeibullAFTFitter]:
    cols = [duration_col, "event_observed"] + features
    sub = df[cols].copy()
    sub[duration_col] = sub[duration_col].clip(lower=1e-3)
    m = WeibullAFTFitter()
    m.fit(sub, duration_col=duration_col, event_col="event_observed")
    return m.predict_median(df[features]), m


def fit_cox(
    df: pd.DataFrame, features: list[str], duration_col: str
) -> tuple[float, CoxPHFitter]:
    cols = [duration_col, "event_observed"] + features
    sub = df[cols].copy()
    sub[duration_col] = sub[duration_col].clip(lower=1e-3)
    m = CoxPHFitter()
    m.fit(sub, duration_col=duration_col, event_col="event_observed")
    return float(m.concordance_index_), m


def plot_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: Path | str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="black")
    lo, hi = float(y_true.min()), float(y_true.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


def _numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    num = df.select_dtypes(include="number").columns
    return [c for c in num if c not in exclude]


def run_industrial_rul(csv_path: Path, *, plot_dir: Path | None = None) -> None:
    df = prepare_data_rul(csv_path)
    exclude = {"RUL", "event_observed", "Fault_Diagnosis"}
    df = scale_features(df, list(exclude))
    features = _numeric_feature_columns(df, exclude)
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    pred_e, _exp = fit_exponential(df, features, "RUL")
    df["pred_rul_exp"] = pred_e
    pred_w, _w = fit_weibull(df, features, "RUL")
    df["pred_rul_weibull"] = pred_w
    conc, _c = fit_cox(df, features, "RUL")

    mse_e = mean_squared_error(df["RUL"], df["pred_rul_exp"])
    mse_w = mean_squared_error(df["RUL"], df["pred_rul_weibull"])
    print(f"Exponential MSE (RUL): {mse_e:.4f}")
    print(f"Weibull MSE (RUL): {mse_w:.4f}")
    print(f"Cox concordance: {conc:.3f}")

    pe = plot_dir or Path(".")
    pe.mkdir(parents=True, exist_ok=True)
    plot_predictions(
        df["RUL"],
        df["pred_rul_exp"],
        "Exponential (Poisson log link)",
        "Actual RUL",
        "Predicted RUL",
        pe / "industrial_exp_vs_rul.png",
    )
    plot_predictions(
        df["RUL"],
        df["pred_rul_weibull"],
        "Weibull AFT",
        "Actual RUL",
        "Predicted RUL",
        pe / "industrial_weibull_vs_rul.png",
    )


def run_industrial_log_rul(csv_path: Path, *, plot_dir: Path | None = None) -> None:
    df = prepare_data_log_rul(csv_path)
    exclude = {"RUL", "log_RUL", "event_observed", "Fault_Diagnosis"}
    df = scale_features(df, list(exclude))
    features = _numeric_feature_columns(df, exclude)
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    pred_e, _ = fit_exponential(df, features, "log_RUL")
    df["pred_log_rul_exp"] = pred_e
    pred_w, _ = fit_weibull(df, features, "log_RUL")
    df["pred_log_rul_weibull"] = pred_w
    conc, _ = fit_cox(df, features, "log_RUL")

    mse_e = mean_squared_error(df["log_RUL"], df["pred_log_rul_exp"])
    mse_w = mean_squared_error(df["log_RUL"], df["pred_log_rul_weibull"])
    print(f"Exponential MSE (log RUL): {mse_e:.4f}")
    print(f"Weibull MSE (log RUL): {mse_w:.4f}")
    print(f"Cox concordance (log RUL): {conc:.3f}")

    pe = plot_dir or Path(".")
    pe.mkdir(parents=True, exist_ok=True)
    plot_predictions(
        df["log_RUL"],
        df["pred_log_rul_exp"],
        "Exponential (log RUL)",
        "Actual log RUL",
        "Predicted log RUL",
        pe / "industrial_log_exp.png",
    )
    plot_predictions(
        df["log_RUL"],
        df["pred_log_rul_weibull"],
        "Weibull AFT (log RUL)",
        "Actual log RUL",
        "Predicted log RUL",
        pe / "industrial_log_weibull.png",
    )
