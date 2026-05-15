"""CMAPSS FD001: RUL regression, Weibull AFT, PCA features, censoring, Cox PH, optional DeepSurv (pycox)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

CMAPSS_COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_measurement_{i}" for i in range(1, 22)]
)


def load_cmapss(filepath: Path | str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
    )
    df["max_cycle"] = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    return df


def model_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "unit_number",
            "time_in_cycles",
            "RUL",
            "sensor_measurement_2",
            "sensor_measurement_3",
        ]
    ].copy()


def _plot_actual_vs_pred(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path | None = None,
    *,
    show: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    lo, hi = float(y_true.min()), float(y_true.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("Actual RUL")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()


def fit_poisson_log_glm(df: pd.DataFrame):
    x = sm.add_constant(df[["sensor_measurement_2", "sensor_measurement_3"]])
    y = df["RUL"]
    return sm.GLM(y, x, family=sm.families.Poisson(link=sm.families.links.Log())).fit()


def fit_weibull_aft_two_sensors(df: pd.DataFrame) -> WeibullAFTFitter:
    aft_df = df[["RUL", "sensor_measurement_2", "sensor_measurement_3"]].copy()
    aft_df["event_observed"] = 1
    aft_df["RUL"] = aft_df["RUL"].clip(lower=1e-3)
    m = WeibullAFTFitter()
    m.fit(aft_df, duration_col="RUL", event_col="event_observed")
    return m


def simulate_censoring(
    df: pd.DataFrame, censor_rate: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    df_c = df.copy()
    rng = np.random.default_rng(seed)
    units = df["unit_number"].unique()
    n_c = max(1, int(censor_rate * len(units)))
    censored_units = set(rng.choice(units, size=n_c, replace=False))

    for unit in censored_units:
        max_cycle = df_c.loc[df_c["unit_number"] == unit, "time_in_cycles"].max()
        cutoff = int(max_cycle * 0.7)
        mask = (df_c["unit_number"] == unit) & (df_c["time_in_cycles"] > cutoff)
        df_c = df_c.loc[~mask]

    df_c["max_cycle"] = df_c.groupby("unit_number")["time_in_cycles"].transform("max")
    df_c["RUL"] = df_c["max_cycle"] - df_c["time_in_cycles"]
    df_c["event_observed"] = 1
    df_c.loc[df_c["unit_number"].isin(censored_units), "event_observed"] = 0
    return df_c


def apply_pca_weibull(
    df: pd.DataFrame, log_sensors: bool, out_plot: Path | None
) -> tuple[float, WeibullAFTFitter]:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_measurement_")]
    x = df[sensor_cols]
    if log_sensors:
        x = np.log(x + 1e-6)
    scaled = StandardScaler().fit_transform(x)
    z = PCA(n_components=3).fit_transform(scaled)
    pca_df = pd.DataFrame(z, columns=["PC1", "PC2", "PC3"])
    base = df[["RUL"]].reset_index(drop=True)
    merged = pd.concat([base, pca_df], axis=1)
    aft_df = merged.copy()
    aft_df["event_observed"] = 1
    aft_df["RUL"] = aft_df["RUL"].clip(lower=1e-3)
    m = WeibullAFTFitter()
    m.fit(aft_df, duration_col="RUL", event_col="event_observed")
    pred = m.predict_median(aft_df[["PC1", "PC2", "PC3"]])
    rmse = float(np.sqrt(mean_squared_error(merged["RUL"], pred)))
    if out_plot:
        _plot_actual_vs_pred(
            merged["RUL"],
            pred,
            "Weibull AFT (PCA features)" + (" — log sensors" if log_sensors else ""),
            "Predicted RUL",
            out_plot,
        )
    return rmse, m


def fit_cox_censored(df: pd.DataFrame) -> CoxPHFitter:
    cox_df = df[
        ["RUL", "event_observed", "sensor_measurement_2", "sensor_measurement_3"]
    ].copy()
    cox_df["RUL"] = cox_df["RUL"].clip(lower=1e-3)
    m = CoxPHFitter()
    m.fit(cox_df, duration_col="RUL", event_col="event_observed")
    return m


def run_pycox_deepsurv(df_censored: pd.DataFrame) -> float | None:
    try:
        import torchtuples as tt
        from pycox.evaluation import EvalSurv
        from pycox.models import DeepSurv
    except ImportError:
        print(
            "Skipping pycox DeepSurv (install optional extra: uv sync --extra deepsurv)"
        )
        return None

    features = ["sensor_measurement_2", "sensor_measurement_3"]
    x = df_censored[features].astype(np.float32).values
    t = df_censored["RUL"].clip(lower=1e-3).astype(np.float32).values
    e = df_censored["event_observed"].astype(np.int32).values
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)
    net = tt.practical.MLPVanilla(
        x_s.shape[1],
        [32, 16],
        1,
        batch_norm=True,
        dropout=0.1,
        output_bias=False,
        activation="ReLU",
    )
    model = DeepSurv(net, tt.optim.Adam)
    model.fit(x_s, (t, e), batch_size=256, epochs=100, verbose=False)
    surv = model.predict_surv_df(x_s)
    ev = EvalSurv(surv, t, e, censor_surv="km")
    c = ev.concordance_td("antolini")
    print("DeepSurv (pycox) time-dependent concordance (Antolini):", float(c))
    return float(c)


def run_cmapss(
    train_path: Path,
    out_dir: Path,
    *,
    run_deepsurv: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cmapss(train_path)
    df_model = model_subset(df)

    glm = fit_poisson_log_glm(df_model)
    df_model["predicted_rul_exp"] = glm.predict(
        sm.add_constant(df_model[["sensor_measurement_2", "sensor_measurement_3"]])
    )
    print(glm.summary())

    aft = fit_weibull_aft_two_sensors(df_model)
    print(aft.summary)
    cov = df_model[["sensor_measurement_2", "sensor_measurement_3"]]
    df_model["predicted_rul_weibull"] = aft.predict_median(cov)

    _plot_actual_vs_pred(
        df_model["RUL"],
        df_model["predicted_rul_exp"],
        "Poisson GLM (log link) vs RUL",
        "Predicted RUL",
        out_dir / "exp_regression_rul.png",
    )
    _plot_actual_vs_pred(
        df_model["RUL"],
        df_model["predicted_rul_weibull"],
        "Weibull AFT vs RUL",
        "Predicted RUL",
        out_dir / "weibull_regression_rul.png",
    )

    x_s = df_model[["sensor_measurement_2", "sensor_measurement_3"]]
    y = df_model["RUL"]
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(x_s, y)
    df_model["predicted_rul_rf"] = rf.predict(x_s)
    print(
        f"Random forest RMSE: {np.sqrt(mean_squared_error(y, df_model['predicted_rul_rf'])):.2f}"
    )
    print(
        f"GLM RMSE: {np.sqrt(mean_squared_error(y, df_model['predicted_rul_exp'])):.2f} | "
        f"Weibull RMSE: {np.sqrt(mean_squared_error(y, df_model['predicted_rul_weibull'])):.2f}"
    )

    rmse_pca, _ = apply_pca_weibull(
        df, log_sensors=False, out_plot=out_dir / "weibull_pca_rul.png"
    )
    rmse_logpca, _ = apply_pca_weibull(
        df, log_sensors=True, out_plot=out_dir / "weibull_logpca_rul.png"
    )
    print(
        f"Weibull + PCA RMSE: {rmse_pca:.2f} | Weibull + log-PCA RMSE: {rmse_logpca:.2f}"
    )

    df_c = simulate_censoring(df)
    cox = fit_cox_censored(df_c)
    print("Cox PH concordance (censored simulation):", float(cox.concordance_index_))

    if run_deepsurv:
        run_pycox_deepsurv(df_c)


def compute_rul(df: pd.DataFrame, time_col: str, group_col: str) -> pd.DataFrame:
    out = df.copy()
    out["max_cycle"] = out.groupby(group_col)[time_col].transform("max")
    out["RUL"] = out["max_cycle"] - out[time_col]
    return out


def run_cmapss_nonlinear_sensors(train_path: Path) -> None:
    """GLM on linear terms + Weibull/Cox on linear + squared + interaction features."""
    df = load_cmapss(train_path)
    df = compute_rul(df, "time_in_cycles", "unit_number")
    df_model = df[
        [
            "unit_number",
            "time_in_cycles",
            "RUL",
            "sensor_measurement_2",
            "sensor_measurement_3",
        ]
    ].copy()
    df_model["sensor_2_sq"] = df_model["sensor_measurement_2"] ** 2
    df_model["sensor_3_sq"] = df_model["sensor_measurement_3"] ** 2
    df_model["sensor_2_x_3"] = (
        df_model["sensor_measurement_2"] * df_model["sensor_measurement_3"]
    )
    features = [
        "sensor_measurement_2",
        "sensor_measurement_3",
        "sensor_2_sq",
        "sensor_3_sq",
        "sensor_2_x_3",
    ]

    x_lin = sm.add_constant(df_model[["sensor_measurement_2", "sensor_measurement_3"]])
    exp_m = sm.GLM(
        df_model["RUL"], x_lin, family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit()
    df_model["rul_pred_exp"] = exp_m.predict(x_lin)

    df_model["event_observed"] = 1
    wdf = df_model[["RUL", "event_observed"] + features].copy()
    wdf["RUL"] = wdf["RUL"].clip(lower=1e-3)
    w = WeibullAFTFitter()
    w.fit(wdf, duration_col="RUL", event_col="event_observed")
    df_model["rul_pred_weibull"] = w.predict_median(df_model[features])

    cdf = df_model[["RUL", "event_observed"] + features].copy()
    cdf["RUL"] = cdf["RUL"].clip(lower=1e-3)
    c = CoxPHFitter()
    c.fit(cdf, duration_col="RUL", event_col="event_observed")

    def _rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    print(
        f"Exponential (linear features) RMSE: {_rmse(df_model['RUL'], df_model['rul_pred_exp']):.2f}"
    )
    print(
        f"Weibull (nonlinear features) RMSE: {_rmse(df_model['RUL'], df_model['rul_pred_weibull']):.2f}"
    )
    print(f"Cox concordance (nonlinear features): {float(c.concordance_index_):.3f}")

    _plot_actual_vs_pred(
        df_model["RUL"],
        df_model["rul_pred_exp"],
        "Poisson GLM vs RUL",
        "Predicted RUL",
        show=True,
    )
    _plot_actual_vs_pred(
        df_model["RUL"],
        df_model["rul_pred_weibull"],
        "Weibull AFT (nonlinear features)",
        "Predicted RUL",
        show=True,
    )
