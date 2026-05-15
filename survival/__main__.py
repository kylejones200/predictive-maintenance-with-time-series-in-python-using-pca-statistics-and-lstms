"""CLI for survival / RUL examples: ``python -m survival --help``."""

from __future__ import annotations

import argparse
from pathlib import Path

from survival.cmapss import run_cmapss, run_cmapss_nonlinear_sensors
from survival.industrial import run_industrial_log_rul, run_industrial_rul


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machine failure survival models (CMAPSS + industrial)."
    )
    parser.add_argument(
        "command",
        choices=["cmapss", "cmapss-nl", "industrial", "industrial-log"],
        help="Pipeline to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing train_FD001.txt and/or IndFD-PM-DT dataset.csv",
    )
    parser.add_argument(
        "--deepsurv",
        action="store_true",
        help="Run optional pycox DeepSurv (requires: uv sync --extra deepsurv)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    if args.command == "cmapss":
        train = data_dir / "train_FD001.txt"
        if not train.is_file():
            raise FileNotFoundError(f"Missing CMAPSS file: {train}")
        run_cmapss(train, data_dir / "survival_outputs", run_deepsurv=args.deepsurv)
    elif args.command == "cmapss-nl":
        train = data_dir / "train_FD001.txt"
        if not train.is_file():
            raise FileNotFoundError(f"Missing CMAPSS file: {train}")
        run_cmapss_nonlinear_sensors(train)
    elif args.command == "industrial":
        csv_path = data_dir / "IndFD-PM-DT dataset.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing industrial CSV: {csv_path}")
        run_industrial_rul(csv_path, plot_dir=data_dir / "survival_outputs")
    elif args.command == "industrial-log":
        csv_path = data_dir / "IndFD-PM-DT dataset.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing industrial CSV: {csv_path}")
        run_industrial_log_rul(csv_path, plot_dir=data_dir / "survival_outputs")


if __name__ == "__main__":
    main()
