"""Regenerate biological L-MNIST LaTeX tables from saved results.

Usage:
    python generate_bio_init_latex_table.py runs_revision_snn/bio_init_table_YYYYMMDD_HHMMSS
    python generate_bio_init_latex_table.py runs_revision_snn/bio_init_table_YYYYMMDD_HHMMSS/results.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from reproduce_bio_init_table import build_latex_table


def resolve_results_path(path: Path) -> Path:
    return path / "results.json" if path.is_dir() else path


def condition_order(summary: Dict[str, Dict[str, Any]]) -> List[str]:
    preferred = ["random_no_freeze", "bio_no_freeze", "bio_freeze"]
    ordered = [condition for condition in preferred if condition in summary]
    ordered.extend(condition for condition in summary if condition not in ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate biological L-MNIST LaTeX tables from saved results.json.")
    parser.add_argument("results", type=Path, help="Output directory containing results.json, or the results.json file itself.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated .tex files. Defaults to the results directory.")
    parser.add_argument("--prefix", type=str, default="bio_init_per_step_table", help="Output filename prefix.")
    args = parser.parse_args()

    results_path = resolve_results_path(args.results)
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = payload["config"]
    summary = payload["summary"]
    conditions = condition_order(summary)
    n_readout_steps = int(config.get("n_readout_steps", 5))
    metric_keys = [f"step{idx}" for idx in range(1, n_readout_steps + 1)] + ["vote_avg"]
    report_checkpoint = str(config.get("report_checkpoint", summary[conditions[0]].get("reported_checkpoint_type", "best")))
    best_metric = str(config.get("best_metric", "step1"))
    train_activation_for_unfrozen = bool(config.get("train_activation_for_unfrozen", False))

    output_dir = args.output_dir or results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_latex_table(
        summary,
        conditions,
        metric_keys,
        report_checkpoint,
        best_metric,
        percent=False,
        train_activation_for_unfrozen=train_activation_for_unfrozen,
    )
    percent_table = build_latex_table(
        summary,
        conditions,
        metric_keys,
        report_checkpoint,
        best_metric,
        percent=True,
        train_activation_for_unfrozen=train_activation_for_unfrozen,
    )

    table_path = output_dir / f"{args.prefix}.tex"
    percent_table_path = output_dir / f"{args.prefix}_percent.tex"
    table_path.write_text(table + "\n", encoding="utf-8")
    percent_table_path.write_text(percent_table + "\n", encoding="utf-8")

    print(f"Wrote {table_path}")
    print(f"Wrote {percent_table_path}")


if __name__ == "__main__":
    main()