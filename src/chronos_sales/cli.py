from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import typer

from .config import PROJECT_ROOT, load_config
from .context_horizon_benchmark import run_context_horizon_benchmark
from .data import validation_indices
from .forecasting import run_evaluation, save_metric_row
from .online_retail import (
    load_online_retail_revenue,
    prepare_online_retail_revenue,
)
from .plotting import plot_revenue_prediction
from .revenue_evaluation import save_naive_predictions, write_revenue_forecasts
from .rolling_benchmark import run_rolling_benchmark
from .training import find_latest_checkpoint, run_finetuning
from .tabular_baselines import SUPPORTED_MODELS, run_tabular_benchmarks


app = typer.Typer(
    help="UCI Online Retail II daily-revenue forecasting with Chronos v1.5.3.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _config(path: Optional[Path], dataset: str = "online-retail-revenue"):
    if path is None:
        path = PROJECT_ROOT / "configs" / "online-retail-revenue.yaml"
    return load_config(path)


def _dataset_name(value: str) -> str:
    if value != "online-retail-revenue":
        raise typer.BadParameter("dataset must be 'online-retail-revenue'")
    return value


@app.command("prepare-online-retail")
def prepare_online_retail_command(
    source_path: Path = typer.Option(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "online_retail_ii"
        / "online_retail_II.xlsx"
    ),
    output_dir: Path = typer.Option(PROJECT_ROOT / "data" / "online_retail_revenue"),
    config_path: Optional[Path] = typer.Option(None),
):
    """Convert UCI Online Retail II transactions into daily GBP revenue."""
    manifest = prepare_online_retail_revenue(
        source_path=source_path,
        output_dir=output_dir,
        config=_config(config_path, "online-retail-revenue"),
    )
    typer.echo(
        f"Prepared {manifest['day_count']} daily revenue observations at "
        f"{manifest['arrow_path']}"
    )


@app.command("zero-shot")
def zero_shot_command(
    dataset: str = typer.Option(
        "online-retail-revenue", help="Only online-retail-revenue is supported."
    ),
    data_path: Optional[Path] = typer.Option(None),
    output_dir: Optional[Path] = typer.Option(None),
    device: str = typer.Option("auto", help="auto, cpu, or cuda"),
    batch_size: int = typer.Option(32),
    config_path: Optional[Path] = typer.Option(None),
):
    """Evaluate public Chronos on the final 28 days of daily revenue."""
    dataset = _dataset_name(dataset)
    config = _config(config_path, dataset)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "online_retail_revenue" / "zero_shot"
    data_path = data_path or (
        PROJECT_ROOT / "data" / "online_retail_revenue" / "daily_revenue.parquet"
    )
    prepared_dataset = load_online_retail_revenue(data_path)
    row = run_evaluation(
        config.model_id,
        output_dir,
        config,
        split_name="test",
        device=device,
        batch_size=batch_size,
        label="zero_shot",
        dataset=prepared_dataset,
        dataset_name=dataset,
    )
    save_metric_row(row, output_dir / "zero_shot_metrics.csv")
    prediction_path = output_dir / "zero_shot_test_predictions.npz"
    write_revenue_forecasts(
        prediction_path,
        prepared_dataset,
        config,
        output_dir,
        "zero_shot",
        entity_column="business_id",
        entity_label="business",
    )
    naive_row, naive_path = save_naive_predictions(
        prepared_dataset, config, output_dir, dataset_name=dataset
    )
    write_revenue_forecasts(
        naive_path,
        prepared_dataset,
        config,
        output_dir,
        "seasonal_naive_7",
        entity_column="business_id",
        entity_label="business",
    )
    pd.DataFrame([naive_row, row]).to_csv(
        output_dir / "baseline_comparison_metrics.csv", index=False
    )
    typer.echo(pd.DataFrame([row]).to_string(index=False))


@app.command("finetune")
def finetune_command(
    dataset: str = typer.Option(
        "online-retail-revenue", help="Only online-retail-revenue is supported."
    ),
    arrow_path: Optional[Path] = typer.Option(None),
    output_dir: Optional[Path] = typer.Option(None),
    max_steps: int = typer.Option(0, help="0 uses the config default (1,000)."),
    save_steps: int = typer.Option(0, help="0 uses the config default (250)."),
    batch_size: int = typer.Option(32),
    resume: bool = typer.Option(False, help="Resume from the newest saved checkpoint."),
    resume_from: Optional[Path] = typer.Option(None),
    cpu: bool = typer.Option(False, help="Use only for the 10-step local smoke test."),
    config_path: Optional[Path] = typer.Option(None),
):
    """Fine-tune Chronos on a leakage-free retail time-series prefix."""
    dataset = _dataset_name(dataset)
    config = _config(config_path, dataset)
    if arrow_path is None:
        arrow_path = PROJECT_ROOT / "data" / "online_retail_revenue" / "arrow" / "train"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "online_retail_revenue" / "finetune"
    checkpoint = resume_from
    if resume and checkpoint is None:
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint is None:
            raise typer.BadParameter("No checkpoint was found to resume.")
    run_dir = run_finetuning(
        arrow_path,
        output_dir,
        config,
        max_steps=max_steps or None,
        save_steps=save_steps or None,
        batch_size=batch_size,
        resume_from_checkpoint=checkpoint,
        cpu=cpu,
    )
    typer.echo(f"Training output: {run_dir}")


def _checkpoint_candidates(output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for run_dir in sorted(output_dir.glob("run-*")):
        candidates.extend(sorted(run_dir.glob("checkpoint-*")))
    return [path for path in candidates if (path / "config.json").exists()]


@app.command("evaluate")
def evaluate_command(
    dataset: str = typer.Option(
        "online-retail-revenue", help="Only online-retail-revenue is supported."
    ),
    data_path: Optional[Path] = typer.Option(None),
    finetune_dir: Optional[Path] = typer.Option(None),
    output_dir: Optional[Path] = typer.Option(None),
    validation_series: int = typer.Option(0, help="0 uses config default (2,048)."),
    device: str = typer.Option("auto"),
    batch_size: int = typer.Option(32),
    config_path: Optional[Path] = typer.Option(None),
):
    """Select the best checkpoint on validation, then evaluate the final test split."""
    dataset = _dataset_name(dataset)
    config = _config(config_path, dataset)
    if finetune_dir is None:
        finetune_dir = PROJECT_ROOT / "outputs" / "online_retail_revenue" / "finetune"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "online_retail_revenue" / "evaluation"
    candidates = _checkpoint_candidates(finetune_dir)
    if not candidates:
        raise typer.BadParameter(f"No Chronos checkpoints found under {finetune_dir}")
    data_path = data_path or (
        PROJECT_ROOT / "data" / "online_retail_revenue" / "daily_revenue.parquet"
    )
    evaluation_dataset = load_online_retail_revenue(data_path)
    selected_indices = validation_indices(
        len(evaluation_dataset),
        validation_series or config.validation_series,
        config.seed,
    )
    validation_rows = []
    for checkpoint in candidates:
        row = run_evaluation(
            str(checkpoint),
            output_dir,
            config,
            split_name="validation",
            indices=selected_indices,
            device=device,
            batch_size=batch_size,
            label=checkpoint.parent.name + "_" + checkpoint.name,
            save_predictions=False,
            dataset=evaluation_dataset,
            dataset_name=dataset,
        )
        validation_rows.append(row)
    validation_frame = pd.DataFrame(validation_rows).sort_values(["WQL", "MASE"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_frame.to_csv(output_dir / "checkpoint_validation_metrics.csv", index=False)
    best_checkpoint = str(validation_frame.iloc[0]["model"])
    (output_dir / "best_checkpoint.txt").write_text(best_checkpoint, encoding="utf-8")

    final_row = run_evaluation(
        best_checkpoint,
        output_dir,
        config,
        split_name="test",
        device=device,
        batch_size=batch_size,
        label="finetuned",
        dataset=evaluation_dataset,
        dataset_name=dataset,
    )
    save_metric_row(final_row, output_dir / "finetuned_metrics.csv")
    write_revenue_forecasts(
        output_dir / "finetuned_test_predictions.npz",
        evaluation_dataset,
        config,
        output_dir,
        "finetuned",
        entity_column="business_id",
        entity_label="business",
    )
    naive_row, naive_path = save_naive_predictions(
        evaluation_dataset, config, output_dir, dataset_name=dataset
    )
    write_revenue_forecasts(
        naive_path,
        evaluation_dataset,
        config,
        output_dir,
        "seasonal_naive_7",
        entity_column="business_id",
        entity_label="business",
    )
    zero_shot_path = (
        PROJECT_ROOT
        / "outputs"
        / "online_retail_revenue"
        / "zero_shot"
        / "zero_shot_metrics.csv"
    )
    comparison_rows = [naive_row]
    if zero_shot_path.exists():
        comparison_rows.extend(pd.read_csv(zero_shot_path).to_dict("records"))
    comparison_rows.append(final_row)
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "comparison_metrics.csv", index=False
    )
    typer.echo(f"Best checkpoint: {best_checkpoint}")
    typer.echo(pd.DataFrame([final_row]).to_string(index=False))


@app.command("plot")
def plot_command(
    dataset: str = typer.Option(
        "online-retail-revenue", help="Only online-retail-revenue is supported."
    ),
    prediction_path: Path = typer.Option(
        PROJECT_ROOT / "outputs" / "evaluation" / "finetuned_test_predictions.npz"
    ),
    output_path: Path = typer.Option(PROJECT_ROOT / "outputs" / "plots" / "forecast.png"),
    store_id: str = typer.Option("ONLINE_RETAIL_UK"),
):
    """Plot the 28-day Online Retail revenue forecast interval."""
    dataset = _dataset_name(dataset)
    default_prediction = (
        PROJECT_ROOT
        / "outputs"
        / "online_retail_revenue"
        / "evaluation"
        / "finetuned_test_predictions.npz"
    )
    if prediction_path == PROJECT_ROOT / "outputs" / "evaluation" / "finetuned_test_predictions.npz":
        prediction_path = default_prediction
    if output_path == PROJECT_ROOT / "outputs" / "plots" / "forecast.png":
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "online_retail_revenue"
            / "plots"
            / f"{store_id}.png"
        )
    path = plot_revenue_prediction(prediction_path, output_path, store_id)
    typer.echo(f"Saved plot: {path}")


@app.command("gpu-check")
def gpu_check_command():
    """Print the minimum GPU checks to run on the remote server."""
    typer.echo(f"torch={torch.__version__}")
    typer.echo(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"device_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            typer.echo(f"device_{index}={torch.cuda.get_device_name(index)}")


@app.command("benchmark-models")
def benchmark_models_command(
    models: str = typer.Option(
        ",".join(SUPPORTED_MODELS),
        help="Comma-separated: random_forest,xgboost,catboost",
    ),
    data_path: Path = typer.Option(
        PROJECT_ROOT
        / "data"
        / "online_retail_revenue"
        / "daily_revenue.parquet"
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT / "outputs" / "online_retail_revenue" / "benchmarks"
    ),
    config_path: Optional[Path] = typer.Option(None),
):
    """Compare Chronos with leakage-free tree-model revenue forecasts."""
    requested = [value.strip() for value in models.split(",") if value.strip()]
    comparison = run_tabular_benchmarks(
        data_path=data_path,
        output_dir=output_dir,
        config=_config(config_path, "online-retail-revenue"),
        model_names=requested,
    )
    typer.echo(comparison.to_string(index=False))


@app.command("rolling-benchmark")
def rolling_benchmark_command(
    stage: str = typer.Option(
        "all", help="prepare, chronos, baselines, summarize, or all"
    ),
    folds: int = typer.Option(3, min=1),
    data_path: Path = typer.Option(
        PROJECT_ROOT
        / "data"
        / "online_retail_revenue"
        / "daily_revenue.parquet"
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT
        / "outputs"
        / "online_retail_revenue"
        / "rolling_benchmark"
    ),
    device: str = typer.Option("cuda", help="cuda or cpu; used by Chronos stage"),
    batch_size: int = typer.Option(4, min=1),
    config_path: Optional[Path] = typer.Option(None),
):
    """Run resumable three-fold Chronos and baseline model comparison."""
    try:
        ranking = run_rolling_benchmark(
            stage=stage,
            data_path=data_path,
            output_dir=output_dir,
            config=_config(config_path, "online-retail-revenue"),
            folds=folds,
            device=device,
            batch_size=batch_size,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if ranking is not None:
        typer.echo(ranking.to_string(index=False))


@app.command("context-horizon-benchmark")
def context_horizon_benchmark_command(
    smoke: bool = typer.Option(
        False, "--smoke", help="Run two folds with a reduced context grid."
    ),
    folds: int = typer.Option(
        0, min=0, help="0 uses 2 folds for smoke or 10 folds for full benchmark."
    ),
    data_path: Path = typer.Option(
        PROJECT_ROOT
        / "data"
        / "online_retail_revenue"
        / "daily_revenue.parquet"
    ),
    model_path: Path = typer.Option(
        PROJECT_ROOT
        / "outputs"
        / "release"
        / "chronos-online-retail-v0.1"
        / "inference"
    ),
    output_dir: Optional[Path] = typer.Option(None),
    device: str = typer.Option("cpu", help="cpu, cuda, or auto"),
    num_samples: int = typer.Option(20, min=1),
):
    """Compare forecast horizons against different history lengths."""
    if output_dir is None:
        mode = "smoke" if smoke else "full"
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "online_retail_revenue"
            / "context_horizon_benchmark"
            / mode
        )
    try:
        summary, recommendations = run_context_horizon_benchmark(
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            smoke=smoke,
            folds=folds or None,
            device=device,
            num_samples=num_samples,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo("\nSummary")
    typer.echo(summary.to_string(index=False))
    typer.echo("\nPreliminary recommendations")
    typer.echo(recommendations.to_string(index=False))


if __name__ == "__main__":
    app()
