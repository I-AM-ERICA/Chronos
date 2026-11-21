from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig, PROJECT_ROOT


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in run_dir.glob("checkpoint-*"):
        if path.name == "checkpoint-final":
            continue
        try:
            candidates.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            pass
    return max(candidates, default=(0, None))[1]


def find_latest_checkpoint(output_root: Path) -> Path | None:
    checkpoints = []
    for run_dir in output_root.glob("run-*"):
        checkpoint = _latest_checkpoint(run_dir)
        if checkpoint:
            checkpoints.append((checkpoint.stat().st_mtime, checkpoint))
    return max(checkpoints, default=(0, None))[1]


def run_finetuning(
    arrow_path: Path,
    output_root: Path,
    config: ExperimentConfig,
    max_steps: int | None = None,
    save_steps: int | None = None,
    batch_size: int = 32,
    resume_from_checkpoint: Path | None = None,
    cpu: bool = False,
) -> Path:
    max_steps = max_steps or config.max_steps
    save_steps = save_steps or config.save_steps
    output_root.mkdir(parents=True, exist_ok=True)
    batch_candidates = [b for b in (32, 16, 8, 4) if b <= batch_size]
    if not batch_candidates:
        batch_candidates = [4]

    last_error = ""
    for candidate in batch_candidates:
        accumulation = max(1, config.effective_batch_size // candidate)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train.py"),
            repr([str(arrow_path.resolve())]),
            "--context-length",
            str(config.context_length),
            "--prediction-length",
            str(config.prediction_length),
            "--min-past",
            str(config.min_past),
            "--max-steps",
            str(max_steps),
            "--save-steps",
            str(save_steps),
            "--log-steps",
            str(min(50, save_steps)),
            "--per-device-train-batch-size",
            str(candidate),
            "--gradient-accumulation-steps",
            str(accumulation),
            "--learning-rate",
            str(config.learning_rate),
            "--model-id",
            config.model_id,
            "--output-dir",
            str(output_root.resolve()),
            "--no-torch-compile",
            "--data-freq",
            config.frequency,
            "--seed",
            str(config.seed),
            "--dataloader-num-workers",
            "0" if cpu else "1",
        ]
        if cpu:
            command.extend(["--no-tf32", "--optim", "adamw_torch"])
        if resume_from_checkpoint is not None:
            command.extend(
                ["--resume-from-checkpoint", str(resume_from_checkpoint.resolve())]
            )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        captured_lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            captured_lines.append(line)
        return_code = process.wait()
        if return_code == 0:
            newest = max(output_root.glob("run-*"), key=lambda p: p.stat().st_mtime)
            return newest
        last_error = f"Training exited with code {return_code}."
        combined_output = "".join(captured_lines).lower()
        is_oom = "out of memory" in combined_output and "cuda" in combined_output
        if cpu or not is_oom:
            break
        print(
            f"Batch size {candidate} failed; retrying with a smaller batch while "
            "preserving effective batch size via gradient accumulation.",
            file=sys.stderr,
        )
    raise RuntimeError(last_error or "Training failed before it started.")
