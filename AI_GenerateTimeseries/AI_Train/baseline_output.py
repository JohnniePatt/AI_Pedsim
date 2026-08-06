"""Shared, research-safe output layout for trajectory baselines.

This module implements the contract documented in ``output baseline.md``.
It deliberately starts every run and evaluation as ``research_valid=false``;
only :func:`finalize_evaluation` may promote a complete compatible evaluation.
Legacy readers remain the responsibility of each caller/UI.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


COMMON_PREDICTION_COLUMNS = (
    "case_id",
    "split",
    "frame",
    "agent_id",
    "pos_x",
    "pos_y",
    "is_active",
)
CANONICAL_HOUSEGAN_TEST_CASES = 862
CANONICAL_HOUSEGAN_TEST_PLANS = 117


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id(seed: int, when: datetime | None = None) -> str:
    timestamp = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"run_{timestamp:%Y%m%dT%H%M%SZ}_seed{int(seed):03d}"


def make_evaluation_id(dataset_id: str, split: str, protocol_version: str = "v1") -> str:
    safe_dataset = _safe_token(dataset_id)
    return f"eval_{safe_dataset}_{_safe_token(split)}_{_safe_token(protocol_version)}"


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())
    return token.strip("_") or "unknown"


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: pathlib.Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(project_root: pathlib.Path | None) -> dict[str, Any]:
    if project_root is None:
        return {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


@dataclass(frozen=True)
class RunLayout:
    root: pathlib.Path
    checkpoints: pathlib.Path
    logs: pathlib.Path
    diagnostics: pathlib.Path
    validation_samples: pathlib.Path
    framing_previews: pathlib.Path
    evaluations: pathlib.Path


@dataclass(frozen=True)
class EvaluationLayout:
    root: pathlib.Path
    predictions: pathlib.Path
    metrics: pathlib.Path
    statistics: pathlib.Path
    previews: pathlib.Path
    reports: pathlib.Path


def create_run_layout(
    outputs_root: str | pathlib.Path,
    *,
    method_id: str,
    seed: int,
    dataset_id: str,
    config: dict[str, Any],
    method_display_name: str | None = None,
    method_family: str | None = None,
    dataset_manifest: str | pathlib.Path | None = None,
    project_root: str | pathlib.Path | None = None,
    run_id: str | None = None,
) -> RunLayout:
    """Create a new immutable run directory and its required manifests."""
    outputs_root = pathlib.Path(outputs_root).resolve()
    requested_id = run_id or make_run_id(seed)
    root = outputs_root / requested_id
    if root.exists():
        suffix = 1
        while (outputs_root / f"{requested_id}_{suffix:02d}").exists():
            suffix += 1
        root = outputs_root / f"{requested_id}_{suffix:02d}"

    layout = RunLayout(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        diagnostics=root / "diagnostics",
        validation_samples=root / "diagnostics" / "validation_samples",
        framing_previews=root / "framing_previews",
        evaluations=root / "evaluations",
    )
    for directory in (
        layout.checkpoints,
        layout.logs,
        layout.validation_samples,
        layout.framing_previews,
        layout.evaluations,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = pathlib.Path(dataset_manifest).resolve() if dataset_manifest else None
    snapshot_path = root / "dataset_manifest_snapshot.csv"
    if manifest_path and manifest_path.is_file():
        shutil.copy2(manifest_path, snapshot_path)

    _atomic_json(
        root / "run_manifest.json",
        {
            "schema_version": "baseline-output-v1",
            "run_id": root.name,
            "method_id": method_id,
            "seed": int(seed),
            "status": "running",
            "dataset_id": dataset_id,
            "train_split": "train",
            "validation_split": "val",
            "research_valid": False,
            "invalid_reason": "training run has not completed and no final evaluation has passed the validity gate",
            "created_at_utc": utc_now(),
        },
    )
    _atomic_json(
        root / "method_manifest.json",
        {
            "method_id": method_id,
            "display_name": method_display_name or method_id,
            "method_family": method_family or "unspecified",
            "output_schema_version": "baseline-output-v1",
        },
    )
    _atomic_json(root / "config_train.json", config)
    _atomic_json(
        root / "environment.json",
        {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    )
    provenance = _git_provenance(pathlib.Path(project_root).resolve() if project_root else None)
    provenance.update({"created_at_utc": utc_now()})
    _atomic_json(root / "code_provenance.json", provenance)
    _atomic_json(
        layout.checkpoints / "checkpoint_manifest.json",
        {"schema_version": "baseline-output-v1", "checkpoints": [], "updated_at_utc": utc_now()},
    )
    _atomic_json(
        layout.framing_previews / "framing_manifest.json",
        {
            "purpose": "framing_only",
            "research_valid": False,
            "invalid_reason": "preview subset; not standardized final evaluation",
            "method_id": method_id,
            "run_id": root.name,
        },
    )
    return layout


def update_checkpoint_manifest(run_dir: str | pathlib.Path, checkpoint: str | pathlib.Path, role: str) -> None:
    run_dir = pathlib.Path(run_dir)
    checkpoint = pathlib.Path(checkpoint)
    manifest_path = run_dir / "checkpoints" / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    entries = [entry for entry in manifest.get("checkpoints", []) if entry.get("role") != role]
    entries.append(
        {
            "role": role,
            "path": str(checkpoint.relative_to(run_dir)),
            "sha256": sha256_file(checkpoint),
            "updated_at_utc": utc_now(),
        }
    )
    manifest.update({"schema_version": "baseline-output-v1", "checkpoints": entries, "updated_at_utc": utc_now()})
    _atomic_json(manifest_path, manifest)


def mark_run_completed(run_dir: str | pathlib.Path) -> None:
    path = pathlib.Path(run_dir) / "run_manifest.json"
    manifest = read_json(path)
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            # Training completion alone never establishes research validity.
            "research_valid": False,
            "invalid_reason": "no final evaluation has passed the validity gate",
        }
    )
    _atomic_json(path, manifest)


def create_evaluation_layout(
    run_dir: str | pathlib.Path,
    *,
    method_id: str,
    dataset_id: str,
    split: str,
    protocol_version: str,
    checkpoint_path: str | pathlib.Path,
    evaluation_config: dict[str, Any],
    dataset_manifest: str | pathlib.Path | None = None,
    constraint_mode: str = "none",
    stochastic_sample_count: int = 1,
    compatibility_ok: bool = True,
    invalid_reason: str | None = None,
) -> EvaluationLayout:
    run_dir = pathlib.Path(run_dir).resolve()
    checkpoint_path = pathlib.Path(checkpoint_path).resolve()
    evaluation_id = make_evaluation_id(dataset_id, split, protocol_version)
    root = run_dir / "evaluations" / evaluation_id
    layout = EvaluationLayout(
        root=root,
        predictions=root / "predictions",
        metrics=root / "metrics",
        statistics=root / "statistics",
        previews=root / "previews",
        reports=root / "reports",
    )
    for directory in (layout.predictions, layout.metrics, layout.statistics, layout.previews, layout.reports):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = pathlib.Path(dataset_manifest).resolve() if dataset_manifest else None
    reason = invalid_reason or "evaluation incomplete; validity gate has not been run"
    if not compatibility_ok and not invalid_reason:
        reason = "checkpoint/dataset compatibility check failed"
    _atomic_json(
        root / "evaluation_manifest.json",
        {
            "schema_version": "baseline-output-v1",
            "evaluation_id": evaluation_id,
            "method_id": method_id,
            "run_id": run_dir.name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "dataset_id": dataset_id,
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "split": split,
            "case_count": 0,
            "floorplan_count": 0,
            "observation_frames": evaluation_config.get("obs_len"),
            "prediction_horizon": evaluation_config.get("prediction_horizon", evaluation_config.get("max_seq_len")),
            "frame_stride": evaluation_config.get("frame_stride"),
            "coordinate_system": "world_metres",
            "constraint_mode": constraint_mode,
            "stochastic_sample_count": int(stochastic_sample_count),
            "compatibility_ok": bool(compatibility_ok),
            "research_valid": False,
            "invalid_reason": reason,
            "created_at_utc": utc_now(),
        },
    )
    _atomic_json(root / "evaluation_config.json", evaluation_config)
    _atomic_json(
        root / "checkpoint_ref.json",
        {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
    )
    _atomic_json(
        root / "dataset_ref.json",
        {
            "dataset_id": dataset_id,
            "manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "split": split,
        },
    )
    _atomic_json(
        root / "ground_truth_ref.json",
        {"dataset_id": dataset_id, "split": split, "embedded_in_predictions": False},
    )
    return layout


def validate_prediction_columns(columns: Iterable[str], *, stochastic: bool = False) -> None:
    available = set(columns)
    required = set(COMMON_PREDICTION_COLUMNS)
    if stochastic:
        required.update({"sample_id", "sample_seed"})
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"prediction output is missing required columns: {', '.join(missing)}")


def write_case_prediction(
    layout: EvaluationLayout,
    case_id: str,
    frame: Any,
    *,
    variant: str = "raw",
    stochastic: bool = False,
) -> pathlib.Path:
    """Validate and write one case DataFrame without mutating raw coordinates."""
    validate_prediction_columns(frame.columns, stochastic=stochastic)
    case_dir = layout.predictions / _safe_token(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / f"prediction_{_safe_token(variant)}.parquet"
    frame.to_parquet(output, index=False)
    return output


def finalize_evaluation(
    layout: EvaluationLayout,
    *,
    case_ids: Iterable[str],
    floorplan_ids: Iterable[str],
    compatibility_ok: bool,
    canonical_test_required: bool,
    additional_failures: Iterable[str] = (),
) -> bool:
    """Run the objective part of the validity gate and update the manifest."""
    manifest_path = layout.root / "evaluation_manifest.json"
    manifest = read_json(manifest_path)
    cases = {str(value) for value in case_ids}
    plans = {str(value) for value in floorplan_ids}
    failures = [str(item) for item in additional_failures if str(item).strip()]
    if not compatibility_ok:
        failures.append("checkpoint/dataset compatibility check failed")
    if not cases:
        failures.append("no prediction cases were written")

    written_cases = {
        case_dir.name
        for case_dir in layout.predictions.iterdir()
        if case_dir.is_dir()
        and (
            (case_dir / "prediction_raw.parquet").exists()
            or (case_dir / "prediction_constrained.parquet").exists()
        )
    }
    expected_safe_cases = {_safe_token(case_id) for case_id in cases}
    if written_cases != expected_safe_cases:
        failures.append(
            f"prediction artifact/case mismatch: expected {len(expected_safe_cases)}, found {len(written_cases)}"
        )
    if not (layout.metrics / "summary_metrics.csv").exists():
        failures.append("metrics/summary_metrics.csv is missing")

    run_dir = layout.root.parents[1]
    run_manifest = read_json(run_dir / "run_manifest.json")
    if not run_manifest:
        failures.append("run_manifest.json is missing (legacy run)")
    else:
        if run_manifest.get("status") != "completed":
            failures.append("training run is not marked completed")
        if run_manifest.get("dataset_id") != manifest.get("dataset_id"):
            failures.append("run/evaluation dataset_id mismatch")
    if not (run_dir / "code_provenance.json").exists():
        failures.append("code_provenance.json is missing")

    dataset_ref = read_json(layout.root / "dataset_ref.json")
    referenced_manifest = pathlib.Path(dataset_ref["manifest_path"]) if dataset_ref.get("manifest_path") else None
    snapshot = run_dir / "dataset_manifest_snapshot.csv"
    if canonical_test_required:
        if referenced_manifest is None or not referenced_manifest.exists():
            failures.append("canonical dataset manifest is missing")
        if not snapshot.exists():
            failures.append("training dataset manifest snapshot is missing")
        elif referenced_manifest and sha256_file(snapshot) != sha256_file(referenced_manifest):
            failures.append("training/evaluation dataset manifest hash mismatch")
        if referenced_manifest and referenced_manifest.exists():
            try:
                split_plans: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
                with referenced_manifest.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        split_name = str(row.get("split", ""))
                        plan_name = str(row.get("plan_name", ""))
                        if split_name in split_plans and plan_name:
                            split_plans[split_name].add(plan_name)
                if split_plans["train"] & split_plans["test"]:
                    failures.append("plan overlap detected between train and test")
                if split_plans["val"] & split_plans["test"]:
                    failures.append("plan overlap detected between val and test")
            except (OSError, csv.Error):
                failures.append("canonical dataset manifest could not be validated")

    constraint_mode = str(manifest.get("constraint_mode", "none"))
    if constraint_mode == "none":
        if written_cases and any(
            not (layout.predictions / case_name / "prediction_raw.parquet").exists()
            for case_name in written_cases
        ):
            failures.append("raw prediction is missing for an unconstrained evaluation")
    elif written_cases and any(
        not (layout.predictions / case_name / "action_trace.parquet").exists()
        for case_name in written_cases
    ):
        failures.append("constrained evaluation is missing action_trace.parquet")
    if canonical_test_required:
        if manifest.get("split") != "test":
            failures.append("canonical final evaluation must use the test split")
        if len(cases) != CANONICAL_HOUSEGAN_TEST_CASES:
            failures.append(
                f"expected {CANONICAL_HOUSEGAN_TEST_CASES} test cases, found {len(cases)}"
            )
        if len(plans) != CANONICAL_HOUSEGAN_TEST_PLANS:
            failures.append(
                f"expected {CANONICAL_HOUSEGAN_TEST_PLANS} test floorplans, found {len(plans)}"
            )
    valid = not failures
    manifest.update(
        {
            "case_count": len(cases),
            "floorplan_count": len(plans),
            "research_valid": valid,
            "invalid_reason": None if valid else "; ".join(dict.fromkeys(failures)),
            "completed_at_utc": utc_now(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return valid


def resolve_checkpoint(run_dir: str | pathlib.Path, names: Iterable[str] = ("best_model.pth", "latest_model.pth")) -> pathlib.Path | None:
    """Resolve canonical checkpoints first, then legacy ``weights`` files."""
    run_dir = pathlib.Path(run_dir)
    for directory_name in ("checkpoints", "weights"):
        directory = run_dir / directory_name
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None
