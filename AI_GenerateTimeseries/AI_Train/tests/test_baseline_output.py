from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd


AI_TRAIN_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_TRAIN_DIR))

from baseline_output import (  # noqa: E402
    create_evaluation_layout,
    create_run_layout,
    finalize_evaluation,
    make_evaluation_id,
    make_run_id,
    validate_prediction_columns,
    write_case_prediction,
)


def test_ids_are_stable():
    assert make_run_id(7).endswith("_seed007")
    assert make_evaluation_id("housegan canonical", "test", "v1") == "eval_housegan_canonical_test_v1"


def test_run_and_evaluation_start_invalid(tmp_path):
    run = create_run_layout(
        tmp_path / "outputs",
        method_id="Method_Test",
        seed=42,
        dataset_id="housegan_canonical_imagebase_split_v1",
        config={"seed": 42},
    )
    assert (run.root / "run_manifest.json").exists()
    assert (run.checkpoints / "checkpoint_manifest.json").exists()
    assert (run.framing_previews / "framing_manifest.json").exists()
    manifest = json.loads((run.root / "run_manifest.json").read_text())
    assert manifest["research_valid"] is False

    checkpoint = run.checkpoints / "best_model.pth"
    checkpoint.write_bytes(b"test checkpoint")
    evaluation = create_evaluation_layout(
        run.root,
        method_id="Method_Test",
        dataset_id="housegan_canonical_imagebase_split_v1",
        split="test",
        protocol_version="v1",
        checkpoint_path=checkpoint,
        evaluation_config={"obs_len": 5, "frame_stride": 8},
    )
    eval_manifest = json.loads((evaluation.root / "evaluation_manifest.json").read_text())
    assert eval_manifest["research_valid"] is False

    frame = pd.DataFrame(
        [{
            "case_id": "case_1", "split": "test", "frame": 8,
            "agent_id": 1, "pos_x": 1.0, "pos_y": 2.0, "is_active": True,
        }]
    )
    output = write_case_prediction(evaluation, "case_1", frame)
    assert output.exists()

    valid = finalize_evaluation(
        evaluation,
        case_ids=["case_1"],
        floorplan_ids=["plan_1"],
        compatibility_ok=True,
        canonical_test_required=True,
    )
    assert valid is False
    eval_manifest = json.loads((evaluation.root / "evaluation_manifest.json").read_text())
    assert "expected 862 test cases" in eval_manifest["invalid_reason"]


def test_prediction_schema_rejects_normalized_or_incomplete_output():
    try:
        validate_prediction_columns(["case_id", "frame", "pos_x", "pos_y"])
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("incomplete prediction schema was accepted")

