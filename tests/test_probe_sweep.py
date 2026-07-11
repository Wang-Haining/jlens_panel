from jlens_panel.calibration import (
    NULL_RENDERING_POLICY,
    POSITION_RESOLVER_SCHEMA,
    NullCalibration,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES
from jlens_panel.sweep.probe_sweep import RESULT_COLUMNS, _gate_status, run_probe_sweep


def test_probe_sweep_emits_all_rows_and_three_gate_outcomes() -> None:
    candidates = tuple(f"candidate{index:02d}" for index in range(16))
    token_ids = {candidate: 100 + index for index, candidate in enumerate(candidates)}
    train_targets = tuple(
        token_ids[candidate] for candidate in candidates for _repeat in range(2)
    )
    dev_targets = tuple(token_ids[candidate] for candidate in candidates)
    train_rows = tuple(
        tuple(float(column == index) for column in range(16))
        for index in range(16)
        for _repeat in range(2)
    )
    dev_rows = tuple(
        tuple(float(column == index) for column in range(16)) for index in range(16)
    )
    train_residuals = {
        (position, 0): train_rows for position in ALL_POSITION_NAMES
    }
    dev_residuals = {(position, 0): dev_rows for position in ALL_POSITION_NAMES}

    jlens_rows = tuple(
        tuple(
            (5.0 if candidate_index == 0 else 0.0)
            + (10.0 if candidate_index == gold_index else 0.0)
            for candidate_index in range(16)
        )
        for gold_index in range(16)
    )
    logit_rows = tuple(tuple(0.0 for _candidate in candidates) for _gold in candidates)
    dev_scores = {
        (method, position, 0): (jlens_rows if method == "jlens" else logit_rows)
        for method in ("jlens", "logit_lens")
        for position in ALL_POSITION_NAMES
    }
    provenance = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "config_sha256": "1" * 64,
        "git_revision": "2" * 40,
        "lens_sha256": "b" * 64,
        "corpus_sha256": "c" * 64,
        "sample_sha256": "d" * 64,
        "sample_seed": 20260711,
        "null_prompt_count": 200,
        "candidate_token_ids": token_ids,
        "source_layers": [0],
        "max_seq_len": 512,
        "decode_steps": [1, 2, 4, 8],
        "ddof": 0,
        "resolver_schema": POSITION_RESOLVER_SCHEMA,
        "rendering_policy": dict(NULL_RENDERING_POLICY),
    }
    calibrations = {
        (method, position, 0): NullCalibration(
            method=method,
            position_type=position,
            layer=0,
            candidate_means={
                candidate: (5.0 if method == "jlens" and index == 0 else 0.0)
                for index, candidate in enumerate(candidates)
            },
            candidate_stds={candidate: 1.0 for candidate in candidates},
            n_null_prompts=200,
            provenance=provenance,
        )
        for method in ("jlens", "logit_lens")
        for position in ALL_POSITION_NAMES
    }
    gates = {
        "g_info_min_probe_top1": 0.25,
        "g_const_max_label_share": 0.5,
        "g_const_min_entropy_bits": 1.5,
        "g_lens_min_probe_ratio": 0.5,
    }

    rows, heatmap, gate_summary = run_probe_sweep(
        candidates=candidates,
        candidate_token_ids=token_ids,
        layers=(0,),
        train_residuals=train_residuals,
        dev_residuals=dev_residuals,
        dev_scores=dev_scores,
        train_target_token_ids=train_targets,
        dev_target_token_ids=dev_targets,
        dev_example_ids=tuple(f"dev-{index:02d}" for index in range(16)),
        calibrations=calibrations,
        c_values=(1.0, 0.01, 0.1),
        gates=gates,
    )

    assert len(rows) == 8 * (3 + 4)
    assert tuple(rows[0].to_dict()) == RESULT_COLUMNS
    assert {row.method for row in rows} == {
        "probe",
        "jlens_raw",
        "jlens_calibrated",
        "logitlens_raw",
        "logitlens_calibrated",
    }
    assert gate_summary["best_probe_cell"] == {
        "position": "template_tail",
        "layer": 0,
        "selected_C": 1.0,
        "dev_top1": 1.0,
    }
    assert gate_summary["g_info"]["status"] == "pass"
    assert gate_summary["g_const"]["status"] == "pass"
    assert gate_summary["g_const"]["max_label_share"] == 1.0 / 16.0
    assert gate_summary["g_const"]["entropy_bits"] == 4.0
    assert gate_summary["g_lens"]["status"] == "pass"
    assert heatmap["probe_dev_top1"]["template_tail"]["0"] == 1.0


def test_exact_gate_threshold_is_reported_as_ambiguous() -> None:
    assert _gate_status(0.25, 0.25) == "ambiguous_equal_threshold"
