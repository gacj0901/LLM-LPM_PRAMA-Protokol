#!/usr/bin/env python
"""Validate labels.csv and certified-kernel raw.json study inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from aptadynamic_llm.evaluation.state_discrimination_metrics import (
    BASELINE_SCORE_FIELDS,
    PRIMARY_PRAMA_SCORE,
    PRAMA_SCORE_FIELDS,
    REQUIRED_LABEL_FIELDS,
)

VALID_LABELS = {"0", "1", "true", "false", "fail", "failed", "pathological", "positive", "healthy", "negative"}
VALID_SPLITS = {"", "train", "calibration", "calib", "test"}
BASELINE_TOKEN_FIELDS = ("entropy", "top1_logprob", "gap")
POSITIVE_LABELS = {"1", "true", "fail", "failed", "pathological", "positive"}
NEGATIVE_LABELS = {"0", "false", "healthy", "negative"}


def _label_to_binary(value: str | None) -> int | None:
    normalized = (value or "").strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    return None


def _read_labels(path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return [], [f"labels file not found: {path}"], warnings
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], ["labels.csv is empty or missing a header"], warnings
        missing = REQUIRED_LABEL_FIELDS.difference(reader.fieldnames)
        if missing:
            errors.append(f"labels.csv missing required fields: {', '.join(sorted(missing))}")
        rows = [row for row in reader if row.get("session_id")]
    if not rows:
        errors.append("labels.csv contains no labeled sessions")
    counts = Counter(row.get("session_id", "") for row in rows)
    duplicates = sorted(session_id for session_id, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate session_id values in labels.csv: {', '.join(duplicates)}")
    binary_labels: list[int] = []
    split_labels: dict[str, list[int]] = {"train": [], "test": []}
    has_explicit_split = False
    for row in rows:
        session_id = row.get("session_id", "<missing>")
        label = (row.get("label") or "").strip().lower()
        binary_label = _label_to_binary(label)
        if label not in VALID_LABELS or binary_label is None:
            errors.append(f"{session_id}: invalid label `{row.get('label')}`")
        else:
            binary_labels.append(binary_label)
        split = (row.get("split") or "").strip().lower()
        if split not in VALID_SPLITS:
            errors.append(f"{session_id}: invalid split `{row.get('split')}`")
        elif split:
            has_explicit_split = True
            normalized_split = "train" if split in {"train", "calibration", "calib"} else "test"
            if binary_label is not None:
                split_labels[normalized_split].append(binary_label)
        for field in ("event_token", "event_turn"):
            try:
                int(float(row.get(field, "")))
            except (TypeError, ValueError):
                errors.append(f"{session_id}: {field} must be numeric")
        if row.get("event_token") and row.get("event_type") and str(row.get("event_type")).lower() in {"final", "final_answer"}:
            warnings.append(f"{session_id}: final-answer event type may make lead time a final-outcome proxy")
    if binary_labels and (0 not in binary_labels or 1 not in binary_labels):
        errors.append("labels.csv must contain at least one positive and one negative label")
    if has_explicit_split:
        for split_name in ("train", "test"):
            labels_for_split = split_labels[split_name]
            if not labels_for_split:
                errors.append(f"split `{split_name}` has no examples")
            elif 0 not in labels_for_split or 1 not in labels_for_split:
                errors.append(f"split `{split_name}` must contain at least one positive and one negative label")
    return rows, errors, warnings


def _session_id(raw: dict[str, Any], path: Path) -> str:
    return str(raw.get("session_id") or path.stem)


def _turn_token_count(turn: dict[str, Any]) -> int:
    return int(turn.get("token_count") or len(turn.get("tokens") or []) or 0)


def _flatten_turns(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative = 0
    for index, turn in enumerate(raw.get("turns") or []):
        token_count = _turn_token_count(turn)
        cumulative += token_count
        summary = turn.get("metrics_summary") or turn.get("summary") or turn
        row = dict(summary)
        row["turn_index"] = int(turn.get("turn_index") or index)
        row["token_position"] = cumulative
        row["token_count"] = token_count
        row["tokens"] = list(turn.get("tokens") or [])
        rows.append(row)
    return rows


def _chosen_row(turns: list[dict[str, Any]], event_token: int) -> dict[str, Any]:
    chosen = turns[0]
    for turn in turns:
        if int(turn.get("token_position", 0)) <= event_token:
            chosen = turn
    return chosen


def validate_inputs(labels_path: Path, input_paths: list[Path], primary_score: str = PRIMARY_PRAMA_SCORE) -> dict[str, Any]:
    if primary_score not in PRAMA_SCORE_FIELDS:
        raise ValueError(f"unknown primary PRAMA score: {primary_score}")
    label_rows, errors, warnings = _read_labels(labels_path)
    labels_by_session = {row["session_id"]: row for row in label_rows if row.get("session_id")}
    seen_inputs: set[str] = set()
    sessions: list[dict[str, Any]] = []
    primary_values: list[float] = []

    if not input_paths:
        errors.append("at least one --inputs raw.json file is required")

    for path in input_paths:
        session_result: dict[str, Any] = {
            "path": str(path),
            "session_id": None,
            "valid": False,
            "errors": [],
            "warnings": [],
            "turn_count": 0,
            "final_token": 0,
            "event_token": None,
            "event_turn": None,
            "final_outcome_proxy": False,
            "primary_score_value": None,
        }
        if not path.exists():
            session_result["errors"].append(f"input file not found: {path}")
            sessions.append(session_result)
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            session_result["errors"].append(f"invalid JSON: {exc}")
            sessions.append(session_result)
            continue
        session_id = _session_id(raw, path)
        session_result["session_id"] = session_id
        seen_inputs.add(session_id)
        label = labels_by_session.get(session_id)
        if not label:
            session_result["errors"].append(f"missing label for session_id {session_id}")
            sessions.append(session_result)
            continue
        turns = _flatten_turns(raw)
        session_result["turn_count"] = len(turns)
        if not turns:
            session_result["errors"].append("raw.json has no turns")
            sessions.append(session_result)
            continue
        final_token = int(turns[-1].get("token_position") or 0)
        final_turn = max(int(turn.get("turn_index", 0)) for turn in turns)
        session_result["final_token"] = final_token
        try:
            event_token = int(float(label["event_token"]))
            event_turn = int(float(label["event_turn"]))
        except (TypeError, ValueError):
            session_result["errors"].append("event_token/event_turn must be numeric")
            sessions.append(session_result)
            continue
        session_result["event_token"] = event_token
        session_result["event_turn"] = event_turn
        if event_token < 0 or event_token > final_token:
            session_result["errors"].append(f"event_token {event_token} outside raw token range 0..{final_token}")
        if event_turn < 0 or event_turn > final_turn:
            session_result["errors"].append(f"event_turn {event_turn} outside raw turn range 0..{final_turn}")
        session_result["final_outcome_proxy"] = event_token >= final_token
        if session_result["final_outcome_proxy"]:
            session_result["warnings"].append("event_token equals/exceeds final token; lead time is a final-outcome proxy")
        chosen = _chosen_row(turns, min(event_token, final_token))
        if primary_score not in chosen:
            session_result["errors"].append(f"primary score `{primary_score}` missing at causal evaluation row")
        else:
            try:
                primary_value = float(chosen[primary_score])
            except (TypeError, ValueError):
                primary_value = float("nan")
            if not math.isfinite(primary_value):
                session_result["errors"].append(f"primary score `{primary_score}` is not finite")
            else:
                session_result["primary_score_value"] = primary_value
                primary_values.append(primary_value)
        token_fields_present = {field: 0 for field in BASELINE_TOKEN_FIELDS}
        total_tokens_with_payload = 0
        for turn in turns:
            for token in turn.get("tokens") or []:
                total_tokens_with_payload += 1
                for field in BASELINE_TOKEN_FIELDS:
                    if field in token and token.get(field) is not None:
                        token_fields_present[field] += 1
        if total_tokens_with_payload == 0:
            session_result["warnings"].append("no token payloads found; logprob baselines may be zero-filled")
        for field, count in token_fields_present.items():
            if total_tokens_with_payload and count == 0:
                session_result["warnings"].append(f"token field `{field}` missing from all token payloads")
        session_result["valid"] = not session_result["errors"]
        sessions.append(session_result)

    missing_inputs = sorted(set(labels_by_session).difference(seen_inputs))
    for session_id in missing_inputs:
        warnings.append(f"label exists without matching input raw.json: {session_id}")

    primary_range = max(primary_values) - min(primary_values) if primary_values else None
    if len(primary_values) >= 2 and primary_range <= 1e-12:
        errors.append(
            f"primary score `{primary_score}` is flat across sessions; sanity wiring failed"
        )

    for session in sessions:
        errors.extend(f"{session.get('session_id') or session['path']}: {message}" for message in session["errors"])
        warnings.extend(f"{session.get('session_id') or session['path']}: {message}" for message in session["warnings"])

    return {
        "valid": not errors,
        "primary_score": primary_score,
        "labels_path": str(labels_path),
        "input_count": len(input_paths),
        "labeled_session_count": len(labels_by_session),
        "valid_session_count": sum(1 for session in sessions if session["valid"]),
        "primary_score_range": primary_range,
        "errors": errors,
        "warnings": warnings,
        "sessions": sessions,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# State-Discrimination Input Validation Report",
        "",
        f"- valid: `{report['valid']}`",
        f"- primary_score: `{report['primary_score']}`",
        f"- input_count: `{report['input_count']}`",
        f"- labeled_session_count: `{report['labeled_session_count']}`",
        f"- valid_session_count: `{report['valid_session_count']}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in report["errors"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"] or ["None"])
    lines.extend(["", "## Sessions", ""])
    for session in report["sessions"]:
        lines.append(
            f"- `{session.get('session_id')}` valid={session['valid']} turns={session['turn_count']} "
            f"final_token={session['final_token']} event_token={session['event_token']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--primary-score", default=PRIMARY_PRAMA_SCORE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/state_discrimination_validation"))
    args = parser.parse_args()

    report = validate_inputs(args.labels, args.inputs, args.primary_score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "state_discrimination_input_validation.json"
    md_path = args.output_dir / "state_discrimination_input_validation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not report["valid"]:
        print("validation failed")
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
