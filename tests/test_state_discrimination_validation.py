import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_state_discrimination_inputs import main as validate_main, validate_inputs


class StateDiscriminationValidationTests(unittest.TestCase):
    def _write_labels(self, path, rows):
        fields = ["session_id", "label", "event_token", "event_turn", "event_type", "split", "verifier_name"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_raw(self, path, session_id="s1", latent_occupancy=0.2, tokens=5):
        path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "turns": [
                        {
                            "turn_index": 0,
                            "token_count": tokens,
                            "tokens": [
                                {"entropy": 0.1, "top1_logprob": -0.2, "gap": 0.3}
                                for _ in range(tokens)
                            ],
                            "summary": {"latent_occupancy": latent_occupancy},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_valid_inputs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw_neg = tmp_path / "raw_neg.json"
            raw_pos = tmp_path / "raw_pos.json"
            self._write_labels(labels, [
                {"session_id": "s_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "", "verifier_name": "auto"},
                {"session_id": "s_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "", "verifier_name": "auto"},
            ])
            self._write_raw(raw_neg, session_id="s_neg", latent_occupancy=0.1)
            self._write_raw(raw_pos, session_id="s_pos", latent_occupancy=0.8)
            report = validate_inputs(labels, [raw_neg, raw_pos])
            self.assertTrue(report["valid"])
            self.assertEqual(report["valid_session_count"], 2)

    def test_missing_label_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw = tmp_path / "raw.json"
            self._write_labels(labels, [{"session_id": "other", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "test", "verifier_name": "auto"}])
            self._write_raw(raw)
            report = validate_inputs(labels, [raw])
            self.assertFalse(report["valid"])
            self.assertTrue(any("missing label" in error for error in report["errors"]))

    def test_event_token_out_of_range_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw = tmp_path / "raw.json"
            self._write_labels(labels, [{"session_id": "s1", "label": "1", "event_token": "20", "event_turn": "0", "event_type": "failure", "split": "test", "verifier_name": "auto"}])
            self._write_raw(raw, tokens=5)
            report = validate_inputs(labels, [raw])
            self.assertFalse(report["valid"])
            self.assertTrue(any("outside raw token range" in error for error in report["errors"]))

    def test_missing_primary_score_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw = tmp_path / "raw.json"
            self._write_labels(labels, [{"session_id": "s1", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "test", "verifier_name": "auto"}])
            raw.write_text(json.dumps({"session_id": "s1", "turns": [{"turn_index": 0, "token_count": 5, "tokens": [{"entropy": 0.1, "top1_logprob": -0.2, "gap": 0.3} for _ in range(5)], "summary": {}}]}), encoding="utf-8")
            report = validate_inputs(labels, [raw])
            self.assertFalse(report["valid"])
            self.assertTrue(any("primary score" in error for error in report["errors"]))


    def test_duplicate_session_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw = tmp_path / "raw.json"
            self._write_labels(labels, [
                {"session_id": "s1", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "", "verifier_name": "auto"},
                {"session_id": "s1", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "", "verifier_name": "auto"},
            ])
            self._write_raw(raw)
            report = validate_inputs(labels, [raw])
            self.assertFalse(report["valid"])
            self.assertTrue(any("duplicate session_id" in error for error in report["errors"]))

    def test_requires_positive_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw = tmp_path / "raw.json"
            self._write_labels(labels, [
                {"session_id": "s1", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "", "verifier_name": "auto"},
            ])
            self._write_raw(raw)
            report = validate_inputs(labels, [raw])
            self.assertFalse(report["valid"])
            self.assertTrue(any("positive and one negative" in error for error in report["errors"]))

    def test_split_train_and_test_need_both_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw_train_neg = tmp_path / "train_neg.json"
            raw_train_pos = tmp_path / "train_pos.json"
            raw_test_neg = tmp_path / "test_neg.json"
            raw_test_pos = tmp_path / "test_pos.json"
            self._write_labels(labels, [
                {"session_id": "train_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "train", "verifier_name": "auto"},
                {"session_id": "train_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "train", "verifier_name": "auto"},
                {"session_id": "test_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "test", "verifier_name": "auto"},
                {"session_id": "test_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "test", "verifier_name": "auto"},
            ])
            self._write_raw(raw_train_neg, session_id="train_neg", latent_occupancy=0.1)
            self._write_raw(raw_train_pos, session_id="train_pos", latent_occupancy=0.9)
            self._write_raw(raw_test_neg, session_id="test_neg", latent_occupancy=0.2)
            self._write_raw(raw_test_pos, session_id="test_pos", latent_occupancy=0.8)
            report = validate_inputs(labels, [raw_train_neg, raw_train_pos, raw_test_neg, raw_test_pos])
            self.assertTrue(report["valid"])

            self._write_labels(labels, [
                {"session_id": "train_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "train", "verifier_name": "auto"},
                {"session_id": "test_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "test", "verifier_name": "auto"},
                {"session_id": "test_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "test", "verifier_name": "auto"},
            ])
            report = validate_inputs(labels, [raw_train_neg, raw_test_neg, raw_test_pos])
            self.assertFalse(report["valid"])
            self.assertTrue(any("split `train`" in error for error in report["errors"]))

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw_neg = tmp_path / "raw_neg.json"
            raw_pos = tmp_path / "raw_pos.json"
            out = tmp_path / "out"
            self._write_labels(labels, [
                {"session_id": "s_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "", "verifier_name": "auto"},
                {"session_id": "s_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "", "verifier_name": "auto"},
            ])
            self._write_raw(raw_neg, session_id="s_neg", latent_occupancy=0.1)
            self._write_raw(raw_pos, session_id="s_pos", latent_occupancy=0.8)
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["validate_state_discrimination_inputs.py", "--labels", str(labels), "--inputs", str(raw_neg), str(raw_pos), "--output-dir", str(out)]
                self.assertEqual(validate_main(), 0)
            finally:
                sys.argv = old_argv
            self.assertTrue((out / "state_discrimination_input_validation.json").exists())
            self.assertTrue((out / "state_discrimination_input_validation.md").exists())

    def test_flat_primary_score_fails_sanity_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = tmp_path / "labels.csv"
            raw_neg = tmp_path / "raw_neg.json"
            raw_pos = tmp_path / "raw_pos.json"
            self._write_labels(labels, [
                {"session_id": "s_neg", "label": "0", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "", "verifier_name": "auto"},
                {"session_id": "s_pos", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "failure", "split": "", "verifier_name": "auto"},
            ])
            self._write_raw(raw_neg, session_id="s_neg", latent_occupancy=0.2)
            self._write_raw(raw_pos, session_id="s_pos", latent_occupancy=0.2)
            report = validate_inputs(labels, [raw_neg, raw_pos])
            self.assertFalse(report["valid"])
            self.assertTrue(any("flat across sessions" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
