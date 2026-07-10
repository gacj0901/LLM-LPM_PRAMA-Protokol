import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from aptadynamic_llm.evaluation.state_discrimination_metrics import (
    BASELINE_SCORE_FIELDS,
    auroc,
    baseline_score_from_tokens,
    confusion_at_threshold,
    first_alarm_index,
    lead_time_tokens,
    lead_time_turns,
    matched_fpr_comparison,
    permutation_pvalue,
    prama_score_from_row,
    precision_recall_at_threshold,
    threshold_at_fpr,
)
from scripts.evaluate_state_discrimination import load_labels, main as eval_main


class StateDiscriminationMetricsTests(unittest.TestCase):
    def test_auroc_perfect(self):
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]), 1.0)

    def test_auroc_reversed(self):
        self.assertAlmostEqual(auroc([1.0, 0.9, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_auroc_tied_scores(self):
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)

    def test_all_positive_and_all_negative_labels_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "negative label"):
            auroc([0.1, 0.2], [1, 1])
        with self.assertRaisesRegex(ValueError, "positive label"):
            auroc([0.1, 0.2], [0, 0])

    def test_threshold_selection_at_matched_fpr(self):
        threshold = threshold_at_fpr([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 0.0)
        self.assertAlmostEqual(threshold, 0.8)

    def test_confusion_matrix(self):
        self.assertEqual(
            confusion_at_threshold([0.1, 0.8, 0.9], [0, 1, 0], 0.8),
            {"tp": 1, "fp": 1, "tn": 1, "fn": 0},
        )

    def test_precision_recall(self):
        pr = precision_recall_at_threshold([0.1, 0.8, 0.9], [0, 1, 0], 0.8)
        self.assertAlmostEqual(pr["precision"], 0.5)
        self.assertAlmostEqual(pr["recall"], 1.0)

    def test_lead_time_positive(self):
        self.assertEqual(lead_time_tokens(10, 25), 15)
        self.assertEqual(lead_time_turns(2, 5), 3)

    def test_no_alarm_returns_null_lead_time(self):
        self.assertIsNone(first_alarm_index([0.1, 0.2], 0.5))
        self.assertIsNone(lead_time_tokens(None, 25))
        self.assertIsNone(lead_time_turns(None, 3))

    def test_baseline_and_prama_helpers(self):
        tokens = [
            {"entropy": 0.2, "top1_logprob": -0.5, "gap": 0.1},
            {"entropy": 0.4, "top1_logprob": -1.0, "gap": 0.3},
        ]
        self.assertAlmostEqual(baseline_score_from_tokens("mean_entropy", tokens), 0.3)
        self.assertAlmostEqual(baseline_score_from_tokens("mean_surprisal", tokens), 0.75)
        self.assertAlmostEqual(baseline_score_from_tokens("negative_mean_top1_gap", tokens), -0.2)
        self.assertGreater(baseline_score_from_tokens("markovian_surprisal_tau64", tokens), 0.0)
        self.assertAlmostEqual(prama_score_from_row("latent_occupancy", {"latent_occupancy": 0.7}), 0.7)
        self.assertAlmostEqual(prama_score_from_row("neg_M", {"M": -0.2}), 0.2)

    def test_confirmatory_baseline_panel_is_exactly_preregistered(self):
        self.assertEqual(
            BASELINE_SCORE_FIELDS,
            (
                "mean_surprisal",
                "mean_entropy",
                "negative_mean_top1_gap",
                "markovian_surprisal_tau64",
            ),
        )

    def test_markovian_baseline_uses_tau_64_recurrence(self):
        tokens = [{"top1_logprob": -value} for value in (1.0, 3.0, 5.0)]
        a = math.exp(-1.0 / 64.0)
        expected = a * ((1.0 - a) * 3.0) + (1.0 - a) * 5.0
        self.assertAlmostEqual(
            baseline_score_from_tokens("markovian_surprisal_tau64", tokens), expected
        )

    def test_prama_vs_baseline_comparison(self):
        result = matched_fpr_comparison(
            prama_scores=[0.1, 0.2, 0.8, 0.9],
            baseline_scores=[0.4, 0.3, 0.2, 0.1],
            labels=[0, 0, 1, 1],
            token_positions=[10, 10, 20, 20],
            event_tokens=[50, 50, 60, 60],
            target_fpr=0.0,
        )
        self.assertGreater(result["prama_auroc"], result["baseline_auroc"])

    def test_permutation_pvalue_detects_perfect_separation(self):
        scores = list(range(20))
        labels = [0] * 10 + [1] * 10
        self.assertLess(permutation_pvalue(scores, labels, permutations=999, seed=7), 0.01)

    def test_missing_labels_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "labels.csv"
            label_path.write_text("session_id,label\ns1,1\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_labels(label_path)

    def _write_fixture(self, tmp_path, labels_rows, score_rows):
        labels_path = tmp_path / "labels.csv"
        fields = ["session_id", "label", "event_token", "event_turn", "event_type", "split"]
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in labels_rows:
                writer.writerow(row)
        inputs = []
        for sid, score, baseline, tokens in score_rows:
            path = tmp_path / f"{sid}.json"
            path.write_text(
                json.dumps(
                    {
                        "session_id": sid,
                        "turns": [
                            {
                                "turn_index": 0,
                                "token_count": tokens,
                                "tokens": [
                                    {"entropy": baseline, "top1_logprob": -baseline, "gap": baseline}
                                    for _ in range(tokens)
                                ],
                                "summary": {"latent_occupancy": score, "delta": score, "xi": score, "M": -score},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inputs.append(path)
        return labels_path, inputs

    def test_split_aware_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_rows = [
                {"session_id": "train_neg", "label": "0", "event_token": "10", "event_turn": "0", "event_type": "final", "split": "train"},
                {"session_id": "train_pos", "label": "1", "event_token": "10", "event_turn": "0", "event_type": "final", "split": "train"},
                {"session_id": "test_neg", "label": "0", "event_token": "10", "event_turn": "0", "event_type": "final", "split": "test"},
                {"session_id": "test_pos", "label": "1", "event_token": "10", "event_turn": "0", "event_type": "final", "split": "test"},
            ]
            labels_path, inputs = self._write_fixture(
                tmp_path,
                labels_rows,
                [("train_neg", 0.1, 0.1, 10), ("train_pos", 0.9, 0.9, 10), ("test_neg", 0.2, 0.2, 10), ("test_pos", 0.8, 0.8, 10)],
            )
            out_dir = tmp_path / "out"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_state_discrimination.py",
                    "--labels",
                    str(labels_path),
                    "--inputs",
                    *[str(p) for p in inputs],
                    "--output-dir",
                    str(out_dir),
                ]
                self.assertEqual(eval_main(), 0)
            finally:
                sys.argv = old_argv
            metrics = json.loads((out_dir / "state_discrimination_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["split_aware"])
            self.assertIn("test", metrics["splits"])
            self.assertEqual(metrics["primary_score"], "latent_occupancy")
            self.assertEqual(metrics["splits"]["test"]["status"], "honest_null")

    def test_inconclusive_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_rows = [
                {"session_id": "a", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "test"},
                {"session_id": "b", "label": "1", "event_token": "5", "event_turn": "0", "event_type": "final", "split": "test"},
            ]
            labels_path, inputs = self._write_fixture(tmp_path, labels_rows, [("a", 0.1, 0.1, 5), ("b", 0.2, 0.2, 5)])
            out_dir = tmp_path / "out"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["evaluate_state_discrimination.py", "--labels", str(labels_path), "--inputs", *[str(p) for p in inputs], "--output-dir", str(out_dir)]
                self.assertEqual(eval_main(), 0)
            finally:
                sys.argv = old_argv
            metrics = json.loads((out_dir / "state_discrimination_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["splits"]["all"]["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
