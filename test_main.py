"""Meaningful regression tests; run: python -m unittest -v."""

import ast
import contextlib
import copy
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


ROOT = Path(__file__).resolve().parent


class CoreTests(unittest.TestCase):
    def test_three_by_three_reference_scores(self):
        cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        x_shape = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        self.assertEqual(main.mac(cross, cross), 5.0)
        self.assertEqual(main.mac(cross, x_shape), 1.0)
        self.assertEqual(main.mac(x_shape, x_shape), 5.0)
        self.assertEqual(main.decide(1, 5, "A", "B"), "B")

    def test_negative_and_fractional_values(self):
        self.assertEqual(main.mac([[0.5, -2], [3, 0]], [[2, 4], [-1, 8]]), -10)

    def test_normalization_and_unknown_labels(self):
        for raw, expected in (("+", "Cross"), ("cross", "Cross"), (" Cross ", "Cross"), ("x", "X"), ("X", "X")):
            self.assertEqual(main.normalize_label(raw), expected)
        for invalid in ("UNDECIDED", "square", None, 1):
            with self.assertRaises(ValueError):
                main.normalize_label(invalid)

    def test_epsilon_strict_boundary_and_roundoff(self):
        self.assertEqual(main.decide(0.1 + 0.2, 0.3), "UNDECIDED")
        self.assertEqual(main.decide(0, 0.999e-9), "UNDECIDED")
        self.assertEqual(main.decide(0, 1e-9), "X")
        self.assertEqual(main.decide(1.001e-9, 0), "Cross")
        self.assertEqual(main.decide(0.9, 0.8999999999999999), "UNDECIDED")

    def test_nonfinite_score_and_bad_epsilon(self):
        for score in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                main.decide(score, 1)
        for epsilon in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                main.decide(1, 2, epsilon=epsilon)

    def test_wrong_shapes_and_nonnumeric_cells(self):
        bad_values = [[], [[1, 2]], [[1, 2], [1]], [[1, 2], "34"], [[1, 2], [3, True]],
                      [[1, 2], [3, "4"]], [[1, 2], [3, None]], [[1, 2], [3, math.nan]],
                      [[1, 2], [3, math.inf]], [[1, 2], [3, 10 ** 1000]]]
        for value in bad_values:
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                main.validate_matrix(value, 2)

    def test_matrix_storage_is_independent(self):
        original = [[1, 2], [3, 4]]
        matrix = main.validate_matrix(original, 2)
        matrix[0][1] = 7
        self.assertEqual(matrix[0][1], 7)
        self.assertEqual(original[0][1], 2)

    def test_generator_and_flat_scores(self):
        for size in (3, 5, 7, 13, 25):
            cross = main.generate_pattern(size, "Cross")
            x_shape = main.generate_pattern(size, "X")
            self.assertEqual(main.mac(cross, cross), 2 * size - 1)
            self.assertEqual(main.mac(cross, x_shape), 1)
            self.assertEqual(main.mac_flat(main.flatten(cross), main.flatten(x_shape)), 1)
            # Both versions traverse the same order, including fractional inputs.
            values = [[(row - col) / 7 for col in range(size)] for row in range(size)]
            self.assertEqual(main.mac(values, cross), main.mac_flat(main.flatten(values), main.flatten(cross)))
        for invalid in (0, 1, 2, 4, -3, True, 3.5):
            with self.assertRaises(ValueError):
                main.generate_pattern(invalid, "X")

    def test_benchmark_units_and_repetitions(self):
        calls = []
        with patch.object(main, "perf_counter_ns", side_effect=[i * 1_000_000 for i in range(20)]):
            average = main.benchmark(lambda a, b: calls.append((a, b)), [[1]], [[2]], 10)
        self.assertEqual(average, 1.0)
        self.assertEqual(len(calls), 15)  # five warm-ups plus ten measured calls
        with self.assertRaises(ValueError):
            main.benchmark(main.mac, [[1]], [[1]], 9)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.document = main.make_sample_document()

    def test_sample_six_passes_and_two_policy_failures(self):
        _, errors, results = main.analyze_document(self.document)
        self.assertFalse(errors)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(item["passed"] for item in results), 6)
        failures = [item for item in results if not item["passed"]]
        self.assertEqual([item["id"] for item in failures], ["size_5_3", "size_13_3"])
        self.assertTrue(all(item["decision"] == "UNDECIDED" for item in failures))

    def test_malformed_case_does_not_stop_later_cases(self):
        self.document["patterns"]["size_5_1"]["input"] = [[1]]
        _, _, results = main.analyze_document(self.document)
        self.assertFalse(results[0]["passed"])
        self.assertIn("행 수", results[0]["reason"])
        self.assertTrue(results[1]["passed"])
        self.assertEqual(len(results), 8)

    def test_bad_pattern_keys_values_and_labels(self):
        bank, errors = main.load_filter_bank(self.document["filters"])
        base_case = self.document["patterns"]["size_5_1"]
        for key, case in (("size_5_bad", base_case), ("bad", base_case), ("size_5_9", []),
                          ("size_5_9", {"input": []}),
                          ("size_5_9", {"input": base_case["input"], "expected": "circle"})):
            with self.subTest(key=key):
                result = main.analyze_case(key, case, bank, errors)
                self.assertFalse(result["passed"])
                self.assertIn("데이터/스키마", result["reason"])

    def test_bad_filter_size_only_affects_related_cases(self):
        self.document["filters"]["size_5"]["cross"] = [[1]]
        _, errors, results = main.analyze_document(self.document)
        self.assertIn("size_5", errors)
        self.assertFalse(results[0]["passed"])
        self.assertFalse(results[1]["passed"])
        self.assertTrue(results[2]["passed"])

    def test_missing_filters_and_missing_partner(self):
        del self.document["filters"]["size_5"]
        del self.document["filters"]["size_13"]["x"]
        _, errors, results = main.analyze_document(self.document)
        self.assertEqual(set(errors), {"size_5", "size_13"})
        self.assertTrue(results[4]["passed"])

    def test_duplicate_normalized_filter_labels(self):
        pair = self.document["filters"]["size_5"]
        pair["+"] = copy.deepcopy(pair["cross"])
        _, errors, _ = main.analyze_document(self.document)
        self.assertIn("중복", errors["size_5"])

    def test_invalid_filters_root_fails_cases_without_crash(self):
        self.document["filters"] = []
        _, errors, results = main.analyze_document(self.document)
        self.assertIn("filters", errors)
        self.assertEqual(len(results), 8)
        self.assertFalse(any(item["passed"] for item in results))

    def test_invalid_document_root_and_empty_patterns(self):
        for value in ([], None, {}, {"filters": {}, "patterns": []}, {"filters": {}, "patterns": {}}):
            with self.assertRaises(ValueError):
                main.analyze_document(value)

    def test_wrong_expected_is_not_silently_corrected(self):
        self.document["patterns"]["size_5_1"]["expected"] = "x"
        _, _, results = main.analyze_document(self.document)
        self.assertFalse(results[0]["passed"])
        self.assertEqual(results[0]["decision"], "Cross")
        self.assertIn("판정 불일치", results[0]["reason"])

    def test_numeric_overflow_is_case_failure(self):
        self.document["patterns"]["size_5_1"]["input"] = [[1e308] * 5 for _ in range(5)]
        _, _, results = main.analyze_document(self.document)
        self.assertFalse(results[0]["passed"])
        self.assertIn("수치 연산 오류", results[0]["reason"])
        self.assertTrue(results[1]["passed"])

    def test_bom_json_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(self.document), encoding="utf-8-sig")
            self.assertEqual(len(main.load_document(path)["patterns"]), 8)
        self.document["patterns"]["size_5_1"]["input"][0][0] = math.nan
        _, _, results = main.analyze_document(self.document)
        self.assertIn("NaN", results[0]["reason"])

    def test_optional_metadata_type_does_not_crash(self):
        self.document["_meta"] = None
        with patch.object(main, "load_document", return_value=self.document), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.json_mode("unused.json", 10), 1)


class CliTests(unittest.TestCase):
    def run_cli(self, args=(), stdin="", cwd=None):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run([sys.executable, str(ROOT / "main.py")] + list(args),
                              input=stdin, text=True, encoding="utf-8", capture_output=True,
                              cwd=cwd or ROOT, env=env, timeout=20)

    def test_interactive_menu_invalid_rows_and_x_result(self):
        lines = "9\n1\n0 1\na b c\nnan 1 0\n0 1 0\n1 1 1\n0 1 0\n1 0 1\n0 1 0\n1 0 1\n1 0 1\n0 1 0\n1 0 1\n"
        result = self.run_cli(["--repeats", "10"], lines)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("입력 형식 오류"), 3)
        self.assertIn("1 또는 2", result.stdout)
        self.assertIn("저장 완료", result.stdout)
        self.assertIn("A 점수: 1.0", result.stdout)
        self.assertIn("B 점수: 5.0", result.stdout)
        self.assertIn("판정: B", result.stdout)

    def test_interactive_tie_and_interrupted_row_count(self):
        result = self.run_cli(["--mode", "manual", "--repeats", "10"], "0 0 0\n" * 9)
        self.assertEqual(result.returncode, 0)
        self.assertIn("판정 불가", result.stdout)
        incomplete = self.run_cli(["--mode", "manual"], "0 1 0\n")
        self.assertEqual(incomplete.returncode, 130)
        self.assertNotIn("Traceback", incomplete.stderr)

    def test_json_cli_summary_and_performance(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(main.make_sample_document()), encoding="utf-8")
            result = self.run_cli(["--mode", "json", "--data", str(path), "--repeats", "10", "--compare-flat"], cwd=directory)
        self.assertEqual(result.returncode, 1)
        self.assertIn("총 테스트: 8개\n통과: 6개\n실패: 2개", result.stdout)
        self.assertIn("625", result.stdout)
        self.assertIn("2D/1D", result.stdout)

    def test_missing_file_and_malformed_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "missing.json"
            result = self.run_cli(["--mode", "json", "--data", str(path)])
            self.assertEqual(result.returncode, 2)
            self.assertIn("테스트 미실행", result.stdout)
            path.write_text("{not-json}", encoding="utf-8")
            result = self.run_cli(["--mode", "json", "--data", str(path)])
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)

    def test_generate_sample_and_protect_existing_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "generated.json"
            result = self.run_cli(["--generate-sample", str(path), "--sizes", "3", "5", "13", "25"])
            self.assertEqual(result.returncode, 0)
            before = path.read_bytes()
            self.assertIn("size_3", json.loads(before)["filters"])
            result = self.run_cli(["--generate-sample", str(path)])
            self.assertEqual(result.returncode, 2)
            self.assertEqual(path.read_bytes(), before)

    def test_repetition_minimum_in_cli(self):
        result = self.run_cli(["--mode", "json", "--repeats", "9"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("10 이상", result.stderr)

    def test_python38_syntax_and_standard_library_imports(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 8))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module.split(".")[0])
        self.assertEqual(names, {"argparse", "json", "math", "re", "sys", "pathlib", "time"})


if __name__ == "__main__":
    unittest.main()
