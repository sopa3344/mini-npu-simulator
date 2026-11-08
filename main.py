"""Mini NPU: Python 3.8+, standard library only.

Core reading order: mac -> decide -> normalize_label -> analyze_case.
Input validation happens before the timed MAC calls.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter_ns


EPSILON = 1e-9
DEFAULT_REPEATS = 1000
REQUIRED_SIZES = (5, 13, 25)
BASE_DIR = Path(__file__).resolve().parent


def normalize_label(label):
    """Convert external labels to the two internal labels."""
    if not isinstance(label, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    aliases = {"+": "Cross", "cross": "Cross", "x": "X"}
    key = label.strip().lower()
    if key not in aliases:
        raise ValueError("지원하지 않는 라벨: {!r} (Cross/+/cross/X/x만 허용)".format(label))
    return aliases[key]


def validate_matrix(raw, size, name="행렬"):
    """Check exact NxN shape and return an independent matrix of finite floats."""
    if not isinstance(raw, list) or len(raw) != size:
        raise ValueError("{}: 행 수는 {}개여야 합니다.".format(name, size))
    matrix = []
    for row_index, row in enumerate(raw, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{}: {}행의 열 수는 {}개여야 합니다.".format(name, row_index, size))
        converted = []
        for col_index, value in enumerate(row, 1):
            # bool is a subclass of int, but true/false is not numeric input here.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{}: ({}, {})는 숫자여야 합니다.".format(name, row_index, col_index))
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise ValueError("{}: float로 표현할 수 없는 숫자입니다.".format(name))
            if not math.isfinite(number):
                raise ValueError("{}: NaN과 무한대는 허용하지 않습니다.".format(name))
            converted.append(number)
        matrix.append(converted)
    return matrix


def mac(pattern, kernel):
    """One multiply and one accumulation per cell, with ordinary loops.

    Contract: both inputs are already validated finite NxN matrices.
    """
    total = 0.0
    size = len(pattern)
    for row in range(size):
        for col in range(size):
            total += pattern[row][col] * kernel[row][col]
    return total


def decide(score_a, score_b, label_a="Cross", label_b="X", epsilon=EPSILON):
    """Strict absolute tolerance: diff < epsilon ties; diff == epsilon does not."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon은 양의 유한수여야 합니다.")
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError("MAC 점수가 유한수가 아닙니다. 곱셈 또는 누적 오버플로를 확인하세요.")
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def generate_pattern(size, label):
    """Create Cross/X patterns for a positive odd N (bonus)."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError("패턴 크기는 3 이상의 홀수여야 합니다.")
    label = normalize_label(label)
    result = []
    for row in range(size):
        line = []
        for col in range(size):
            if label == "Cross":
                filled = row == size // 2 or col == size // 2
            else:
                filled = row == col or row + col == size - 1
            line.append(float(filled))
        result.append(line)
    return result


def flatten(matrix):
    """Flatten once, outside the timed loop."""
    return [value for row in matrix for value in row]


def mac_flat(pattern, kernel):
    """Bonus: the same arithmetic and traversal order in one dimension."""
    total = 0.0
    for index in range(len(pattern)):
        total += pattern[index] * kernel[index]
    return total


def benchmark(function, pattern, kernel, repeats=DEFAULT_REPEATS):
    """Mean milliseconds per call. File/input/print/conversion are excluded."""
    if repeats < 10:
        raise ValueError("측정 반복 횟수는 최소 10회여야 합니다.")
    for _ in range(5):
        function(pattern, kernel)  # warm-up, excluded
    elapsed_ns = 0
    for _ in range(repeats):
        start = perf_counter_ns()
        function(pattern, kernel)
        elapsed_ns += perf_counter_ns() - start
    return elapsed_ns / repeats / 1_000_000


def show_performance(sizes, repeats, compare_flat=False, manual_pair=None):
    print("\n[성능 분석] 필터 1개, {}회 평균, 워밍업 5회 제외".format(repeats))
    print("크기(NxN) | 평균 시간(ms) | MAC 횟수(N^2)")
    flat_rows = []
    for size in sizes:
        if manual_pair is None:
            pattern = generate_pattern(size, "Cross")
            kernel = generate_pattern(size, "Cross")
        else:
            pattern, kernel = manual_pair
        average = benchmark(mac, pattern, kernel, repeats)
        print("{:>2}x{:<2}     | {:>13.6f} | {:>6}".format(size, size, average, size * size))
        if compare_flat:
            flat_pattern, flat_kernel = flatten(pattern), flatten(kernel)
            if mac_flat(flat_pattern, flat_kernel) != mac(pattern, kernel):
                raise ValueError("1D/2D 연산 결과가 다릅니다.")
            flat_average = benchmark(mac_flat, flat_pattern, flat_kernel, repeats)
            flat_rows.append((size, average, flat_average))
    print("MAC 1회는 곱셈 1회와 덧셈 1회입니다. 필터 2개의 판정은 2N^2 MAC입니다.")
    if manual_pair is None:
        print("성능표는 자동 생성한 Cross 입력과 Cross 필터를 사용합니다.")
    else:
        print("성능표는 직접 입력한 패턴과 필터 A를 사용합니다.")
    if flat_rows:
        print("\n[보너스: 같은 입력과 반복 횟수로 2D/1D 비교]")
        print("크기 | 2D(ms) | 1D(ms) | 2D/1D 비율")
        for size, original, flat in flat_rows:
            ratio = "{:.2f}".format(original / flat) if flat else "측정 불가"
            print("{}x{} | {:.6f} | {:.6f} | {}".format(size, size, original, flat, ratio))
        print("1D 변환 비용은 제외했습니다. 1D가 항상 더 빠른 것은 아닙니다.")


def read_matrix(name, size=3):
    print("\n{}: {}줄, 각 줄에 {}개의 숫자를 공백으로 구분하세요.".format(name, size, size))
    matrix = []
    while len(matrix) < size:
        row_number = len(matrix) + 1
        raw = input("{}행: ".format(row_number)).split()
        try:
            if len(raw) != size:
                raise ValueError("열 개수 불일치")
            values = [float(token) for token in raw]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("NaN/무한대 입력")
        except (ValueError, OverflowError):
            print("입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.".format(size))
            print("{}행을 다시 입력합니다.".format(row_number))
            continue
        matrix.append(values)
    return matrix


def manual_mode(repeats, compare_flat=False):
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("\n필터 A와 B 저장 완료:")
    for name, matrix in (("A", filter_a), ("B", filter_b)):
        print(name)
        for row in matrix:
            print(" ".join("{:g}".format(value) for value in row))
    pattern = read_matrix("패턴")
    elapsed_ns = 0
    for _ in range(5):
        mac(pattern, filter_a)
        mac(pattern, filter_b)
    for _ in range(repeats):
        start = perf_counter_ns()
        score_a = mac(pattern, filter_a)
        score_b = mac(pattern, filter_b)
        elapsed_ns += perf_counter_ns() - start
    print("\n[MAC 결과]")
    print("A 점수: {!r}\nB 점수: {!r}".format(score_a, score_b))
    print("연산 시간(필터 2개, 평균/{}회): {:.6f} ms".format(repeats, elapsed_ns / repeats / 1_000_000))
    try:
        decision = decide(score_a, score_b, "A", "B")
    except ValueError as error:
        print("연산 오류: {}".format(error))
        return 1
    print("판정: {}".format("판정 불가 (UNDECIDED)" if decision == "UNDECIDED" else decision))
    print("동점 기준: abs(A-B) < {}".format(EPSILON))
    show_performance((3,), repeats, compare_flat, (pattern, filter_a))
    return 0


def load_filter_bank(raw_filters):
    if not isinstance(raw_filters, dict):
        raise ValueError("filters는 객체여야 합니다.")
    bank, errors = {}, {}
    for key, raw_pair in raw_filters.items():
        try:
            if not re.fullmatch(r"size_([1-9][0-9]*)", key):
                raise ValueError("필터 키는 size_N 형식이어야 합니다.")
            size = int(key.split("_")[1])
            if not isinstance(raw_pair, dict):
                raise ValueError("Cross와 X 필터를 담은 객체여야 합니다.")
            pair = {}
            for label, raw_matrix in raw_pair.items():
                standard = normalize_label(label)
                if standard in pair:
                    raise ValueError("정규화 후 {} 필터가 중복됩니다.".format(standard))
                pair[standard] = validate_matrix(raw_matrix, size, key + "/" + standard)
            if set(pair) != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다.")
            bank[key] = pair
        except ValueError as error:
            errors[key] = str(error)
    for size in REQUIRED_SIZES:
        key = "size_{}".format(size)
        if key not in raw_filters:
            errors[key] = "필수 필터 묶음이 없습니다."
    return bank, errors


def analyze_case(case_id, case, bank, filter_errors):
    result = {"id": case_id, "scores": None, "decision": None,
              "expected": None, "passed": False, "reason": ""}
    try:
        match = re.fullmatch(r"size_([1-9][0-9]*)_([0-9]+)", case_id)
        if not match:
            raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")
        size = int(match.group(1))
        if not isinstance(case, dict):
            raise ValueError("패턴 항목은 input과 expected를 가진 객체여야 합니다.")
        if "input" not in case or "expected" not in case:
            raise ValueError("input 또는 expected가 없습니다.")
        expected = normalize_label(case["expected"])
        result["expected"] = expected
        pattern = validate_matrix(case["input"], size, "input")
        filter_key = "size_{}".format(size)
        if filter_key in filter_errors:
            raise ValueError("{}: {}".format(filter_key, filter_errors[filter_key]))
        if filter_key not in bank:
            raise ValueError("{} 필터를 찾을 수 없습니다.".format(filter_key))
        pair = bank[filter_key]
    except ValueError as error:
        result["reason"] = "데이터/스키마 오류: {}".format(error)
        return result
    cross_score = mac(pattern, pair["Cross"])
    x_score = mac(pattern, pair["X"])
    result["scores"] = (cross_score, x_score)
    try:
        decision = decide(cross_score, x_score)
    except ValueError as error:
        result["reason"] = "수치 연산 오류: {}".format(error)
        return result
    result["decision"] = decision
    result["passed"] = decision == expected
    if decision == "UNDECIDED":
        result["reason"] = "동점 규칙: 점수 차이 {!r} < {}; expected={}".format(abs(cross_score - x_score), EPSILON, expected)
    elif not result["passed"]:
        result["reason"] = "판정 불일치: 예상 {}, 실제 {}. 원본 라벨과 데이터, MAC 로직을 점검하세요.".format(expected, decision)
    return result


def analyze_document(document):
    if not isinstance(document, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")
    patterns = document.get("patterns")
    if not isinstance(patterns, dict) or not patterns:
        raise ValueError("patterns는 한 개 이상의 케이스를 가진 객체여야 합니다.")
    raw_filters = document.get("filters")
    if isinstance(raw_filters, dict):
        bank, filter_errors = load_filter_bank(raw_filters)
    else:
        bank = {}
        filter_errors = {"filters": "filters가 없거나 객체가 아닙니다."}
    results = [analyze_case(key, value, bank, filter_errors) for key, value in patterns.items()]
    return bank, filter_errors, results


def load_document(path):
    # utf-8-sig accepts both ordinary UTF-8 and Windows UTF-8 BOM files.
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def json_mode(path, repeats, compare_flat=False):
    try:
        document = load_document(path)
        bank, filter_errors, results = analyze_document(document)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print("파일/전체 스키마 오류: {}".format(error))
        print("테스트 미실행: 케이스 목록을 읽을 수 없어 PASS/FAIL 총합을 만들지 않습니다.")
        print("원본 data.json은 main.py 옆에 두거나 --data 경로로 지정하세요.")
        print("포함된 예제로 실행: python main.py --mode json --data data.sample.json")
        return 2
    print("\n[필터 로드] 파일: {}".format(Path(path).resolve()))
    metadata = document.get("_meta")
    if isinstance(metadata, dict) and metadata.get("source") == "generated_learning_sample":
        print("주의: 직접 생성한 학습용 예제입니다. 교육기관 원본 데이터가 아닙니다.")
    for key in bank:
        print("{} 필터 로드 완료 (Cross, X)".format(key))
    for key, reason in filter_errors.items():
        print("필터 오류 {}: {}".format(key, reason))
    print("\n[패턴 분석]")
    for result in results:
        print("\n{}".format(result["id"]))
        if result["scores"] is not None:
            print("Cross 점수: {!r} | X 점수: {!r}".format(*result["scores"]))
        else:
            print("Cross 점수: 계산 안 함 | X 점수: 계산 안 함")
        status = "PASS" if result["passed"] else "FAIL"
        print("판정: {} | expected: {} | {}".format(result["decision"] or "검증/연산 실패", result["expected"] or "확인 불가", status))
        if result["reason"]:
            print("사유: " + result["reason"])
    show_performance((3, 5, 13, 25), repeats, compare_flat)
    failed = [result for result in results if not result["passed"]]
    print("\n[결과 요약]")
    print("총 테스트: {}개\n통과: {}개\n실패: {}개".format(len(results), len(results) - len(failed), len(failed)))
    print("필터 설정 오류: {}개 (케이스 합계와 별도)".format(len(filter_errors)))
    for result in failed:
        print("- {}: {}".format(result["id"], result["reason"]))
    if not failed:
        print("실패 케이스 없음. 라벨 정규화 및 epsilon 비교 정책을 적용했습니다.")
    return 1 if failed or filter_errors else 0


def make_sample_document(sizes=REQUIRED_SIZES):
    filters, patterns = {}, {}
    for size in sizes:
        cross, x_pattern = generate_pattern(size, "Cross"), generate_pattern(size, "X")
        filters["size_{}".format(size)] = {"cross": cross, "x": x_pattern}
        patterns["size_{}_1".format(size)] = {"input": cross, "expected": "+"}
        patterns["size_{}_2".format(size)] = {"input": x_pattern, "expected": "x"}
    if 5 in sizes:
        patterns["size_5_3"] = {"input": [[0.0] * 5 for _ in range(5)], "expected": "+"}
    if 13 in sizes:
        near_tie = [[0.0] * 13 for _ in range(13)]
        near_tie[0][6] = 0.1 + 0.2
        near_tie[0][0] = 0.3
        patterns["size_13_3"] = {"input": near_tie, "expected": "x"}
    return {"_meta": {"source": "generated_learning_sample", "note": "직접 만든 예제. 원본 과제 데이터가 아니며 동점 FAIL 2개를 포함합니다(5/13 포함 시)."},
            "filters": filters, "patterns": patterns}


def repeat_count(value):
    number = int(value)
    if number < 10:
        raise argparse.ArgumentTypeError("--repeats는 10 이상이어야 합니다.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="표준 라이브러리만 사용하는 Mini NPU")
    parser.add_argument("--mode", choices=("manual", "json"))
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data.json")
    parser.add_argument("--repeats", type=repeat_count, default=DEFAULT_REPEATS)
    parser.add_argument("--compare-flat", action="store_true", help="1D/2D 성능 비교")
    parser.add_argument("--generate-sample", type=Path, metavar="PATH", help="학습용 JSON 생성, 기존 파일 덮어쓰기 금지")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(REQUIRED_SIZES), help="생성할 홀수 크기 N")
    args = parser.parse_args(argv)
    print("=== Mini NPU Simulator ===")
    try:
        if args.generate_sample:
            sample = make_sample_document(args.sizes)
            with args.generate_sample.open("x", encoding="utf-8") as stream:
                json.dump(sample, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print("학습용 예제 생성: {}".format(args.generate_sample.resolve()))
            return 0
        mode = args.mode
        while mode is None:
            print("\n1. 사용자 입력 (3x3)\n2. data.json 분석")
            choice = input("선택: ").strip()
            mode = {"1": "manual", "2": "json"}.get(choice)
            if mode is None:
                print("1 또는 2를 입력하세요.")
        if mode == "manual":
            return manual_mode(args.repeats, args.compare_flat)
        return json_mode(args.data, args.repeats, args.compare_flat)
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되었습니다. 각 행을 입력해 다시 실행하세요.")
        return 130
    except (OSError, ValueError) as error:
        print("실행 오류: {}".format(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
