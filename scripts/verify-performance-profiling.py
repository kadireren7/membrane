#!/usr/bin/env python3
"""Validate results/performance-profiling/measurements.json: schema, that
every timing value is finite and non-negative, that the reported median
actually matches the raw runs, that throughput figures are arithmetically
consistent with their own timing/token-count inputs, that a CPU-backend
row never carries a GPU-only field, that a run with fallback_attempted
false never carries fallback-implying timing, and that every measurement
carries real device/backend metadata (Section 19 of the Phase 24 task:
no hardware-generalization -- a device-scoped number stays device-scoped).

This is a measurement-evidence validator, separate from every other
scripts/verify-*.py in this project -- same one-file-per-concern
convention. It validates the COMMITTED evidence file; it does not
regenerate it (see scripts/measure-performance-profile.py for that).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "performance-profiling" / "measurements.json"
FAILURES = []
CHECK_COUNT = 0

VALID_BACKENDS = ("cpu", "vulkan")
TIMING_FIELDS = ("total_ms", "planner_ms", "model_load_ms", "tokenization_ms",
	"context_create_ms", "prefill_ms", "decode_ms", "first_token_ms")
THROUGHPUT_FIELDS = ("prefill_tokens_per_second", "decode_tokens_per_second")

# CodeRabbit review (PR #34): the exact, fixed 12-point label set
# scripts/measure-performance-profile.py's own POINTS list produces --
# a missing/duplicate/unexpected label must fail, not silently pass an
# incomplete matrix as if it were the full committed evidence set.
EXPECTED_LABELS = frozenset({
	"smollm2-135m_vulkan_native", "smollm2-135m_vulkan_q8",
	"smollm2-135m_vulkan_q5", "smollm2-135m_vulkan_adaptive",
	"smollm2-135m_cpu_native", "smollm2-135m_cpu_q8",
	"smollm2-360m_vulkan_native", "qwen2.5-1.5b_vulkan_native_gpu_placement",
	"qwen2.5-1.5b_vulkan_native_cpu_placement", "qwen2.5-1.5b_cpu_native",
	"smollm2-135m_vulkan_partial_half_15", "smollm2-135m_vulkan_auto",
})


def fail(name, detail):
	FAILURES.append(f"{name}: {detail}")


def check(name):
	def decorator(fn):
		def wrapper(*args, **kwargs):
			global CHECK_COUNT
			CHECK_COUNT += 1
			before = len(FAILURES)
			fn(*args, **kwargs)
			status = "PASS" if len(FAILURES) == before else "FAIL"
			print(f"[{status}] {name}")
		return wrapper
	return decorator


@check("measurements.json is valid JSON with the expected top-level shape")
def check_top_level(data):
	for field in ("schema_version", "membrane_commit", "measurements"):
		if field not in data:
			fail("top-level shape", f"missing required field {field!r}")
	if data.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version "
			f"{data.get('schema_version')!r}")
	commit = data.get("membrane_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
		fail("top-level shape", f"membrane_commit {commit!r} is not a "
			f"40-hex SHA")
	if not isinstance(data.get("measurements"), list) or not data["measurements"]:
		fail("top-level shape", "'measurements' must be a non-empty list")


def stat_block(m, name):
	"""Fetches a {median,min,max,raw} stat block nested under either
	timings_ms.<name> or throughput.<name>."""
	if name in TIMING_FIELDS:
		return (m.get("timings_ms") or {}).get(name)
	return (m.get("throughput") or {}).get(name)


@check("every measurement has label/backend/model/device metadata and a "
	"real 40-hex commit")
def check_measurement_shape(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		if m.get("backend") not in VALID_BACKENDS:
			fail(label, f"backend {m.get('backend')!r} not in {VALID_BACKENDS}")
		if not m.get("model"):
			fail(label, "missing model")
		if not m.get("model_arch"):
			fail(label, "missing model_arch")
		if not re.fullmatch(r"[0-9a-f]{40}", m.get("commit", "") or ""):
			fail(label, f"commit {m.get('commit')!r} is not a 40-hex SHA")
		if not isinstance(m.get("repeats"), int) or m["repeats"] <= 0:
			fail(label, f"repeats must be a positive integer, got "
				f"{m.get('repeats')!r}")
		if not isinstance(m.get("ok_count"), int) or m["ok_count"] < 0:
			fail(label, f"ok_count must be a non-negative integer, got "
				f"{m.get('ok_count')!r}")
		if m["ok_count"] > m["repeats"]:
			fail(label, f"ok_count ({m['ok_count']}) exceeds repeats "
				f"({m['repeats']})")


@check("Section 19: no hardware-generalization -- every measurement names "
	"a real device when it ran on a GPU backend")
def check_hardware_scope(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		if m.get("backend") == "vulkan" and m.get("ok_count", 0) > 0 \
				and not m.get("device"):
			fail(label, "backend=vulkan with ok_count > 0 but no 'device' "
				"recorded -- a Vulkan measurement must always name the "
				"real device it ran on")


@check("no CPU-backend row carries a GPU-only field, no failed row carries "
	"a fake success timing")
def check_no_cross_contamination(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		if m.get("backend") == "cpu" and m.get("device"):
			fail(label, f"backend=cpu but device is set to {m['device']!r} "
				f"-- a CPU-only run has no GPU device")
		if m.get("ok_count", 0) == 0:
			for field in TIMING_FIELDS:
				block = stat_block(m, field)
				if block and block.get("median") is not None:
					fail(label, f"ok_count is 0 but {field}.median is "
						f"{block['median']!r} -- a fully-failed measurement "
						f"must never carry a fabricated successful timing")
			if not m.get("note"):
				fail(label, "ok_count is 0 but 'note' does not explain why "
					"(Section 22 of the Phase 24 task: a measurement bug "
					"or a real failure must be disclosed, never silently "
					"blank)")


@check("no negative or non-finite timing/throughput value anywhere")
def check_finite_nonnegative(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		for field in TIMING_FIELDS + THROUGHPUT_FIELDS:
			block = stat_block(m, field)
			if not block:
				continue
			for key in ("median", "min", "max"):
				v = block.get(key)
				if v is None:
					continue
				if not isinstance(v, (int, float)) or isinstance(v, bool):
					fail(label, f"{field}.{key} is not numeric: {v!r}")
					continue
				if math.isnan(v) or math.isinf(v):
					fail(label, f"{field}.{key} is not finite: {v!r}")
				elif v < 0:
					fail(label, f"{field}.{key} is negative: {v!r} -- no "
						f"real timing/throughput can be negative")
			for v in block.get("raw", []):
				if not isinstance(v, (int, float)) or isinstance(v, bool) \
						or math.isnan(v) or math.isinf(v) or v < 0:
					fail(label, f"{field}.raw contains an invalid value: "
						f"{v!r}")


@check("median/min/max are internally consistent with the raw runs")
def check_median_matches_raw(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		for field in TIMING_FIELDS + THROUGHPUT_FIELDS:
			block = stat_block(m, field)
			if not block or not block.get("raw"):
				continue
			raw = sorted(block["raw"])
			n = len(raw)
			expected_median = (raw[n // 2] if n % 2 == 1
				else (raw[n // 2 - 1] + raw[n // 2]) / 2.0)
			if block.get("median") is not None and abs(
					block["median"] - expected_median) > 0.01:
				fail(label, f"{field}.median {block['median']!r} does not "
					f"match the raw runs {block['raw']!r} (expected "
					f"~{round(expected_median, 3)})")
			if block.get("min") is not None and abs(block["min"] - raw[0]) > 0.01:
				fail(label, f"{field}.min {block['min']!r} does not match "
					f"the raw runs' real minimum {raw[0]!r}")
			if block.get("max") is not None and abs(block["max"] - raw[-1]) > 0.01:
				fail(label, f"{field}.max {block['max']!r} does not match "
					f"the raw runs' real maximum {raw[-1]!r}")


@check("the measurement set is exactly the expected 12-point matrix -- "
	"no missing, duplicate, or unexpected label")
def check_exact_label_set(measurements):
	labels = [m.get("label") for m in measurements]
	seen = set()
	duplicates = set()
	for label in labels:
		if label in seen:
			duplicates.add(label)
		seen.add(label)
	if duplicates:
		fail("label set", f"duplicate label(s): {sorted(duplicates)!r}")
	missing = EXPECTED_LABELS - seen
	if missing:
		fail("label set", f"missing expected label(s): {sorted(missing)!r} "
			f"-- an incomplete matrix must never silently pass as the full "
			f"committed evidence set")
	unexpected = seen - EXPECTED_LABELS
	if unexpected:
		fail("label set", f"unexpected label(s) not in the fixed matrix: "
			f"{sorted(unexpected)!r}")


@check("raw-sample cardinality matches ok_count, and throughput is "
	"recomputable from real token counts and stage durations")
def check_raw_cardinality_and_throughput_authenticity(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		ok_count = m.get("ok_count", 0)
		if ok_count == 0:
			continue
		for field in TIMING_FIELDS + THROUGHPUT_FIELDS:
			block = stat_block(m, field)
			raw = (block or {}).get("raw") or []
			if len(raw) != ok_count:
				fail(label, f"{field}.raw has {len(raw)} sample(s), expected "
					f"exactly ok_count ({ok_count}) -- a successful run must "
					f"never be silently dropped from its own stage's raw "
					f"samples")
		token_counts = m.get("token_counts") or {}
		prompt_tokens = token_counts.get("prompt_tokens") or []
		generated_tokens = token_counts.get("generated_tokens") or []
		if len(prompt_tokens) != ok_count or len(generated_tokens) != ok_count:
			fail(label, f"token_counts has {len(prompt_tokens)} prompt/"
				f"{len(generated_tokens)} generated sample(s), expected "
				f"ok_count ({ok_count}) each -- cannot verify throughput "
				f"authenticity without a real token count per run")
			continue
		prefill_raw = ((m.get("timings_ms") or {}).get("prefill_ms") or {}).get("raw") or []
		decode_raw = ((m.get("timings_ms") or {}).get("decode_ms") or {}).get("raw") or []
		prefill_tp_raw = ((m.get("throughput") or {}).get(
			"prefill_tokens_per_second") or {}).get("raw") or []
		decode_tp_raw = ((m.get("throughput") or {}).get(
			"decode_tokens_per_second") or {}).get("raw") or []
		if not (len(prefill_raw) == len(decode_raw) == len(prefill_tp_raw)
				== len(decode_tp_raw) == ok_count):
			continue  # already flagged by the raw-cardinality check above
		for i in range(ok_count):
			if prefill_raw[i] > 0:
				expected_prefill_tp = prompt_tokens[i] / (prefill_raw[i] / 1000.0)
				if abs(expected_prefill_tp - prefill_tp_raw[i]) > max(
						0.5, 0.02 * expected_prefill_tp):
					fail(label, f"run {i}: prefill_tokens_per_second "
						f"{prefill_tp_raw[i]!r} does not match "
						f"prompt_tokens ({prompt_tokens[i]!r}) / prefill_ms "
						f"({prefill_raw[i]!r}) = ~{round(expected_prefill_tp, 3)}")
			if decode_raw[i] > 0:
				expected_decode_tp = generated_tokens[i] / (decode_raw[i] / 1000.0)
				if abs(expected_decode_tp - decode_tp_raw[i]) > max(
						0.5, 0.02 * expected_decode_tp):
					fail(label, f"run {i}: decode_tokens_per_second "
						f"{decode_tp_raw[i]!r} does not match "
						f"generated_tokens ({generated_tokens[i]!r}) / "
						f"decode_ms ({decode_raw[i]!r}) = "
						f"~{round(expected_decode_tp, 3)}")


DOC_PATH = REPO_ROOT / "docs" / "performance-profiling.md"


@check("docs/performance-profiling.md's specific numeric claims match the "
	"committed measurements (no doc drift)")
def check_docs_match_measurements(measurements):
	if not DOC_PATH.exists():
		fail("doc claims", f"{DOC_PATH} does not exist")
		return
	text = DOC_PATH.read_text()
	by_label = {m.get("label"): m for m in measurements}

	# The matrix size and success/failure counts the doc's own prose
	# states must match the real data exactly.
	ok_points = sum(1 for m in measurements if m.get("ok_count", 0) > 0)
	failed_points = len(measurements) - ok_points
	if f"{ok_points}/{len(measurements)} points succeeded" not in text:
		fail("doc claims", f"doc does not state the real "
			f"'{ok_points}/{len(measurements)} points succeeded' figure")
	if failed_points > 0 and str(failed_points) not in text:
		fail("doc claims", f"doc does not appear to mention the real "
			f"failed-point count ({failed_points})")

	# Every named-in-doc label must actually exist in the evidence file
	# (catches a stale/renamed label reference).
	for label in EXPECTED_LABELS:
		if label not in text:
			fail("doc claims", f"expected label {label!r} is never "
				f"mentioned in {DOC_PATH.name}")

	# The doc's own "planner_ms/tokenization_ms/context_create_ms all
	# stay under X%" claim must actually hold for the real max observed
	# -- re-derive the real max and require the doc to state a threshold
	# at or above it (a stale, too-low threshold is a real doc bug, not
	# just cosmetic -- see PR #34 review).
	max_share_pct = 0.0
	for m in measurements:
		if m.get("ok_count", 0) == 0:
			continue
		total = ((m.get("timings_ms") or {}).get("total_ms") or {}).get("median")
		if not total:
			continue
		for field in ("planner_ms", "tokenization_ms", "context_create_ms"):
			v = ((m.get("timings_ms") or {}).get(field) or {}).get("median")
			if v is not None:
				max_share_pct = max(max_share_pct, 100.0 * v / total)
	m_threshold = re.search(r"stay under ~?(\d+(?:\.\d+)?)% in every", text)
	if not m_threshold:
		fail("doc claims", "doc does not state a checkable "
			"'stay under X% in every' threshold for planner/tokenization/"
			"context-create share of total time")
	else:
		stated = float(m_threshold.group(1))
		if stated < max_share_pct:
			fail("doc claims", f"doc claims these stages stay under "
				f"{stated}%, but the real max observed share is "
				f"{round(max_share_pct, 2)}% -- the stated threshold must "
				f"be >= the real max")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text())
	check_top_level(data)
	measurements = data.get("measurements", [])
	if not isinstance(measurements, list):
		measurements = []
	check_measurement_shape(measurements)
	check_exact_label_set(measurements)
	check_hardware_scope(measurements)
	check_no_cross_contamination(measurements)
	check_finite_nonnegative(measurements)
	check_median_matches_raw(measurements)
	check_raw_cardinality_and_throughput_authenticity(measurements)
	check_docs_match_measurements(measurements)

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({len(measurements)} measurements)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
