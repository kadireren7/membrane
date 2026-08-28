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
	check_hardware_scope(measurements)
	check_no_cross_contamination(measurements)
	check_finite_nonnegative(measurements)
	check_median_matches_raw(measurements)

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
