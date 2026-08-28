#!/usr/bin/env python3
"""Validate results/planner-accuracy/measurements.json: schema, that error
fields are only populated when their real inputs are both present (never a
fabricated error from a missing observation), that GPU fields are never
populated for a CPU-only measurement, and that no field claims a GPU "peak"
this measurement method cannot actually provide (see docs/planner-accuracy.md
-- the observed GPU figure is a single before/after driver-reported free-heap
re-query, never a continuous peak sample).

This is a measurement-evidence validator, separate from
scripts/verify-results.py and scripts/verify-compatibility.py. Kept as a
genuinely separate script/file on purpose (same reasoning as Phase 18's
compatibility validator); verify-results.py runs this one too, as another
unnumbered final step.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "planner-accuracy" / "measurements.json"
FAILURES = []
CHECK_COUNT = 0

VALID_OUTCOMES = [
	"FIT", "MODEL_LOAD_FAILED", "CONTEXT_CREATION_OR_GENERATION_FAILED",
	"BACKEND_ALLOCATION_OR_COMPAT_REJECTED", "UNKNOWN_FAILURE",
]
VALID_BACKENDS = ["cpu", "vulkan"]

# Fields that are real physical quantities and can never be negative.
# Deliberately excludes anything under "errors" (a signed delta) and
# anything under "outcome" (exit_code/wall_seconds have their own checks).
NONNEGATIVE_FIELD_PATHS = [
	("planner", "available_gpu_bytes"), ("planner", "reserved_gpu_bytes"),
	("planner", "estimated_weight_gpu_bytes"),
	("planner", "estimated_kv_gpu_bytes"),
	("planner", "estimated_total_gpu_bytes"),
	("planner", "estimated_host_kv_bytes"),
	("observed", "gpu_free_before_bytes"), ("observed", "gpu_free_after_bytes"),
	("observed", "host_rss_after_model_load_kb"),
	("observed", "host_rss_after_context_kb"),
	("observed", "host_rss_peak_kb"),
]


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
			fail("top-level shape", f"missing required field '{field}'")
	if data.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version {data.get('schema_version')!r}")
	if not isinstance(data.get("measurements"), list) or len(data["measurements"]) == 0:
		fail("top-level shape", "'measurements' must be a non-empty list")
	commit = data.get("membrane_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
		fail("top-level shape", f"membrane_commit {commit!r} is not a 40-hex SHA")


@check("every measurement has label/backend/outcome and no impossible negative byte/kb value")
def check_measurement_shape(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		if m.get("backend") not in VALID_BACKENDS:
			fail(label, f"backend {m.get('backend')!r} not in {VALID_BACKENDS}")
		outcome = (m.get("outcome") or {}).get("classification")
		if outcome not in VALID_OUTCOMES:
			fail(label, f"outcome.classification {outcome!r} not in {VALID_OUTCOMES}")
		for section, field in NONNEGATIVE_FIELD_PATHS:
			v = (m.get(section) or {}).get(field)
			if v is not None and v < 0:
				fail(label, f"{section}.{field} is negative ({v}) -- a real "
					f"byte/kb count can never be negative")


@check("GPU fields are never populated for a CPU-backend measurement")
def check_cpu_has_no_gpu_fields(measurements):
	gpu_planner_fields = ("estimated_weight_gpu_bytes", "estimated_kv_gpu_bytes",
		"estimated_total_gpu_bytes", "available_gpu_bytes", "reserved_gpu_bytes")
	gpu_observed_fields = ("gpu_free_before_bytes", "gpu_free_after_bytes",
		"gpu_observed_delta_bytes")
	for m in measurements:
		if m.get("backend") != "cpu":
			continue
		label = m.get("label", "<no label>")
		planner = m.get("planner") or {}
		observed = m.get("observed") or {}
		for f in gpu_planner_fields:
			if planner.get(f) is not None:
				fail(label, f"backend=cpu but planner.{f} is populated "
					f"({planner[f]!r}) -- there is no GPU device to estimate")
		for f in gpu_observed_fields:
			if observed.get(f) is not None:
				fail(label, f"backend=cpu but observed.{f} is populated "
					f"({observed[f]!r}) -- there is no GPU device to observe")


@check("error fields are recomputed from their real inputs and match exactly")
def check_errors_recompute_correctly(measurements):
	for m in measurements:
		label = m.get("label", "<no label>")
		planner = m.get("planner") or {}
		observed = m.get("observed") or {}
		errors = m.get("errors") or {}

		est_total = planner.get("estimated_total_gpu_bytes")
		obs_delta = observed.get("gpu_observed_delta_bytes")
		gpu_err = errors.get("gpu_estimate_error_bytes")
		gpu_err_pct = errors.get("gpu_estimate_error_percent")
		if gpu_err is not None and (est_total is None or obs_delta is None):
			fail(label, "errors.gpu_estimate_error_bytes is populated but "
				"planner.estimated_total_gpu_bytes or observed."
				"gpu_observed_delta_bytes is missing -- a fabricated error")
		elif est_total is not None and obs_delta is not None:
			expected_err = obs_delta - est_total
			if gpu_err != expected_err:
				fail(label, f"errors.gpu_estimate_error_bytes={gpu_err!r} does "
					f"not match recomputed value {expected_err} "
					f"(observed_delta - estimated_total)")
			expected_pct = (round(100.0 * expected_err / obs_delta, 3)
				if obs_delta != 0 else None)
			if gpu_err_pct != expected_pct:
				fail(label, f"errors.gpu_estimate_error_percent={gpu_err_pct!r} "
					f"does not match recomputed value {expected_pct!r}")

		# storage.kv_allocated_bytes (-> estimated_host_kv_bytes) and the
		# RSS checkpoints are populated for EVERY run, GPU or CPU -- an
		# RSS delta after context creation on a GPU-KV run reflects
		# host-side driver/context bookkeeping, not the KV cache itself
		# (which lives on the GPU there), so a host-side error is only
		# ever expected when KV genuinely lives in host RAM: backend
		# "cpu", or kv_placement_resolved "cpu" (weights on GPU, KV
		# forced to host). Mirrors measure-planner-accuracy.py's own
		# kv_is_on_host gate -- kept in sync deliberately, not by
		# importing that script, so this validator still catches a
		# regression in either file independently.
		kv_is_on_host = (m.get("backend") == "cpu"
			or m.get("kv_placement_resolved") == "cpu")
		est_kv_bytes = planner.get("estimated_host_kv_bytes")
		rss_delta_kb = observed.get("host_rss_after_context_delta_kb")
		host_err = errors.get("host_kv_estimate_error_kb")
		host_err_pct = errors.get("host_kv_estimate_error_percent")
		if not kv_is_on_host:
			if host_err is not None or host_err_pct is not None:
				fail(label, "errors.host_kv_estimate_error_kb/_percent is "
					"populated but KV does not live in host RAM for this "
					"measurement (backend != cpu and kv_placement_resolved "
					"!= cpu) -- an RSS delta here reflects driver/context "
					"bookkeeping, not the KV cache, and comparing it to the "
					"KV-bytes estimate would be misleading")
		elif host_err is not None and (est_kv_bytes is None or rss_delta_kb is None):
			fail(label, "errors.host_kv_estimate_error_kb is populated but "
				"its real inputs are missing -- a fabricated error")
		elif est_kv_bytes is not None and rss_delta_kb is not None:
			expected_err_kb = round(rss_delta_kb - est_kv_bytes / 1024.0, 1)
			if host_err != expected_err_kb:
				fail(label, f"errors.host_kv_estimate_error_kb={host_err!r} "
					f"does not match recomputed value {expected_err_kb}")
			expected_host_pct = (round(100.0 * expected_err_kb / rss_delta_kb, 3)
				if rss_delta_kb != 0 else None)
			if host_err_pct != expected_host_pct:
				fail(label,
					f"errors.host_kv_estimate_error_percent={host_err_pct!r} "
					f"does not match recomputed value {expected_host_pct!r}")


@check("no field or provenance string claims a continuous GPU peak sample")
def check_no_fabricated_gpu_peak_claim(data, measurements):
	# This schema deliberately has no gpu-peak field at all (see
	# docs/planner-accuracy.md) -- guard against one being silently added
	# without updating this validator and the docs' own disclosure.
	for m in measurements:
		observed = m.get("observed") or {}
		for key in observed:
			if "peak" in key.lower() and "gpu" in key.lower():
				fail(m.get("label", "<no label>"),
					f"observed.{key} claims a GPU peak -- this measurement "
					f"method (a single before/after re-query) cannot "
					f"actually provide one; see docs/planner-accuracy.md")
	method_texts = [
		(m.get("provenance") or {}).get("measurement_method", "")
		for m in measurements
	]
	for label_idx, text in enumerate(method_texts):
		if re.search(r"\bgpu\b.{0,40}\bpeak\b", text, re.IGNORECASE):
			fail(measurements[label_idx].get("label", "<no label>"),
				"provenance.measurement_method claims a GPU peak")


def reject_non_finite_json(constant_name):
	# json.loads() otherwise silently accepts the bare tokens NaN/
	# Infinity/-Infinity as float('nan')/float('inf')/float('-inf') --
	# a NaN in particular would pass every `v < 0` check below (NaN
	# compares false to everything), letting a corrupted/hand-edited
	# byte count through undetected.
	raise ValueError(
		f"measurements.json contains the non-finite JSON constant "
		f"{constant_name!r} -- not valid for a real byte/kb count")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text(),
		parse_constant=reject_non_finite_json)
	check_top_level(data)
	measurements = data.get("measurements", [])
	if not isinstance(measurements, list):
		measurements = []
	check_measurement_shape(measurements)
	check_cpu_has_no_gpu_fields(measurements)
	check_errors_recompute_correctly(measurements)
	check_no_fabricated_gpu_peak_claim(data, measurements)

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
