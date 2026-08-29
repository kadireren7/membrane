#!/usr/bin/env python3
"""Validate results/performance-optimization/validation.json: schema, that
every "optimizations" before/after entry's median/percentage arithmetic is
real (recomputable from its own raw runs, never claiming a speedup when
median_after >= median_before), that baseline/optimized runs share the
same resolved configuration and generated-token count, that commit fields
are real 40-hex SHAs, and that every "investigations" entry (Phase 25's
required per-target classification, Section 35 of the Phase 25 task) uses
one of the five sanctioned decision values.

This is a measurement-evidence validator, same one-file-per-concern
convention as every other scripts/verify-*.py in this project. It
validates the COMMITTED evidence file; Phase 25 performed no automated
regeneration script for it (the underlying data is a small, one-off
investigation, not a repeatable sweep -- see docs/performance-
optimization.md for full methodology).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "performance-optimization" / "validation.json"
DOC_PATH = REPO_ROOT / "docs" / "performance-optimization.md"
FAILURES = []
CHECK_COUNT = 0

VALID_DECISIONS = frozenset({
	"OPTIMIZED", "INVESTIGATED_NOT_ACTIONABLE", "DEFERRED_NEEDS_MORE_DATA",
	"REJECTED_LOW_UPSIDE", "REJECTED_TOO_RISKY",
})
REQUIRED_OPT_FIELDS = (
	"id", "target", "baseline_commit", "optimized_commit", "model",
	"backend", "device", "context", "resolved_plan", "raw_baseline_runs",
	"raw_optimized_runs", "median_before", "median_after", "delta_percent",
	"correctness", "notes",
)
EXPECTED_INVESTIGATION_IDS = frozenset({
	"OPT-01", "OPT-02", "OPT-03", "OPT-04", "OPT-05", "OPT-06",
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


@check("validation.json is valid JSON with the expected top-level shape")
def check_top_level(data):
	for field in ("schema_version", "membrane_commit", "generated_at",
			"optimizations", "investigations"):
		if field not in data:
			fail("top-level shape", f"missing required field {field!r}")
	if data.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version "
			f"{data.get('schema_version')!r}")
	commit = data.get("membrane_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
		fail("top-level shape", f"membrane_commit {commit!r} is not a "
			f"40-hex SHA")
	if not isinstance(data.get("optimizations"), list):
		fail("top-level shape", "'optimizations' must be a list "
			"(may be empty -- Section 21/'no safe optimization' is a "
			"sanctioned outcome)")
	if not isinstance(data.get("investigations"), list) \
			or not data["investigations"]:
		fail("top-level shape", "'investigations' must be a non-empty list")


@check("at most two real performance optimizations are claimed "
	"(Section 21 of the Phase 25 task)")
def check_optimization_cap(optimizations):
	if len(optimizations) > 2:
		fail("optimization cap", f"{len(optimizations)} entries in "
			f"'optimizations' -- Phase 25 may implement at most two real "
			f"performance optimizations")


@check("every 'optimizations' entry has the full required schema, real "
	"commit provenance, and matching baseline/optimized configuration")
def check_optimization_schema(optimizations):
	for opt in optimizations:
		oid = opt.get("id", "<no id>")
		for field in REQUIRED_OPT_FIELDS:
			if field not in opt:
				fail(oid, f"missing required field {field!r}")
		for field in ("baseline_commit", "optimized_commit"):
			v = opt.get(field, "")
			if not re.fullmatch(r"[0-9a-f]{40}", v or ""):
				fail(oid, f"{field} {v!r} is not a 40-hex SHA")
		if not opt.get("device"):
			fail(oid, "missing/empty 'device' -- every optimization claim "
				"must be hardware-scoped, never generalized")
		base_cfg = opt.get("resolved_plan")
		if not isinstance(base_cfg, dict) or not base_cfg:
			fail(oid, "'resolved_plan' must be a non-empty object -- the "
				"SAME resolved configuration must be documented for both "
				"the baseline and optimized run")


@check("every 'optimizations' entry's median/delta arithmetic is real, "
	"and no entry claims a speedup when median_after >= median_before")
def check_optimization_arithmetic(optimizations):
	for opt in optimizations:
		oid = opt.get("id", "<no id>")
		base_runs = opt.get("raw_baseline_runs")
		opt_runs = opt.get("raw_optimized_runs")
		if not isinstance(base_runs, list) or not base_runs \
				or not isinstance(opt_runs, list) or not opt_runs:
			fail(oid, "raw_baseline_runs/raw_optimized_runs must be "
				"non-empty lists")
			continue
		if len(base_runs) != len(opt_runs):
			fail(oid, f"raw_baseline_runs has {len(base_runs)} sample(s) "
				f"but raw_optimized_runs has {len(opt_runs)} -- same "
				f"repeat count expected for a fair before/after "
				f"comparison")
		gen_before = opt.get("generated_tokens_baseline")
		gen_after = opt.get("generated_tokens_optimized")
		if gen_before is not None and gen_after is not None \
				and gen_before != gen_after:
			fail(oid, f"generated_tokens differs between baseline "
				f"({gen_before}) and optimized ({gen_after}) runs -- not "
				f"a fair comparison")
		real_median_before = statistics.median(base_runs)
		real_median_after = statistics.median(opt_runs)
		claimed_before = opt.get("median_before")
		claimed_after = opt.get("median_after")
		if claimed_before is None or abs(claimed_before - real_median_before) \
				> max(0.01, 0.001 * real_median_before):
			fail(oid, f"median_before {claimed_before!r} does not match "
				f"median(raw_baseline_runs) = {real_median_before!r}")
		if claimed_after is None or abs(claimed_after - real_median_after) \
				> max(0.01, 0.001 * real_median_after):
			fail(oid, f"median_after {claimed_after!r} does not match "
				f"median(raw_optimized_runs) = {real_median_after!r}")
		if claimed_before and claimed_after is not None:
			real_delta = (claimed_before - claimed_after) \
				/ claimed_before * 100.0
			claimed_delta = opt.get("delta_percent")
			if claimed_delta is None \
					or abs(claimed_delta - real_delta) > max(0.1,
						0.02 * abs(real_delta) if real_delta else 0.1):
				fail(oid, f"delta_percent {claimed_delta!r} does not "
					f"match the real (median_before - median_after) / "
					f"median_before * 100 = {round(real_delta, 3)!r}")
			if claimed_after >= claimed_before and claimed_delta \
					and claimed_delta > 0:
				fail(oid, f"median_after ({claimed_after}) >= "
					f"median_before ({claimed_before}) but delta_percent "
					f"({claimed_delta}) claims a positive speedup")
		if "correctness" not in opt or not opt.get("correctness"):
			fail(oid, "missing/empty 'correctness' field -- an "
				"optimization must state how output correctness was "
				"verified, not just that it was faster")


@check("every 'investigations' entry covers all six Phase 25 targets and "
	"uses a sanctioned decision value")
def check_investigations(investigations):
	seen_ids = set()
	for inv in investigations:
		iid = inv.get("id", "<no id>")
		seen_ids.add(iid)
		if not inv.get("target"):
			fail(iid, "missing 'target'")
		decision = inv.get("decision")
		if decision not in VALID_DECISIONS:
			fail(iid, f"decision {decision!r} is not one of "
				f"{sorted(VALID_DECISIONS)} (Section 35 of the Phase 25 "
				f"task)")
		if not inv.get("conclusion") and not inv.get("note_on_classification"):
			fail(iid, "missing a real 'conclusion' explaining the "
				"decision")
	missing = EXPECTED_INVESTIGATION_IDS - seen_ids
	if missing:
		fail("investigations", f"missing required investigation id(s): "
			f"{sorted(missing)}")
	extra = seen_ids - EXPECTED_INVESTIGATION_IDS
	if extra:
		fail("investigations", f"unexpected investigation id(s) not in "
			f"the Phase 25 task's own OPT-01..OPT-06 list: {sorted(extra)}")


@check("OPT-04's raw precision-stability samples are internally "
	"consistent with their own reported medians")
def check_opt04_raw_data(investigations):
	opt04 = next((i for i in investigations if i.get("id") == "OPT-04"), None)
	if opt04 is None:
		return
	raw = opt04.get("raw_data")
	if not isinstance(raw, dict) or not raw:
		fail("OPT-04", "missing 'raw_data' object")
		return
	for mode, block in raw.items():
		for metric in ("decode_tokens_per_second", "prefill_tokens_per_second"):
			m = block.get(metric)
			if not isinstance(m, dict):
				fail("OPT-04", f"{mode}.{metric} missing")
				continue
			samples = m.get("raw_8_repeats_interleaved")
			if not isinstance(samples, list) or len(samples) != 8:
				fail("OPT-04", f"{mode}.{metric} must have exactly 8 raw "
					f"samples (8 warm repeats, Section 16 of the Phase 25 "
					f"task), found "
					f"{len(samples) if isinstance(samples, list) else 'none'}")
				continue
			if any((not isinstance(x, (int, float))) or x <= 0
					for x in samples):
				fail("OPT-04", f"{mode}.{metric} has a non-positive or "
					f"non-numeric sample")
			real_median_all = statistics.median(samples)
			claimed_all = m.get("median_all_8")
			if claimed_all is None or abs(claimed_all - real_median_all) \
					> max(0.01, 0.001 * real_median_all):
				fail("OPT-04", f"{mode}.{metric}.median_all_8 "
					f"{claimed_all!r} does not match "
					f"median(raw_8_repeats_interleaved) = "
					f"{round(real_median_all, 3)!r}")
			real_median_warm = statistics.median(samples[1:])
			claimed_warm = m.get(
				"median_rounds_2_8_excl_first_as_residual_warm_up")
			if claimed_warm is None or abs(claimed_warm - real_median_warm) \
					> max(0.01, 0.001 * real_median_warm):
				fail("OPT-04", f"{mode}.{metric}."
					f"median_rounds_2_8_excl_first_as_residual_warm_up "
					f"{claimed_warm!r} does not match "
					f"median(samples[1:]) = {round(real_median_warm, 3)!r}")


@check("OPT-03's structured planner-substage records sum to their own "
	"reported planner_ms (CodeRabbit review, PR #35)")
def check_opt03_stage_attribution(investigations):
	opt03 = next((i for i in investigations if i.get("id") == "OPT-03"), None)
	if opt03 is None:
		return
	raw = opt03.get("raw_data")
	if not isinstance(raw, dict) or not raw:
		fail("OPT-03", "missing 'raw_data' object -- the stage-attribution "
			"claim in 'correctness_check'/'real_finding' must be backed by "
			"structured, independently-checkable records, not prose alone")
		return
	repeats = raw.get("gpu_requested_repeats")
	if not isinstance(repeats, list) or not repeats:
		fail("OPT-03", "'raw_data.gpu_requested_repeats' must be a "
			"non-empty list")
		repeats = []
	for r in repeats:
		label = f"repeat {r.get('repeat', '<no repeat>')}"
		fields = ("device_enumeration_ms", "gguf_prescan_ms",
			"joint_planner_core_ms", "planner_ms")
		if any(not isinstance(r.get(f), (int, float)) for f in fields):
			fail("OPT-03", f"{label}: one of {fields} is missing/non-numeric")
			continue
		real_sum = round(r["device_enumeration_ms"] + r["gguf_prescan_ms"]
			+ r["joint_planner_core_ms"], 3)
		claimed_sum = r.get("stage_sum_ms")
		if claimed_sum is None or abs(claimed_sum - real_sum) > 0.01:
			fail("OPT-03", f"{label}: stage_sum_ms {claimed_sum!r} does "
				f"not match device_enumeration_ms + gguf_prescan_ms + "
				f"joint_planner_core_ms = {real_sum!r}")
		# Real clock_gettime() calls between the summed sub-timers and the
		# outer planner_ms timer are not perfectly nested (Section 12 of
		# the Phase 25 task: "approximately") -- 1.0 ms absolute or 5%
		# relative, whichever is larger, is the real observed noise band
		# across the 5 committed repeats, not an arbitrarily loose bound.
		tolerance = max(1.0, 0.05 * r["planner_ms"])
		if abs(real_sum - r["planner_ms"]) > tolerance:
			fail("OPT-03", f"{label}: stage sum {real_sum!r} ms does not "
				f"approximately match planner_ms {r['planner_ms']!r} ms "
				f"(tolerance {round(tolerance, 3)} ms)")
	cpu_ref = raw.get("cpu_only_reference")
	if not isinstance(cpu_ref, dict):
		fail("OPT-03", "missing 'raw_data.cpu_only_reference'")
	else:
		for f in ("device_enumeration_ms", "gguf_prescan_ms",
				"joint_planner_core_ms"):
			if cpu_ref.get(f) != 0.0:
				fail("OPT-03", f"cpu_only_reference.{f} = "
					f"{cpu_ref.get(f)!r}, expected exactly 0.0 -- a "
					f"CPU-only request never runs any GPU-requested "
					f"planner sub-step")


@check("docs/performance-optimization.md exists and names every "
	"OPT-01..OPT-06 target")
def check_docs_exist_and_cover_targets():
	if not DOC_PATH.exists():
		fail("doc claims", f"{DOC_PATH} does not exist")
		return
	text = DOC_PATH.read_text()
	for oid in sorted(EXPECTED_INVESTIGATION_IDS):
		if oid not in text:
			fail("doc claims", f"{oid} is never mentioned in "
				f"{DOC_PATH.name}")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text())
	check_top_level(data)
	optimizations = data.get("optimizations", [])
	if not isinstance(optimizations, list):
		optimizations = []
	investigations = data.get("investigations", [])
	if not isinstance(investigations, list):
		investigations = []
	check_optimization_cap(optimizations)
	check_optimization_schema(optimizations)
	check_optimization_arithmetic(optimizations)
	check_investigations(investigations)
	check_opt04_raw_data(investigations)
	check_opt03_stage_attribution(investigations)
	check_docs_exist_and_cover_targets()

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed "
		f"({len(optimizations)} optimizations, "
		f"{len(investigations)} investigations)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
