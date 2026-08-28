#!/usr/bin/env python3
"""Validate results/auto-fallback/validation.json: schema, candidate-index/
attempt-ordering sanity, explicit-constraint preservation, that REAL runs
never claim a failure class this project's real apply adapter cannot
actually prove (see docs/auto-fallback.md -- llama.h exposes no error code
from llama_model_load_from_file()/llama_init_from_model(), so GPU_OOM_
CONFIRMED/HOST_OOM_CONFIRMED/DEVICE_LOST/BACKEND_ALLOCATION_FAILED are
taxonomy slots this project cannot honestly populate today), and that
"REAL" vs "SIMULATED" provenance is never blurred.

Separate from scripts/verify-planner-accuracy.py and
scripts/verify-compatibility.py -- same one-evidence-file-per-validator
convention this project already uses.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "auto-fallback" / "validation.json"
FAILURES = []
CHECK_COUNT = 0

MAX_AUTO_ATTEMPTS = 3
VALID_LABELS = ("REAL", "SIMULATED")
VALID_FINAL_STATUS = ("success", "exhausted", "cleanup_blocked", "not_applicable")
# Section 21/2 of the Phase 21 task: these classes require proof this
# project's real apply adapter has no API surface to obtain (llama.h
# returns a bare NULL, never an error code) -- a REAL run must never
# claim one.
UNPROVABLE_FAILURE_CLASSES = (
	"GPU_OOM_CONFIRMED", "HOST_OOM_CONFIRMED", "DEVICE_LOST",
	"BACKEND_ALLOCATION_FAILED",
)


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
	for field in ("schema_version", "label", "runs"):
		if field not in data:
			fail("top-level shape", f"missing required field '{field}'")
	if data.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version {data.get('schema_version')!r}")
	if data.get("label") not in VALID_LABELS:
		fail("top-level shape", f"top-level label {data.get('label')!r} not in {VALID_LABELS}")
	if not isinstance(data.get("runs"), list) or len(data["runs"]) == 0:
		fail("top-level shape", "'runs' must be a non-empty list")
	base = data.get("membrane_base_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", base or ""):
		fail("top-level shape", f"membrane_base_commit {base!r} is not a 40-hex SHA")


@check("every run has a valid label and, if a fallback object exists, a valid final_status")
def check_run_shape(runs):
	for r in runs:
		rid = r.get("id", "<no id>")
		if r.get("label") not in VALID_LABELS:
			fail(rid, f"label {r.get('label')!r} not in {VALID_LABELS}")
		fb = r.get("fallback")
		if fb is None:
			# A run with no fallback object at all is only legitimate when
			# the planner itself never ran (Section 26's Qwen2 regression
			# case) -- never for a run that actually reached apply time.
			if r.get("planner_used") not in (False, None):
				fail(rid, "no 'fallback' object present but planner_used "
					"is not false -- a run that reached the planner must "
					"report its fallback outcome")
			continue
		if fb.get("final_status") not in VALID_FINAL_STATUS:
			fail(rid, f"fallback.final_status {fb.get('final_status')!r} "
				f"not in {VALID_FINAL_STATUS}")


@check("a not_applicable run (--plan-only) never pretends a runtime attempt happened")
def check_not_applicable_shape(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None or fb.get("final_status") != "not_applicable":
			continue
		rid = r.get("id", "<no id>")
		if fb.get("attempted") is not False:
			fail(rid, "final_status=not_applicable but attempted is not false")
		if fb.get("attempt_count", 0) != 0:
			fail(rid, f"final_status=not_applicable but attempt_count is "
				f"{fb.get('attempt_count')!r}, not 0")
		if fb.get("attempts"):
			fail(rid, "final_status=not_applicable but 'attempts' is "
				"non-empty -- --plan-only never applies anything")


@check("attempt_count never exceeds MEMBRANE_MAX_AUTO_ATTEMPTS")
def check_max_attempts(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None:
			continue
		rid = r.get("id", "<no id>")
		ac = fb.get("attempt_count", 0)
		if ac > MAX_AUTO_ATTEMPTS:
			fail(rid, f"attempt_count {ac} exceeds the {MAX_AUTO_ATTEMPTS}-"
				f"attempt bound")
		real_apply_calls = sum(1 for a in fb.get("attempts", [])
			if a.get("apply_started"))
		if real_apply_calls != ac:
			fail(rid, f"attempt_count ({ac}) does not match the number of "
				f"attempts entries with apply_started=true ({real_apply_calls})")


@check("candidate indices in each run's attempts are never repeated")
def check_no_duplicate_candidate(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None:
			continue
		rid = r.get("id", "<no id>")
		indices = [a.get("candidate_index") for a in fb.get("attempts", [])]
		if len(indices) != len(set(indices)):
			fail(rid, f"a candidate_index is repeated across attempts: {indices}")
		if indices and indices[0] != fb.get("initial_candidate_index"):
			fail(rid, "the first attempt's candidate_index does not match "
				"initial_candidate_index")


@check("a successful run's final_candidate_index matches its last successful attempt")
def check_final_candidate_matches_last_success(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None or fb.get("final_status") != "success":
			continue
		rid = r.get("id", "<no id>")
		attempts = fb.get("attempts", [])
		successes = [a for a in attempts if a.get("apply_started")
			and a.get("apply_ok")]
		if len(successes) != 1:
			fail(rid, f"a 'success' run must have exactly one successful "
				f"apply attempt, found {len(successes)}")
			continue
		if successes[0].get("candidate_index") != fb.get("final_candidate_index"):
			fail(rid, "final_candidate_index does not match the one "
				"successful attempt's candidate_index")


@check("an exhausted or cleanup_blocked run has no successful attempt")
def check_exhausted_has_no_success(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None or fb.get("final_status") not in ("exhausted", "cleanup_blocked"):
			continue
		rid = r.get("id", "<no id>")
		for a in fb.get("attempts", []):
			if a.get("apply_started") and a.get("apply_ok"):
				fail(rid, f"final_status={fb.get('final_status')!r} but "
					f"candidate {a.get('candidate_index')} reports apply_ok=true")
		if fb.get("final_candidate_index", -1) != -1:
			fail(rid, f"final_status={fb.get('final_status')!r} but "
				f"final_candidate_index is {fb.get('final_candidate_index')!r}, "
				f"not -1")


@check("a REAL run never claims a failure class this project's real adapter cannot prove")
def check_real_runs_never_fabricate_oom(runs):
	for r in runs:
		if r.get("label") != "REAL":
			continue
		fb = r.get("fallback")
		if fb is None:
			continue
		rid = r.get("id", "<no id>")
		for a in fb.get("attempts", []):
			fc = a.get("failure_class")
			if fc in UNPROVABLE_FAILURE_CLASSES:
				fail(rid, f"REAL run claims failure_class={fc!r}, which this "
					f"project's real apply adapter has no API surface to "
					f"honestly prove (see docs/auto-fallback.md)")


@check("skipped attempts never claim they were applied")
def check_skips_never_claim_applied(runs):
	for r in runs:
		fb = r.get("fallback")
		if fb is None:
			continue
		rid = r.get("id", "<no id>")
		for a in fb.get("attempts", []):
			if not a.get("apply_started") and (
					"apply_ok" in a or "failure_class" in a
					or "cleanup_complete" in a):
				fail(rid, f"candidate {a.get('candidate_index')} was never "
					f"applied (apply_started=false) but carries apply-"
					f"result fields -- a skip must not look like an attempt")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text())
	check_top_level(data)
	runs = data.get("runs", [])
	if not isinstance(runs, list):
		runs = []
	check_run_shape(runs)
	check_not_applicable_shape(runs)
	check_max_attempts(runs)
	check_no_duplicate_candidate(runs)
	check_final_candidate_matches_last_success(runs)
	check_exhausted_has_no_success(runs)
	check_real_runs_never_fabricate_oom(runs)
	check_skips_never_claim_applied(runs)

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({len(runs)} runs)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
