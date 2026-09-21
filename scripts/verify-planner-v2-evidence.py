#!/usr/bin/env python3
"""Validate results/planner-v2-evidence/validation.json (Milestone G3):
schema/shape checks, that every feasibility-confusion-matrix entry uses
a real, project-consistent classification, that no byte/kb/percentage
field is an impossible negative where negatives are never physically
valid, that the performance-floor threshold was defined BEFORE the
throughput numbers it's evaluated against (both are present, this just
checks internal consistency of the recorded classification), and that
the commit hash recorded is a real 40-hex SHA.

This is a pure schema/invariant validator over the already-committed,
already-curated JSON artifact -- it never re-runs any real workload,
downloads a model, or touches the network, so it is safe for CI (see
.github/workflows/ci.yml).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "planner-v2-evidence" / "validation.json"
FAILURES = []
CHECK_COUNT = 0

VALID_CLASSIFICATIONS = {
	"TRUE_FEASIBLE", "FALSE_FEASIBLE", "TRUE_REJECT", "UNVERIFIED_REJECT",
	"FALSE_REJECT",
}
REQUIRED_TOP_LEVEL = (
	"schema_version", "base_commit", "host", "hypotheses",
	"performance_floor_definition", "sibling_scaling_error",
	"feasibility_confusion_matrix", "context_capacity_experiment",
	"performance_floor_evaluation", "latency", "gpu_evidence",
	"policy_evaluation", "objective_modes_decision",
	"execution_integration_readiness", "limitations", "no_ollama_vllm_work",
)
VALID_INTEGRATION_READINESS = {
	"READY_FOR_READ_ONLY_ONLY", "READY_FOR_OPT_IN_EXECUTION",
	"READY_FOR_DEFAULT_EXECUTION",
}


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


@check("validation.json has the expected top-level shape")
def check_top_level(data):
	for field in REQUIRED_TOP_LEVEL:
		if field not in data:
			fail("top-level shape", f"missing required field '{field}'")
	commit = data.get("base_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
		fail("top-level shape", f"base_commit {commit!r} is not a 40-hex SHA")


@check("performance floor threshold was declared BEFORE the results it classifies")
def check_floor_predeclared(data):
	floor = data.get("performance_floor_definition", {})
	if floor.get("defined_before_seeing_throughput_results") is not True:
		fail("performance floor", "defined_before_seeing_throughput_results "
			"must be explicitly true (Part 8 of the G3 task: do not invent "
			"a marketing-friendly threshold after seeing results)")
	if "text" not in floor or "%" not in floor["text"]:
		fail("performance floor", "threshold text must state a concrete "
			"percentage, not a vague claim")


@check("every feasibility_confusion_matrix entry uses a real, project-consistent classification")
def check_confusion_matrix(data):
	entries = data.get("feasibility_confusion_matrix", [])
	if not entries:
		fail("confusion matrix", "must be non-empty -- G3 requires at least "
			"one real feasibility validation")
	for e in entries:
		case = e.get("case", "<no case>")
		cls = e.get("classification")
		if cls not in VALID_CLASSIFICATIONS:
			fail(case, f"classification {cls!r} not in {sorted(VALID_CLASSIFICATIONS)}")
		if "plan_feasible" not in e or "real_load_generate_succeeded" not in e:
			fail(case, "must record both plan_feasible and "
				"real_load_generate_succeeded booleans")
		plan_feasible = e.get("plan_feasible")
		real_ok = e.get("real_load_generate_succeeded")
		# Internal consistency between the two booleans and the label.
		if cls == "TRUE_FEASIBLE" and not (plan_feasible and real_ok):
			fail(case, "TRUE_FEASIBLE requires plan_feasible=true and "
				"real_load_generate_succeeded=true")
		if cls == "TRUE_REJECT" and (plan_feasible or real_ok):
			fail(case, "TRUE_REJECT requires plan_feasible=false and "
				"real_load_generate_succeeded=false")
		if cls == "FALSE_REJECT" and (plan_feasible or not real_ok):
			fail(case, "FALSE_REJECT requires plan_feasible=false and "
				"real_load_generate_succeeded=true")
		if cls == "FALSE_FEASIBLE" and (not plan_feasible or real_ok):
			fail(case, "FALSE_FEASIBLE requires plan_feasible=true and "
				"real_load_generate_succeeded=false")


@check("execution_integration_readiness decision is one of the three sanctioned values")
def check_integration_decision(data):
	decision = (data.get("execution_integration_readiness") or {}).get("decision")
	if decision not in VALID_INTEGRATION_READINESS:
		fail("execution_integration_readiness",
			f"{decision!r} not in {sorted(VALID_INTEGRATION_READINESS)}")


@check("sibling_scaling_error pairs report a signed error_pct for every predicted field")
def check_sibling_error(data):
	sib = data.get("sibling_scaling_error") or {}
	pairs = sib.get("scaling_error_pairs", [])
	if not pairs:
		fail("sibling scaling error", "scaling_error_pairs must be "
			"non-empty -- H5 requires a measured, real comparison")
	for p in pairs:
		label = f"{p.get('base_quant')}->{p.get('target_quant')}"
		err = p.get("error_pct", {})
		for field in ("bytes_per_layer", "output_role_bytes", "total_weight_bytes"):
			if field not in err or not isinstance(err[field], (int, float)):
				fail(label, f"error_pct.{field} missing or not numeric")


@check("no byte/kb/tok-per-s field anywhere in raw_evidence is an impossible negative")
def check_no_negative_physical_quantities(data):
	def walk(obj, path):
		if isinstance(obj, dict):
			for k, v in obj.items():
				if isinstance(v, (int, float)) and not isinstance(v, bool):
					lk = k.lower()
					if any(s in lk for s in ("_bytes", "_kb", "tok_per_s",
							"tok_per_sec", "rss", "peak_")) and v < 0:
						fail(f"{path}.{k}", f"negative physical quantity: {v}")
				else:
					walk(v, f"{path}.{k}")
		elif isinstance(obj, list):
			for i, v in enumerate(obj):
				walk(v, f"{path}[{i}]")

	walk(data.get("raw_evidence", {}), "raw_evidence")


@check("limitations is non-empty (G3 requires disclosed, not hidden, limitations)")
def check_limitations_present(data):
	if not data.get("limitations"):
		fail("limitations", "must be a non-empty list")


@check("no_ollama_vllm_work is an explicit confirmation string, not empty/missing")
def check_no_adapter_work(data):
	v = data.get("no_ollama_vllm_work", "")
	if "confirmed" not in v.lower():
		fail("no_ollama_vllm_work", "must explicitly confirm no Ollama/vLLM "
			"work was started")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text())
	check_top_level(data)
	check_floor_predeclared(data)
	check_confusion_matrix(data)
	check_integration_decision(data)
	check_sibling_error(data)
	check_no_negative_physical_quantities(data)
	check_limitations_present(data)
	check_no_adapter_work(data)

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed")
	return 0


if __name__ == "__main__":
	sys.exit(main())
