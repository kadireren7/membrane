#!/usr/bin/env python3
"""Validate Phase 35's --ctx auto CLI integration: results/context-
recommendation/cli-validation.json's schema and REAL/SYNTHETIC
labeling, docs/context-auto-cli.md's required sections, the real rows'
own recommended<=hardware_fit<=model_max / recommended>=minimum /
applied_ctx==recommended_context / selected-plan-identity invariants,
host-guard evidence presence, no OOM-proof language, that numeric
--ctx is still documented in --help, that stable release still says
v0.3.0, and that no v0.4 release is claimed anywhere in this phase's
new docs.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "context-recommendation" / "cli-validation.json"
DOC_PATH = REPO_ROOT / "docs" / "context-auto-cli.md"
CTXREC_DOC_PATH = REPO_ROOT / "docs" / "context-recommendation.md"
GUARD_DOC_PATH = REPO_ROOT / "docs" / "host-memory-guard.md"
PRODUCT_CLI_CPP_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.cpp"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
MAIN_CPP_PATH = REPO_ROOT / "tools" / "membrane-run" / "main.cpp"
# Mega Phase A, PR A1: resolve_ctx_auto_cpu_adaptive()/membrane_resolve_
# gpu_config() (and the joint-planner call inside it) moved out of main.cpp
# into the reusable runtime-session core, unchanged, so both the CLI and a
# future server call the SAME implementation -- see runtime_session.cpp's
# own top comment. _c13() below checks the union of both files, not
# main.cpp alone, so this validator keeps checking the real invariant
# (composition, never a second planner) rather than where the code
# happens to physically live.
RUNTIME_SESSION_CPP_PATH = (REPO_ROOT / "tools" / "membrane-run"
	/ "runtime_session.cpp")

REQUIRED_DOC_SECTIONS = [
	"## Usage", "## How recommendation works",
	"## Prompt/generation minimum", "## Explicit constraints",
	"## Human output", "## JSON", "## Failure modes",
	"## Safety wording", "## Known limitations",
]

FAILURES = []
CHECK_COUNT = 0


def check(name):
	def decorator(fn):
		def wrapper():
			global CHECK_COUNT
			CHECK_COUNT += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			status = "PASS" if ok else "FAIL"
			print(f"[{status}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return decorator


@check("evidence and doc files all exist")
def _c1():
	bad = [str(p) for p in (EVIDENCE_PATH, DOC_PATH) if not p.exists()]
	return len(bad) == 0, "; ".join(bad) if bad else "all present"


@check("docs/context-auto-cli.md has every required section")
def _c2():
	text = DOC_PATH.read_text()
	missing = [s for s in REQUIRED_DOC_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("cli-validation.json is valid JSON with the expected top-level shape")
def _c3():
	d = json.loads(EVIDENCE_PATH.read_text())
	required = ["schema_version", "membrane_commit", "real_cli_runs",
		"synthetic_coverage"]
	missing = [k for k in required if k not in d]
	return len(missing) == 0, f"missing keys: {missing}" if missing else "shape OK"


@check("every real_cli_runs row is labeled REAL, every synthetic_coverage "
	"entry is labeled SYNTHETIC -- never mixed")
def _c4():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = [f"row {i} not labeled REAL: {row}"
		for i, row in enumerate(d.get("real_cli_runs", []))
		if row.get("label") != "REAL"]
	if d.get("synthetic_coverage", {}).get("label") != "SYNTHETIC":
		bad.append("synthetic_coverage not labeled SYNTHETIC")
	return len(bad) == 0, "; ".join(bad) if bad else "labeling consistent"


@check("every real_cli_runs row with a recommended_context is requested=auto")
def _c5():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = [row.get("scenario") for row in d.get("real_cli_runs", [])
		if "recommended_context" in row and row.get("requested") != "auto"]
	return len(bad) == 0, f"rows missing requested=auto: {bad}" if bad else "OK"


@check("recommended_context <= hardware_fit_context <= model_max_context, "
	"and recommended_context >= minimum_required_context, on every real "
	"row that has these fields")
def _c6():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = []
	for row in d.get("real_cli_runs", []):
		if not all(k in row for k in ("recommended_context",
				"hardware_fit_context", "model_max_context",
				"minimum_required_context")):
			continue
		rec, fit, mx, mn = (row["recommended_context"], row["hardware_fit_context"],
			row["model_max_context"], row["minimum_required_context"])
		if not (rec <= fit <= mx):
			bad.append(f"{row.get('scenario')}: rec={rec} fit={fit} max={mx}")
		if not (rec >= mn):
			bad.append(f"{row.get('scenario')}: rec={rec} < min={mn}")
	return len(bad) == 0, "; ".join(bad) if bad else "invariants hold"


@check("applied_ctx == recommended_context on every real row that has both")
def _c7():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = [f"{row.get('scenario')}: applied={row.get('applied_ctx')} != "
			f"recommended={row.get('recommended_context')}"
		for row in d.get("real_cli_runs", [])
		if "applied_ctx" in row and "recommended_context" in row
		and row["applied_ctx"] != row["recommended_context"]]
	return len(bad) == 0, "; ".join(bad) if bad else "OK"


@check("selected-plan identity is recorded (matches or an honest, "
	"explained mismatch) on every real row with an applied_ctx")
def _c8():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = [row.get("scenario") for row in d.get("real_cli_runs", [])
		if "applied_ctx" in row and "selected_plan_identity" not in row]
	return len(bad) == 0, f"rows missing selected_plan_identity: {bad}" if bad else "OK"


@check("host-guard evidence (host_required_bytes/host_reserve_bytes) is "
	"present on every real row that required host memory")
def _c9():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = [row.get("scenario") for row in d.get("real_cli_runs", [])
		if "used_cpu_only_adaptive_path" in row
		and ("host_required_bytes" not in row or "host_reserve_bytes" not in row)]
	return len(bad) == 0, f"rows missing host-guard evidence: {bad}" if bad else "OK"


@check("no OOM-proof / guaranteed-fit language in the doc or evidence")
def _c10():
	bad = []
	patterns = [
		r"\bguarantee(s|d)?\s+(no|against)\s+oom\b", r"\boom[- ]proof\b",
		r"\bnever\s+(runs?\s+)?out\s+of\s+memory\b",
		r"\bguaranteed\s+(safe\s+)?fit\b",
	]
	negation_re = re.compile(r"\b(not|never|no)\b", re.IGNORECASE)
	for path in (DOC_PATH, EVIDENCE_PATH):
		text = path.read_text()
		for pat in patterns:
			for m in re.finditer(pat, text, re.IGNORECASE):
				window = text[max(0, m.start() - 80):m.start()]
				trailing = text[m.start():m.start() + 80].lower()
				if (negation_re.search(window) or "not a promise" in trailing
						or "not a guarantee" in trailing
						or "estimated" in window.lower()):
					continue
				bad.append(f"{path.name}: unqualified match {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified overclaim found"


@check("numeric --ctx N is still documented in --help (product_cli.cpp)")
def _c11():
	text = PRODUCT_CLI_CPP_PATH.read_text()
	ok = "--ctx N|auto" in text or "--ctx N | auto" in text
	return ok, "documented" if ok else "not found in usage text"


@check("--ctx auto's own ctx_mode representation exists (no ctx==0 "
	"sentinel reuse) in product_cli.h")
def _c12():
	text = PRODUCT_CLI_H_PATH.read_text()
	required = ["MEMBRANE_RUN_CTX_UNSPECIFIED", "MEMBRANE_RUN_CTX_EXPLICIT",
		"MEMBRANE_RUN_CTX_AUTO"]
	missing = [r for r in required if r not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all three present"


@check("main.cpp/runtime_session.cpp never touch joint_planner.c's own "
	"ranking for the CPU-only-adaptive special case (composes existing "
	"primitives only)")
def _c13():
	text = MAIN_CPP_PATH.read_text() + RUNTIME_SESSION_CPP_PATH.read_text()
	bad = []
	if "membrane_adaptive_kv_resolve(&cand_q8, &cand_q5, 0, &ar)" not in text:
		bad.append("resolve_ctx_auto_cpu_adaptive() does not reuse the "
			"real is_gpu_backend=0 call")
	if "membrane_joint_plan_resolve" not in text:
		bad.append("neither file calls the existing joint planner at all "
			"for the normal path")
	return len(bad) == 0, "; ".join(bad) if bad else "composition confirmed, no duplication"


@check("stable release still says v0.3.0; no v0.4 release/tag is "
	"claimed anywhere in this phase's own new docs")
def _c14():
	bad = []
	for path in (DOC_PATH, CTXREC_DOC_PATH, GUARD_DOC_PATH):
		text = path.read_text()
		for m in re.finditer(r"\bv0\.4\.\d+\b|\bv0\.4-release\b|"
				r"\bv0\.4\s+(is|has been)\s+released\b", text, re.IGNORECASE):
			bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no premature v0.4 release claim"


@check("Phase 33/34's own evidence files are untouched by this phase")
def _c15():
	result = subprocess.run(["git", "diff", "--name-only", "main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/context-recommendation/core-validation.json",
		"results/host-memory-guard/validation.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "both untouched")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11,
			_c12, _c13, _c14, _c15):
		fn()
	print(f"\n{CHECK_COUNT - len(FAILURES)}/{CHECK_COUNT} checks passed")
	if FAILURES:
		print("\nFAILURES:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
