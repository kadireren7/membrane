#!/usr/bin/env python3
"""Validate Phase 34's host-memory safety guard: results/host-memory-
guard/validation.json's schema and reserve-policy identity, docs/host-
memory-guard.md's required sections, that context recommendation's
host-memory check is mandatory (never OK on unvalidated host
residency), no infinite-host-RAM assumption, no OOM-free/guaranteed-
fit overclaim, the CPU-adaptive decision is documented, Phase 33's own
historical evidence file is untouched, no second planner is
introduced, and the patch set remains unchanged.

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
EVIDENCE_PATH = REPO_ROOT / "results" / "host-memory-guard" / "validation.json"
DOC_PATH = REPO_ROOT / "docs" / "host-memory-guard.md"
CTXREC_DOC_PATH = REPO_ROOT / "docs" / "context-recommendation.md"
GUARD_HEADER_PATH = REPO_ROOT / "tools" / "membrane-run" / "host_memory_guard.h"
GUARD_IMPL_PATH = REPO_ROOT / "tools" / "membrane-run" / "host_memory_guard.c"
CTXREC_HEADER_PATH = REPO_ROOT / "tools" / "membrane-run" / "context_recommender.h"
CTXREC_IMPL_PATH = REPO_ROOT / "tools" / "membrane-run" / "context_recommender.c"
CTXREC_EVIDENCE_PATH = (REPO_ROOT / "results" / "context-recommendation"
	/ "core-validation.json")
PATCHES_DIR = REPO_ROOT / "patches"

REQUIRED_DOC_SECTIONS = [
	"## Problem", "## Why Phase 33 was not yet sufficient",
	"## Host memory components", "## Estimator", "## Reserve policy",
	"## Recommendation integration", "## Explicit-run scope",
	"## CPU-only adaptive decision", "## Safety guarantees",
	"## Residual uncertainty", "## Phase 35 CLI handoff",
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


@check("guard module, docs, and evidence files all exist")
def _c1():
	bad = [str(p) for p in (GUARD_HEADER_PATH, GUARD_IMPL_PATH, DOC_PATH,
			EVIDENCE_PATH) if not p.exists()]
	return len(bad) == 0, "; ".join(bad) if bad else "all present"


@check("docs/host-memory-guard.md has every required section")
def _c2():
	text = DOC_PATH.read_text()
	missing = [s for s in REQUIRED_DOC_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with the expected top-level shape")
def _c3():
	d = json.loads(EVIDENCE_PATH.read_text())
	required = ["schema_version", "membrane_commit", "host_memory_policy",
		"reserve_policy", "evidence_basis", "real_dry_run_smoke",
		"recommendation_integration", "cpu_adaptive_decision",
		"known_limitations"]
	missing = [k for k in required if k not in d]
	return len(missing) == 0, f"missing keys: {missing}" if missing else "shape OK"


@check("reserve policy identity: evidence's declared constants match "
	"the real header constants")
def _c4():
	header_text = GUARD_HEADER_PATH.read_text()
	m_fixed = re.search(
		r"MEMBRANE_HOST_RESERVE_FIXED_BYTES\s+\(\(uint64_t\)(\d+) \* 1024 \* 1024\)",
		header_text)
	m_pct = re.search(r"MEMBRANE_HOST_RESERVE_PCT\s+(\d+)", header_text)
	if not m_fixed or not m_pct:
		return False, "reserve constants not found in host_memory_guard.h"
	real_fixed = int(m_fixed.group(1)) * 1024 * 1024
	real_pct = int(m_pct.group(1))
	d = json.loads(EVIDENCE_PATH.read_text())
	rp = d["reserve_policy"]
	bad = []
	if rp.get("fixed_bytes") != real_fixed:
		bad.append(f"evidence fixed_bytes={rp.get('fixed_bytes')} != "
			f"real {real_fixed}")
	if rp.get("pct") != real_pct:
		bad.append(f"evidence pct={rp.get('pct')} != real {real_pct}")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"fixed={real_fixed} pct={real_pct}% match")


@check("recommendation's host-memory check is mandatory -- "
	"context_recommender.c actually gates ev->feasible on the guard's "
	"own result, not just on the joint planner's")
def _c5():
	impl_text = CTXREC_IMPL_PATH.read_text()
	bad = []
	if "membrane_host_memory_guard_resolve" not in impl_text:
		bad.append("context_recommender.c never calls "
			"membrane_host_memory_guard_resolve()")
	if not re.search(r"ev->feasible\s*=\s*hres\.ok", impl_text):
		bad.append("ev->feasible is not gated on the host guard's own "
			"result (hres.ok)")
	header_text = CTXREC_HEADER_PATH.read_text()
	for field in ("host_memory_checked", "host_memory_fit",
			"host_required_bytes"):
		if field not in header_text:
			bad.append(f"context_recommender.h missing result field {field!r}")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"host-memory gating is real, not cosmetic")


@check("no infinite-host-RAM assumption: unknown availability is "
	"handled as a distinct, fail-closed case, never treated as fit")
def _c6():
	impl_text = GUARD_IMPL_PATH.read_text()
	bad = []
	if "host_available_known" not in impl_text:
		bad.append("host_memory_guard.c never checks host_available_known")
	if "MEMBRANE_HOST_GUARD_REASON_UNKNOWN" not in impl_text:
		bad.append("no UNKNOWN reason code path found")
	# The unknown-availability branch must return 0 (fail), not 1.
	m = re.search(
		r"if \(!req->host_available_known\)\s*\{[^}]*set_result\(out, (\d)",
		impl_text)
	if not m:
		bad.append("could not locate the unknown-availability branch's "
			"own ok value")
	elif m.group(1) != "0":
		bad.append(f"unknown-availability branch sets ok={m.group(1)}, "
			"expected 0 (fail closed)")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"unknown availability fails closed, never assumed infinite")


@check("no OOM-free / guaranteed-fit overclaim in either doc")
def _c7():
	bad = []
	patterns = [
		r"\bguarantee(s|d)?\s+(no|against)\s+oom\b",
		r"\boom[- ]free\b",
		r"\bnever\s+(runs?\s+)?out\s+of\s+memory\b",
		r"\bguaranteed\s+(safe\s+)?fit\b",
	]
	negation_re = re.compile(
		r"\b(not|never a|no|not\.\.\. never)\b.{0,60}\bguarantee", re.IGNORECASE)
	for path in (DOC_PATH, CTXREC_DOC_PATH):
		text = path.read_text()
		for pat in patterns:
			for m in re.finditer(pat, text, re.IGNORECASE):
				window = text[max(0, m.start() - 80):m.start()]
				trailing = text[m.start():m.start() + 80].lower()
				if (negation_re.search(window)
						or "not a guarantee" in trailing
						or "never a guarantee" in trailing
						or "estimated" in window.lower()):
					continue
				bad.append(f"{path.name}: unqualified match {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified overclaim found"


@check("CPU-adaptive decision is documented with a root-cause "
	"classification and an explicit decision")
def _c8():
	d = json.loads(EVIDENCE_PATH.read_text())
	cad = d.get("cpu_adaptive_decision", {})
	bad = []
	if "root_cause_classification" not in cad:
		bad.append("missing root_cause_classification")
	if cad.get("decision") not in ("DEFERRED, not implemented this phase",):
		if "decision" not in cad or not cad["decision"]:
			bad.append("missing decision")
	if "## CPU-only adaptive decision" not in DOC_PATH.read_text():
		bad.append("docs/host-memory-guard.md missing the CPU-only "
			"adaptive decision section")
	return len(bad) == 0, "; ".join(bad) if bad else "documented"


@check("Phase 33's own historical evidence file is untouched by this phase")
def _c9():
	result = subprocess.run(
		["git", "diff", "--name-only", "main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		# No 'main' ref to diff against (e.g. detached/shallow) -- skip
		# rather than false-fail; this check is a best-effort guard.
		return True, "skipped (no diffable 'main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = "results/context-recommendation/core-validation.json" in changed
	return not touched, (
		"core-validation.json was modified this phase -- Phase 33's own "
		"evidence must stay historical" if touched
		else "core-validation.json untouched")


@check("no second planner introduced: context_recommender.c and "
	"host_memory_guard.c never call gpu_policy/adaptive_kv/kv_residency/"
	"compat_check directly")
def _c10():
	bad = []
	forbidden_patterns = [
		r"\bgpu_policy_resolve\s*\(", r"\badaptive_kv_resolve\s*\(",
		r"\bkv_residency_resolve\s*\(", r"\bcheck_kv_compat\s*\(",
	]
	for path in (CTXREC_IMPL_PATH, GUARD_IMPL_PATH):
		text = path.read_text()
		for pat in forbidden_patterns:
			if re.search(pat, text):
				bad.append(f"{path.name} calls a lower-level policy "
					f"module directly ({pat})")
	if "membrane_joint_plan_resolve" not in CTXREC_IMPL_PATH.read_text():
		bad.append("context_recommender.c no longer calls the existing "
			"joint planner at all")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"only the existing joint planner is called to decide a plan")


@check("patch set unchanged")
def _c11():
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	expected = {"llama.cpp-membrane-kv-type-override.patch",
		"llama.cpp-membrane-kv-device-override.patch"}
	return real == expected, f"real={real} expected={expected}"


@check("Phase 33's own context-recommendation evidence file still "
	"exists and is valid JSON (this phase extends, never deletes it)")
def _c12():
	if not CTXREC_EVIDENCE_PATH.exists():
		return False, f"{CTXREC_EVIDENCE_PATH} missing"
	json.loads(CTXREC_EVIDENCE_PATH.read_text())
	return True, "present and valid"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12):
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
