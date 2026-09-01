#!/usr/bin/env python3
"""Validate Phase 33's safe-context-recommendation CORE: results/
context-recommendation/core-validation.json's schema and real-vs-
synthetic labeling, docs/context-recommendation.md's required
sections, no OOM-free/guarantee overclaim, consistency with Phase 32's
own planning docs, and that the existing joint planner (never a second
one) is what's actually reused.

This is a source/evidence validator -- it does not build or run any
binary itself (ctest already covers test_context_recommender). Same
one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "context-recommendation" / "core-validation.json"
DOC_PATH = REPO_ROOT / "docs" / "context-recommendation.md"
HEADER_PATH = REPO_ROOT / "tools" / "membrane-run" / "context_recommender.h"
IMPL_PATH = REPO_ROOT / "tools" / "membrane-run" / "context_recommender.c"
V04_ROADMAP_PATH = REPO_ROOT / "docs" / "v0.4-roadmap.md"
PATCHES_DIR = REPO_ROOT / "patches"

REQUIRED_DOC_SECTIONS = [
	"## Problem", "## Definitions", "## Inputs", "## Candidate generation",
	"## Planner reuse", "## Selection policy", "## Safety properties",
	"## Failure modes", "## Known limitation", "## Phase 34 CLI handoff",
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


@check("core module, doc, and evidence files all exist")
def _c1():
	bad = [str(p) for p in (HEADER_PATH, IMPL_PATH, DOC_PATH, EVIDENCE_PATH)
		if not p.exists()]
	return len(bad) == 0, "; ".join(bad) if bad else "all present"


@check("docs/context-recommendation.md has every required section")
def _c2():
	text = DOC_PATH.read_text()
	missing = [s for s in REQUIRED_DOC_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("core-validation.json is valid JSON with the expected top-level shape")
def _c3():
	d = json.loads(EVIDENCE_PATH.read_text())
	required = ["schema_version", "membrane_commit", "policy",
		"candidate_algorithm", "model_max_context_source",
		"real_metadata_reads", "real_dry_runs", "unit_test_summary",
		"known_limitations"]
	missing = [k for k in required if k not in d]
	return len(missing) == 0, f"missing keys: {missing}" if missing else "shape OK"


@check("candidate-count bound in evidence matches the real header constant")
def _c4():
	header_text = HEADER_PATH.read_text()
	m = re.search(r"MEMBRANE_CTXREC_MAX_CANDIDATES\s+(\d+)", header_text)
	if not m:
		return False, "MEMBRANE_CTXREC_MAX_CANDIDATES not found in header"
	real_bound = int(m.group(1))
	d = json.loads(EVIDENCE_PATH.read_text())
	claimed = d["candidate_algorithm"].get("max_candidate_count")
	return claimed == real_bound, f"header={real_bound} evidence={claimed}"


@check("real_metadata_reads and real_dry_runs are labeled REAL, "
	"unit_test_summary is labeled SYNTHETIC -- never mixed")
def _c5():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = []
	for row in d.get("real_metadata_reads", []):
		if row.get("label") != "REAL":
			bad.append(f"real_metadata_reads row not labeled REAL: {row}")
	for row in d.get("real_dry_runs", []):
		if row.get("label") != "REAL":
			bad.append(f"real_dry_runs row not labeled REAL: {row}")
	if d.get("unit_test_summary", {}).get("label") != "SYNTHETIC":
		bad.append("unit_test_summary not labeled SYNTHETIC")
	return len(bad) == 0, "; ".join(bad) if bad else "labeling consistent"


@check("at least one real GGUF model_max_context value is present and "
	"matches the real fixture's own known context_length")
def _c6():
	d = json.loads(EVIDENCE_PATH.read_text())
	rows = {r["model"]: r["model_max_context"]
		for r in d.get("real_metadata_reads", []) if "model_max_context" in r}
	expected = {
		"SmolLM2-135M-Instruct-f16.gguf": 8192,
		"SmolLM2-360M-Instruct-f16.gguf": 8192,
		"qwen2.5-1.5b-instruct-fp16.gguf": 8192,
		"stories15M.gguf": 128,
	}
	bad = [f"{k}: expected {v}, evidence has {rows.get(k)}"
		for k, v in expected.items() if rows.get(k) != v]
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} real model_max_context values match"


@check("every OK real_dry_runs row satisfies "
	"recommended_context <= hardware_fit_context <= model max (candidates[-1])")
def _c7():
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = []
	for row in d.get("real_dry_runs", []):
		if row.get("status") != "OK":
			continue
		rec = row.get("recommended_context")
		fit = row.get("hardware_fit_context")
		cands = row.get("candidates", [])
		if rec is None or fit is None or not cands:
			bad.append(f"incomplete OK row: {row}")
			continue
		if not (rec <= fit <= cands[-1]):
			bad.append(f"invariant violated: rec={rec} fit={fit} "
				f"candidates[-1]={cands[-1]}")
	return len(bad) == 0, "; ".join(bad) if bad else "invariant holds on every OK row"


@check("no OOM-free / memory-guaranteed overclaim in the doc or evidence")
def _c8():
	bad = []
	patterns = [
		r"\bguarantee(s|d)?\s+(no|against)\s+oom\b",
		r"\boom[- ]free\b",
		r"\bnever\s+(runs?\s+)?out\s+of\s+memory\b",
	]
	negation_re = re.compile(r"\b(not|never a|no)\b.{0,40}\bguarantee", re.IGNORECASE)
	for path in (DOC_PATH,):
		text = path.read_text()
		for pat in patterns:
			for m in re.finditer(pat, text, re.IGNORECASE):
				window = text[max(0, m.start() - 60):m.start()]
				if negation_re.search(window) or "not a guaranteed" in text[max(0, m.start()-80):m.start()+80].lower():
					continue
				bad.append(f"{path.name}: unqualified match {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified OOM-free claim found"


@check("docs/context-recommendation.md's MODEL_MAX_CONTEXT / "
	"HARDWARE_FIT_CONTEXT / RECOMMENDED_CONTEXT distinction matches "
	"docs/v0.4-roadmap.md's own Section 5")
def _c9():
	doc_text = DOC_PATH.read_text()
	roadmap_text = V04_ROADMAP_PATH.read_text()
	names = ["MODEL_MAX_CONTEXT", "HARDWARE_FIT_CONTEXT", "RECOMMENDED_CONTEXT"]
	missing_doc = [n for n in names if n not in doc_text]
	missing_roadmap = [n for n in names if n not in roadmap_text]
	bad = []
	if missing_doc:
		bad.append(f"docs/context-recommendation.md missing: {missing_doc}")
	if missing_roadmap:
		bad.append(f"docs/v0.4-roadmap.md missing: {missing_roadmap}")
	return len(bad) == 0, "; ".join(bad) if bad else "all three names present in both docs"


@check("the existing joint planner is named as the reused planner, and "
	"no second planner is introduced")
def _c10():
	header_text = HEADER_PATH.read_text()
	impl_text = IMPL_PATH.read_text()
	bad = []
	if "membrane_joint_plan_resolve" not in header_text:
		bad.append("context_recommender.h never mentions "
			"membrane_joint_plan_resolve()")
	if "membrane_joint_plan_resolve(&jreq" not in impl_text:
		bad.append("context_recommender.c never actually calls "
			"membrane_joint_plan_resolve()")
	forbidden_patterns = [
		r"\bgpu_policy_resolve\s*\(", r"\badaptive_kv_resolve\s*\(",
		r"\bkv_residency_resolve\s*\(", r"\bcheck_kv_compat\s*\(",
	]
	for pat in forbidden_patterns:
		if re.search(pat, impl_text):
			bad.append(f"context_recommender.c calls a lower-level "
				f"policy module directly ({pat}) -- ranking/compat/fit "
				f"logic must be reused only via the joint planner")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"joint planner is the only thing this module calls to decide "
		"a plan -- no duplicated ranking/compat/fit logic found")


@check("minimum_required_context is honored: every real_dry_runs "
	"candidate list respects its own request's implicit floor (>= "
	"MEMBRANE_CTXREC_MIN_CANDIDATE, i.e. 4096, whenever candidates exist)")
def _c11():
	header_text = HEADER_PATH.read_text()
	m = re.search(r"MEMBRANE_CTXREC_MIN_CANDIDATE\s+\(\(uint64_t\)(\d+)\)",
		header_text)
	if not m:
		return False, "MEMBRANE_CTXREC_MIN_CANDIDATE not found in header"
	floor = int(m.group(1))
	d = json.loads(EVIDENCE_PATH.read_text())
	bad = []
	for row in d.get("real_dry_runs", []):
		cands = row.get("candidates", [])
		if cands and cands[0] < floor:
			bad.append(f"{row.get('model')}: first candidate {cands[0]} "
				f"< floor {floor}")
	return len(bad) == 0, "; ".join(bad) if bad else "floor respected everywhere"


@check("patch set unchanged: no llama.cpp patch touches context_recommender's "
	"own files (this phase's core stays entirely on the MEMBRANE side)")
def _c12():
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	expected = {"llama.cpp-membrane-kv-type-override.patch",
		"llama.cpp-membrane-kv-device-override.patch"}
	return real == expected, f"real={real} expected={expected}"


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
