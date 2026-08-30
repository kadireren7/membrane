#!/usr/bin/env python3
"""Validate release metadata consistency for the CURRENT RC3 release-prep
work: MEMBRANE_VERSION is 0.3.0-rc3, the release doc and readiness
evidence exist and are self-consistent, the CURRENT (live)
docs/compatibility.json counts match what docs/release-v0.3.0-rc3.md
claims, the Qwen2 support claim has direct Phase 26 evidence behind it,
no stable v0.3.0 claim exists, external validation is still disclosed as
incomplete, CITATION.cff still targets stable v0.2.0, and no performance/
speedup claim appears anywhere in RC3's own release-facing docs (Phase 25
shipped zero speed-changing code).

Deliberately small, same one-file-per-concern convention as every other
scripts/verify-*.py -- this validates CURRENT/live state (unlike
scripts/verify-release-readiness.py, which stays pinned to RC2's own
frozen historical facts and is never touched by this file).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "0.3.0-rc3"
FAILURES = []
CHECK_COUNT = 0

RC3_FACING_DOCS = [
	"README.md",
	"docs/release-v0.3.0-rc3.md",
	"docs/compat-expansion.md",
	"docs/compatibility.md",
	"docs/performance-optimization.md",
	"docs/performance-profiling.md",
]

# Same list as scripts/verify-release-readiness.py's OVERCLAIM_PATTERNS,
# plus the performance-claim phrases Section 11 of the Phase 27 task
# explicitly forbids for this cycle (Phase 25 shipped zero speed-
# changing code).
OVERCLAIM_PATTERNS = [
	r"supports all llama\.cpp models",
	r"supports all vulkan gpus",
	r"all qwen2 models",
	r"automatic optimal configuration",
	r"guaranteed fit",
	r"zero oom",
	r"production[- ]ready",
	r"performance optimized",
	r"improved inference speed",
]

READINESS_PATH = REPO_ROOT / "results" / "release-v0.3.0-rc3" / "readiness.json"
RELEASE_DOC_PATH = REPO_ROOT / "docs" / "release-v0.3.0-rc3.md"
COMPAT_JSON_PATH = REPO_ROOT / "docs" / "compatibility.json"
COMPAT_EXPANSION_PATH = REPO_ROOT / "results" / "compat-expansion" / "validation.json"
CITATION_PATH = REPO_ROOT / "CITATION.cff"


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


@check("MEMBRANE_VERSION == 0.3.0-rc3 (live product_cli.h)")
def check_version():
	path = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', path.read_text())
	found = m.group(1) if m else None
	if found != EXPECTED_VERSION:
		fail("version", f"MEMBRANE_VERSION is {found!r}, expected "
			f"{EXPECTED_VERSION!r}")


@check("docs/release-v0.3.0-rc3.md exists")
def check_release_doc_exists():
	if not RELEASE_DOC_PATH.exists():
		fail("release doc", f"{RELEASE_DOC_PATH} does not exist")


@check("results/release-v0.3.0-rc3/readiness.json exists, is valid JSON, "
	"and its membrane_version/base_release are correct")
def check_readiness_evidence():
	if not READINESS_PATH.exists():
		fail("readiness evidence", f"{READINESS_PATH} does not exist")
		return
	data = json.loads(READINESS_PATH.read_text())
	for field in ("schema_version", "label", "membrane_commit",
			"membrane_version", "base_release"):
		if field not in data:
			fail("readiness evidence", f"missing required field {field!r}")
	if data.get("label") not in ("REAL", "SIMULATED"):
		fail("readiness evidence", f"label {data.get('label')!r} not in "
			f"('REAL', 'SIMULATED')")
	if not re.fullmatch(r"[0-9a-f]{40}", data.get("membrane_commit", "") or ""):
		fail("readiness evidence", f"membrane_commit "
			f"{data.get('membrane_commit')!r} is not a 40-hex SHA")
	if data.get("membrane_version") != EXPECTED_VERSION:
		fail("readiness evidence", f"membrane_version "
			f"{data.get('membrane_version')!r} != {EXPECTED_VERSION!r}")
	if data.get("base_release") != "v0.3.0-rc2":
		fail("readiness evidence", f"base_release "
			f"{data.get('base_release')!r} != 'v0.3.0-rc2'")
	# CodeRabbit review (PR #37): the envelope-field checks above passed
	# even while this file was still a draft (pull_request a placeholder
	# string, ci_runs empty) -- this is the release GATE, so draft
	# evidence must fail it, not silently pass through to merge.
	pr_url = data.get("pull_request", "")
	if not re.fullmatch(
			r"https://github\.com/kadireren7/membrane/pull/\d+", pr_url):
		fail("readiness evidence", f"pull_request {pr_url!r} is not a "
			f"real PR URL (still a placeholder?) -- this must be filled "
			f"in with the real PR before this evidence can gate a merge")
	ci_runs = data.get("ci_runs") or {}
	workflow_runs = ci_runs.get("workflow_runs") or []
	jobs_all_passing = ci_runs.get("jobs_all_passing") or []
	if not workflow_runs or not all(
			isinstance(r, int) for r in workflow_runs):
		fail("readiness evidence", f"ci_runs.workflow_runs "
			f"{workflow_runs!r} is empty or not all-integer -- real "
			f"GitHub Actions run IDs are required before this evidence "
			f"can gate a merge")
	if not jobs_all_passing:
		fail("readiness evidence", "ci_runs.jobs_all_passing is empty -- "
			"the real list of passing CI job names is required before "
			"this evidence can gate a merge")


@check("current live compatibility counts match what "
	"docs/release-v0.3.0-rc3.md claims")
def check_compat_counts_match_doc():
	if not COMPAT_JSON_PATH.exists():
		fail("compat counts", f"{COMPAT_JSON_PATH} does not exist")
		return
	compat = json.loads(COMPAT_JSON_PATH.read_text())
	rows = compat.get("rows", [])
	counts = Counter(r.get("status") for r in rows)
	supported = counts.get("SUPPORTED", 0)
	unsupported = counts.get("UNSUPPORTED", 0)
	not_yet = counts.get("NOT_YET_VALIDATED", 0)
	if not RELEASE_DOC_PATH.exists():
		return
	# Line-wrap-tolerant: markdown prose can wrap "6\nUNSUPPORTED" across
	# a line break -- collapse all whitespace runs to a single space
	# before searching, so this check verifies the real number is
	# stated near the right label rather than demanding one exact
	# literal string that happens to never wrap.
	text = re.sub(r"\s+", " ", RELEASE_DOC_PATH.read_text())
	if f"{len(rows)} total rows" not in text:
		fail("compat counts", f"doc does not state the real "
			f"'{len(rows)} total rows' figure")
	for label, n in (("SUPPORTED", supported), ("UNSUPPORTED", unsupported),
			("NOT_YET_VALIDATED", not_yet)):
		if f"{n} {label}" not in text:
			fail("compat counts", f"doc does not state the real "
				f"'{n} {label}' figure (live docs/compatibility.json has "
				f"{n} {label} row(s))")


EXPECTED_QWEN2_ROW_IDS = frozenset({
	"CE-01", "CE-02", "CE-03", "CE-04", "CE-05", "CE-06", "CE-07", "CE-08",
})
EXPECTED_QWEN2_MODEL = "Qwen2.5-1.5B-Instruct"


@check("the Qwen2 compressed-KV support claim has direct Phase 26 "
	"evidence (results/compat-expansion/validation.json)")
def check_qwen2_claim_has_evidence():
	if not COMPAT_EXPANSION_PATH.exists():
		fail("Qwen2 claim", f"{COMPAT_EXPANSION_PATH} does not exist -- "
			f"the release doc's Qwen2 claim would be unsubstantiated")
		return
	data = json.loads(COMPAT_EXPANSION_PATH.read_text())
	rows = data.get("rows") or []
	# CodeRabbit review (PR #37): a non-empty rows list alone does not
	# prove the SPECIFIC rows this release's own claim cites (CE-01..
	# CE-08) still exist, or that they still cover the exact model the
	# release doc names -- bind the claim to those exact rows, not just
	# "some evidence exists somewhere in this file".
	by_id = {r.get("id"): r for r in rows}
	missing = EXPECTED_QWEN2_ROW_IDS - set(by_id)
	if missing:
		fail("Qwen2 claim", f"results/compat-expansion/validation.json "
			f"is missing required row id(s) {sorted(missing)} -- the "
			f"release doc's Qwen2 claim cites these specific rows")
	for rid in EXPECTED_QWEN2_ROW_IDS & set(by_id):
		row = by_id[rid]
		if row.get("model") != EXPECTED_QWEN2_MODEL:
			fail("Qwen2 claim", f"{rid}.model is {row.get('model')!r}, "
				f"expected {EXPECTED_QWEN2_MODEL!r} -- the release doc's "
				f"claim is scoped to this exact model")
	if not RELEASE_DOC_PATH.exists():
		return
	text = RELEASE_DOC_PATH.read_text()
	if EXPECTED_QWEN2_MODEL not in text:
		fail("Qwen2 claim", "release doc does not name the exact tested "
			"model (Qwen2.5-1.5B-Instruct) -- an unscoped 'Qwen2' claim "
			"would overclaim beyond the real evidence")
	# Cross-check the two specific byte-ratio percentages the release
	# doc states against this same evidence file's own committed
	# kv_bytes fields, so a stale/hand-edited percentage in the doc
	# would be caught, not just the row's existence.
	by_precision = {}
	for rid in ("CE-01", "CE-02", "CE-03"):
		row = by_id.get(rid)
		if row is not None:
			by_precision[row.get("precision")] = row.get("kv_bytes")
	native_kv = by_precision.get("native")
	if native_kv:
		for precision, doc_pct in (("q8", "53.125%"), ("q5", "37.5%")):
			kv = by_precision.get(precision)
			if kv is None:
				continue
			real_pct = round(100 * kv / native_kv, 3)
			doc_val = float(doc_pct.rstrip("%"))
			if abs(real_pct - doc_val) > 0.01:
				fail("Qwen2 claim", f"release doc states {precision} = "
					f"{doc_pct}, but the cited evidence's own kv_bytes "
					f"({kv} / {native_kv}) computes to {real_pct}%")
	if "all Qwen2 models" in text.lower():
		fail("Qwen2 claim", "release doc appears to generalize to 'all "
			"Qwen2 models' -- the evidence only covers "
			"Qwen2.5-1.5B-Instruct")


@check("no stable v0.3.0 claim exists in RC3-facing docs")
def check_no_stable_v030_claim():
	pattern = re.compile(r"v0\.3\.0(?!-rc)")
	for rel in RC3_FACING_DOCS:
		path = REPO_ROOT / rel
		if not path.exists():
			continue
		text = path.read_text()
		for m in pattern.finditer(text):
			fail("stable v0.3.0 scan", f"{rel} contains a bare "
				f"{m.group(0)!r} (no -rcN suffix) -- reads as a stable "
				f"v0.3.0 claim")


@check("no known overclaim or unsupported performance-speedup phrase "
	"appears in RC3-facing docs")
def check_no_overclaim():
	compiled = [re.compile(p, re.IGNORECASE) for p in OVERCLAIM_PATTERNS]
	for rel in RC3_FACING_DOCS:
		path = REPO_ROOT / rel
		if not path.exists():
			continue
		text = path.read_text()
		for pat in compiled:
			for m in pat.finditer(text):
				# Same negation-window heuristic as
				# verify-release-readiness.py's own overclaim scan -- a
				# disclosed/negated mention ("must not say 'faster'",
				# "not 'all Vulkan GPUs'") is the whole point of this
				# project's honesty discipline, not a violation of it.
				# 80 chars, not 40 (verify-release-readiness.py's own
				# single-phrase window): this file's own release doc uses
				# a "must not say X, Y, or Z" list -- the third quoted
				# phrase can sit more than 40 chars past the negation.
				window = text[max(0, m.start() - 80):m.start()].lower()
				if not re.search(r"\bnot\b|\bnever\b|\bno[t]?\b|"
						r"does not|n['’]t\b|must not say", window):
					fail("overclaim scan", f"{rel} contains "
						f"{m.group(0)!r} without an apparent negation "
						f"nearby")


@check("external validation is still disclosed as incomplete")
def check_external_validation_disclosed():
	if not RELEASE_DOC_PATH.exists():
		return
	text = RELEASE_DOC_PATH.read_text()
	if "External validation status" not in text:
		fail("external validation", "release doc has no 'External "
			"validation status' section")
		return
	if "one" not in text.lower() and "1" not in text:
		fail("external validation", "release doc's external-validation "
			"section does not appear to disclose the single-host scope")
	if READINESS_PATH.exists():
		data = json.loads(READINESS_PATH.read_text())
		ev = data.get("external_validation_status", {})
		if ev.get("independent_hosts_validated", -1) != 0:
			fail("external validation", f"readiness.json claims "
				f"independent_hosts_validated="
				f"{ev.get('independent_hosts_validated')!r}, expected 0 "
				f"-- Phase 28 (independent-host validation) has not "
				f"started")


@check("CITATION.cff still targets stable v0.2.0, not bumped to an RC")
def check_citation_unchanged():
	if not CITATION_PATH.exists():
		fail("CITATION.cff", f"{CITATION_PATH} does not exist")
		return
	text = CITATION_PATH.read_text()
	m = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
	found = m.group(1) if m else None
	if found != "0.2.0":
		fail("CITATION.cff", f"version field is {found!r}, expected "
			f"'0.2.0'")


def main():
	check_version()
	check_release_doc_exists()
	check_readiness_evidence()
	check_compat_counts_match_doc()
	check_qwen2_claim_has_evidence()
	check_no_stable_v030_claim()
	check_no_overclaim()
	check_external_validation_disclosed()
	check_citation_unchanged()

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
