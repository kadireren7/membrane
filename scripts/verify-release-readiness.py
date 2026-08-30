#!/usr/bin/env python3
"""Validate release metadata consistency for the current RC2 release-prep
work: the version string embedded in the product binary matches what the
release notes/README claim, no stable-v0.3.0 claim exists anywhere, the
stable v0.2.0 citation is untouched, and no known overclaim phrase appears
in the release-facing docs. Deliberately small -- a handful of targeted
checks, not a general documentation linter.

Exit code: 0 if every check passes, 1 otherwise.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_RC_VERSION = "0.3.0-rc2"
FAILURES = []
CHECK_COUNT = 0

# Section 20 of the Phase 22 task: reject wording equivalent to these,
# case-insensitively, across the release-facing docs below.
OVERCLAIM_PATTERNS = [
	r"supports all llama\.cpp models",
	r"supports all vulkan gpus",
	r"automatic optimal configuration",
	r"guaranteed fit",
	r"zero oom",
	r"production[- ]ready",
]

RELEASE_FACING_DOCS = [
	"README.md",
	"docs/release-v0.3.0-rc2.md",
	"docs/auto-fallback.md",
	"docs/joint-planner.md",
	"docs/compatibility.md",
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


@check("MEMBRANE_VERSION matches the expected RC2 version string")
def check_version_string():
	path = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
	text = path.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	if not m:
		fail("version string", f"MEMBRANE_VERSION not found in {path}")
		return
	found = m.group(1)
	if found != EXPECTED_RC_VERSION:
		fail("version string", f"MEMBRANE_VERSION is {found!r}, expected "
			f"{EXPECTED_RC_VERSION!r}")


@check("release-notes document exists for the expected RC2 version")
def check_release_notes_exist():
	path = REPO_ROOT / "docs" / f"release-v{EXPECTED_RC_VERSION}.md"
	if not path.exists():
		fail("release notes", f"{path} does not exist")


@check("README's release-status line references the current RC2 version")
def check_readme_release_status():
	text = (REPO_ROOT / "README.md").read_text()
	if EXPECTED_RC_VERSION not in text:
		fail("README release status", f"README.md never mentions "
			f"{EXPECTED_RC_VERSION!r}")


@check("no stable v0.3.0 claim exists anywhere in tracked release-facing docs")
def check_no_stable_v030_claim():
	# A bare "v0.3.0" (no -rcN suffix) as a STABLE claim -- explicitly
	# excludes "v0.3.0-rc1"/"v0.3.0-rc2" (both legitimate, real tags).
	pattern = re.compile(r"v0\.3\.0(?!-rc)")
	for rel in RELEASE_FACING_DOCS + ["CITATION.cff"]:
		path = REPO_ROOT / rel
		if not path.exists():
			continue
		text = path.read_text()
		if pattern.search(text):
			fail("no-stable-v0.3.0", f"{rel} appears to claim a stable "
				f"v0.3.0 (bare, no -rcN suffix)")


@check("CITATION.cff still targets stable v0.2.0, not bumped to an RC")
def check_citation_still_v020():
	path = REPO_ROOT / "CITATION.cff"
	if not path.exists():
		fail("CITATION.cff", "file does not exist")
		return
	text = path.read_text()
	m = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
	if not m:
		fail("CITATION.cff", "no top-level version: field found")
		return
	if m.group(1) != "0.2.0":
		fail("CITATION.cff", f"version is {m.group(1)!r}, expected the "
			f"stable '0.2.0' -- an RC must never become the citation "
			f"target without a deliberate, reviewed decision")


@check("no known overclaim phrase in release-facing docs")
def check_no_overclaim_phrases():
	compiled = [re.compile(p, re.IGNORECASE) for p in OVERCLAIM_PATTERNS]
	for rel in RELEASE_FACING_DOCS:
		path = REPO_ROOT / rel
		if not path.exists():
			continue
		text = path.read_text()
		for pat in compiled:
			for m in pat.finditer(text):
				# A negated mention ("does NOT claim...", "never...",
				# "not \"all Vulkan GPUs\"") within ~40 chars before the
				# match is a disclosure, not an overclaim -- this
				# project's own docs deliberately name these phrases to
				# disclaim them (see docs/compatibility.md,
				# docs/github-presentation.md).
				window = text[max(0, m.start() - 40):m.start()].lower()
				if not re.search(r"\bnot\b|\bnever\b|\bno[t]?\b|"
						r"does not|n['’]t\b", window):
					fail("overclaim scan", f"{rel} contains "
						f"{m.group(0)!r} without an apparent negation "
						f"nearby")


@check("compatibility counts match the documented RC2 scope (16/9/1) "
	"AT THE v0.3.0-rc2 TAG -- not the live, ongoing docs/compatibility.json")
def check_compat_counts():
	import json
	import shutil
	import subprocess

	# Phase 26 fix: this is a claim about what v0.3.0-rc2 itself
	# documented at release time -- docs/compatibility.json is a living
	# file main keeps growing/reclassifying (e.g. Phase 26 moved
	# MC-17/MC-18/MC-19 from UNSUPPORTED to SUPPORTED), so pinning this
	# check to the CURRENT working-tree file would make it fail every
	# time a later phase legitimately changes compatibility, which is
	# not what "RC2's documented scope" means. Reading the tag's own
	# historical snapshot via git keeps this check meaningful forever,
	# regardless of how far main moves ahead of RC2 (Section 31/32 of
	# the Phase 26 task: "Current main may now be ahead of RC2").
	#
	# Prerequisite: the checkout running this script must have the
	# v0.3.0-rc2 tag available locally (a normal `git clone` has it; a
	# shallow/tagless CI checkout may not -- `git fetch --tags` or
	# `actions/checkout`'s `fetch-tags: true` fixes that). This script
	# is not currently run from CI (no CI job invokes it), only locally
	# via scripts/verify-results.py/prepare-release.sh -- CodeRabbit
	# review (PR #36): documented here rather than silently assumed, and
	# the failure message below names the fix instead of surfacing a
	# raw git error.
	git_bin = shutil.which("git")
	if git_bin is None:
		fail("compatibility counts", "git is not on PATH -- cannot read "
			"docs/compatibility.json at the v0.3.0-rc2 tag")
		return
	tag_check = subprocess.run(
		[git_bin, "rev-parse", "--verify", "-q", "v0.3.0-rc2^{commit}"],
		cwd=REPO_ROOT, capture_output=True, text=True,
	)
	if tag_check.returncode != 0:
		fail("compatibility counts", "the v0.3.0-rc2 tag is not available "
			"in this checkout -- run `git fetch --tags` (or, in a CI "
			"checkout, set `actions/checkout`'s `fetch-tags: true`) "
			"before running this script")
		return
	try:
		raw = subprocess.run(
			[git_bin, "show", "v0.3.0-rc2:docs/compatibility.json"],
			cwd=REPO_ROOT, capture_output=True, text=True, check=True,
		).stdout
	except subprocess.CalledProcessError as e:
		fail("compatibility counts", f"could not read docs/compatibility.json "
			f"at the v0.3.0-rc2 tag via git show: {e.stderr.strip()}")
		return
	data = json.loads(raw)
	rows = data.get("rows", [])
	from collections import Counter
	counts = Counter(r.get("status") for r in rows)
	expected = {"SUPPORTED": 16, "UNSUPPORTED": 9, "NOT_YET_VALIDATED": 1}
	unexpected = [status for status in counts if status not in expected]
	if unexpected:
		fail("compatibility counts", f"unexpected statuses: {unexpected!r}")
	if len(rows) != sum(expected.values()):
		fail("compatibility counts", f"row count is {len(rows)}, expected "
			f"{sum(expected.values())}")
	for status, n in expected.items():
		if counts.get(status, 0) != n:
			fail("compatibility counts", f"{status} count is "
				f"{counts.get(status, 0)}, expected {n}")


READINESS_EVIDENCE_PATH = REPO_ROOT / "results" / "release-v0.3.0-rc2" / "readiness.json"


@check("readiness.json is valid JSON with the expected top-level shape")
def check_readiness_evidence_shape():
	if not READINESS_EVIDENCE_PATH.exists():
		fail("readiness evidence", f"{READINESS_EVIDENCE_PATH} does not exist")
		return
	import json
	data = json.loads(READINESS_EVIDENCE_PATH.read_text())
	for field in ("schema_version", "label", "membrane_commit",
			"membrane_version", "test_results"):
		if field not in data:
			fail("readiness evidence", f"missing required field {field!r}")
	if data.get("label") not in ("REAL", "SIMULATED"):
		fail("readiness evidence", f"label {data.get('label')!r} not in "
			f"('REAL', 'SIMULATED')")
	if not re.fullmatch(r"[0-9a-f]{40}", data.get("membrane_commit", "")):
		fail("readiness evidence", f"membrane_commit "
			f"{data.get('membrane_commit')!r} is not a 40-hex SHA")


@check("readiness evidence's membrane_version matches the product's own MEMBRANE_VERSION")
def check_readiness_evidence_version_matches():
	if not READINESS_EVIDENCE_PATH.exists():
		return
	import json
	data = json.loads(READINESS_EVIDENCE_PATH.read_text())
	path = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', path.read_text())
	product_version = m.group(1) if m else None
	if data.get("membrane_version") != product_version:
		fail("readiness evidence", f"evidence membrane_version "
			f"{data.get('membrane_version')!r} does not match the product's "
			f"own MEMBRANE_VERSION {product_version!r}")


@check("readiness evidence never reports a partial pass as release-ready")
def check_readiness_evidence_no_partial_pass():
	if not READINESS_EVIDENCE_PATH.exists():
		return
	import json
	data = json.loads(READINESS_EVIDENCE_PATH.read_text())
	for name, result in (data.get("test_results") or {}).items():
		if not isinstance(result, dict) or "passed" not in result:
			continue
		if result["passed"] != result.get("total"):
			fail("readiness evidence", f"test_results.{name} reports "
				f"{result['passed']}/{result.get('total')} -- a partial "
				f"pass must never be committed as release-readiness evidence")


def main():
	check_version_string()
	check_release_notes_exist()
	check_readme_release_status()
	check_no_stable_v030_claim()
	check_citation_still_v020()
	check_no_overclaim_phrases()
	check_compat_counts()
	check_readiness_evidence_shape()
	check_readiness_evidence_version_matches()
	check_readiness_evidence_no_partial_pass()

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
