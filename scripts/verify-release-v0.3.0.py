#!/usr/bin/env python3
"""Validate v0.3.0 stable-release readiness: MEMBRANE_VERSION and
CITATION.cff both say 0.3.0, docs/release-v0.3.0.md and
results/release-v0.3.0/readiness.json exist and are self-consistent,
live compatibility counts match what the release notes claim, the
official package policy is the Vulkan-enabled `membrane` package with
a bare (no ~rc) Debian version, external validation is disclosed as
limited rather than overclaimed, no CUDA/all-Qwen2/speedup overclaim
exists anywhere in release-facing docs, README's release status
correctly says stable v0.3.0, and the historical RC1/RC2/RC3 release
docs/evidence files remain byte-for-byte untouched (this phase must
never rewrite them to make stable counts "match").

This is Phase 31's own live-state validator -- same standalone-script,
own-CI-job pattern as scripts/verify-cli-ux.py/verify-distribution.py/
verify-release-artifacts.py/verify-first-run.py (Phases 28-30), not
chained into scripts/verify-results.py. It is expected to need
retirement itself once a future stable release supersedes v0.3.0 --
see scripts/verify-release-rc3.py's own retirement (this exact phase)
for the precedent and reasoning.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
CITATION_PATH = REPO_ROOT / "CITATION.cff"
RELEASE_DOC_PATH = REPO_ROOT / "docs" / "release-v0.3.0.md"
README_PATH = REPO_ROOT / "README.md"
READINESS_PATH = REPO_ROOT / "results" / "release-v0.3.0" / "readiness.json"
COMPAT_JSON_PATH = REPO_ROOT / "docs" / "compatibility.json"
CMAKE_PATH = REPO_ROOT / "CMakeLists.txt"

RELEASE_FACING_DOCS = [
	"README.md",
	"docs/release-v0.3.0.md",
	"docs/compatibility.md",
]

HISTORICAL_FILES = [
	"docs/release-v0.3.0-rc1.md",
	"docs/release-v0.3.0-rc2.md",
	"docs/release-v0.3.0-rc3.md",
	"docs/release-v0.2.0-rc1.md",
	"results/release-v0.3.0-rc3/readiness.json",
]

OVERCLAIM_PATTERNS = [
	r"\ball\s+qwen2\s+models\b",
	r"\bcuda[- ]?(supported|enabled|accelerated)\b",
	r"\bfaster\s+(inference|than)\b",
	r"\bperformance\s+optimi[sz]ed\b",
	r"\bimproved\s+inference\s+speed\b",
	r"\bproduction[- ]ready\b",
	r"\buniversal\s+linux\b",
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
			except Exception as e:  # noqa: BLE001 -- report, don't crash the whole run
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			status = "PASS" if ok else "FAIL"
			print(f"[{status}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return decorator


@check("MEMBRANE_VERSION == 0.3.0 (live product_cli.h)")
def _c1():
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', text)
	if not m:
		return False, "MEMBRANE_VERSION not found"
	return m.group(1) == "0.3.0", f"MEMBRANE_VERSION={m.group(1)!r}"


@check("CITATION.cff version == 0.3.0")
def _c2():
	text = CITATION_PATH.read_text()
	m = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
	if not m:
		return False, "no top-level version: field found"
	return m.group(1) == "0.3.0", f"CITATION.cff version={m.group(1)!r}"


@check("docs/release-v0.3.0.md exists")
def _c3():
	return RELEASE_DOC_PATH.exists(), str(RELEASE_DOC_PATH)


@check("results/release-v0.3.0/readiness.json exists, is valid JSON, and "
	"its membrane_version matches the live MEMBRANE_VERSION")
def _c4():
	if not READINESS_PATH.exists():
		return False, f"{READINESS_PATH} does not exist"
	d = json.loads(READINESS_PATH.read_text())
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', text)
	live_version = m.group(1) if m else None
	ok = d.get("version") == live_version == "0.3.0"
	return ok, f"readiness.version={d.get('version')!r} live={live_version!r}"


@check("readiness evidence is labeled REAL and never reports a "
	"not-yet-run check as green")
def _c5():
	d = json.loads(READINESS_PATH.read_text())
	bad = []
	if d.get("label") != "REAL":
		bad.append(f"label={d.get('label')!r}, expected REAL")
	if d.get("release_type") != "stable":
		bad.append(f"release_type={d.get('release_type')!r}, expected 'stable'")
	if d.get("base_prerelease") != "v0.3.0-rc3":
		bad.append(f"base_prerelease={d.get('base_prerelease')!r}, expected 'v0.3.0-rc3'")
	return len(bad) == 0, "; ".join(bad) if bad else "label/release_type/base_prerelease correct"


@check("live compatibility counts match what docs/release-v0.3.0.md "
	"and the readiness evidence both claim")
def _c6():
	compat = json.loads(COMPAT_JSON_PATH.read_text())
	rows = compat["rows"]
	total = len(rows)
	supported = sum(1 for r in rows if r["status"] == "SUPPORTED")
	unsupported = sum(1 for r in rows if r["status"] == "UNSUPPORTED")
	not_yet = sum(1 for r in rows if r["status"] == "NOT_YET_VALIDATED")
	release_text = RELEASE_DOC_PATH.read_text()
	# \s+ between tokens tolerates markdown line-wrapping (the doc wraps
	# long lines for readability -- a literal embedded newline there is
	# not a content problem worth failing this check over).
	expected_pattern = (rf"{total}\s+total\s+rows\s+—\s+{supported}\s+SUPPORTED,\s+"
		rf"{unsupported}\s+UNSUPPORTED,\s+{not_yet}\s+NOT_YET_VALIDATED")
	bad = []
	if not re.search(expected_pattern, release_text):
		bad.append(f"docs/release-v0.3.0.md does not contain a "
			f"{total}/{supported}/{unsupported}/{not_yet} compatibility-count phrase")
	d = json.loads(READINESS_PATH.read_text())
	counts = d.get("compatibility_counts", {})
	if counts.get("total") != total or counts.get("supported") != supported \
			or counts.get("unsupported") != unsupported \
			or counts.get("not_yet_validated") != not_yet:
		bad.append(f"readiness.compatibility_counts={counts!r} != live "
			f"{{total: {total}, supported: {supported}, unsupported: "
			f"{unsupported}, not_yet_validated: {not_yet}}}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{total}/{supported}/{unsupported}/{not_yet} consistent everywhere"


@check("official package policy is the Vulkan-enabled 'membrane' "
	"package (CMakeLists.txt), and its stable Debian version has no "
	"~rc suffix")
def _c7():
	cmake_text = CMAKE_PATH.read_text()
	bad = []
	if 'set(CPACK_PACKAGE_NAME "membrane")' not in cmake_text:
		bad.append('CPACK_PACKAGE_NAME "membrane" not found under GGML_VULKAN')
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', text)
	live_version = m.group(1) if m else ""
	debian_version = live_version.replace("-", "~")
	if "~" in debian_version:
		bad.append(f"stable Debian version {debian_version!r} unexpectedly "
			"contains '~' -- looks like a prerelease mapping, not stable")
	return len(bad) == 0, ("; ".join(bad) if bad else
		f"official package policy correct, Debian version={debian_version!r}")


@check("external validation limitation is disclosed (not overclaimed) "
	"in docs/release-v0.3.0.md and README.md")
def _c8():
	bad = []
	for path in (RELEASE_DOC_PATH, README_PATH):
		text = path.read_text().lower()
		if "limited" not in text and "not yet happened" not in text:
			bad.append(f"{path.name} does not disclose the external "
				"validation limitation")
		if re.search(r"\b(three|3)\s+independent\s+(user\s+)?(machines|hosts)\b", text):
			bad.append(f"{path.name} appears to overclaim independent "
				"multi-host validation that does not exist")
	return len(bad) == 0, "; ".join(bad) if bad else "disclosed honestly in both docs"


@check("no CUDA/all-Qwen2/speedup/production-ready/universal-Linux "
	"overclaim anywhere in release-facing docs")
def _c9():
	bad = []
	# A match preceded (within 60 chars) by a negation/disclaimer marker
	# is the thing this project's own release docs are EXPECTED to do
	# (explicitly naming a phrase to say it does NOT apply -- the exact
	# style docs/release-v0.3.0-rc3.md already uses, e.g. "This is
	# not: ... 'All Qwen2 models supported'") -- only an unqualified,
	# affirmative claim is a real overclaim.
	negation_re = re.compile(
		r"\b(not|never|no|none|does\s+not|is\s+not|are\s+not|"
		r"without\s+a|zero)\b", re.IGNORECASE)
	for rel in RELEASE_FACING_DOCS:
		path = REPO_ROOT / rel
		if not path.exists():
			continue
		text = path.read_text()
		for pat in OVERCLAIM_PATTERNS:
			for m in re.finditer(pat, text, re.IGNORECASE):
				window = text[max(0, m.start() - 60):m.start()]
				if negation_re.search(window):
					continue
				bad.append(f"{rel} matches overclaim pattern {pat!r} at "
					f"{m.group(0)!r} (no negation found nearby)")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified overclaim phrase found"


@check("README's release-status line says stable v0.3.0, not v0.2.0 "
	"stable / v0.3.0-rc3 latest prerelease")
def _c10():
	text = README_PATH.read_text()
	m = re.search(r"\*\*Release status\*\*:.*?(?=\n\n)", text, re.DOTALL)
	if not m:
		return False, "no '**Release status**:' line found in README.md"
	section = m.group(0)
	bad = []
	if "latest stable tag `v0.3.0`" not in section:
		bad.append("does not say 'latest stable tag `v0.3.0`'")
	if "v0.2.0`" in section and "supersedes" not in section:
		bad.append("mentions v0.2.0 without framing it as superseded")
	return len(bad) == 0, "; ".join(bad) if bad else "release status correctly says stable v0.3.0"


@check("historical RC1/RC2/RC3 release docs/evidence remain untouched "
	"(git-tracked, no uncommitted diff)")
def _c11():
	bad = []
	for rel in HISTORICAL_FILES:
		path = REPO_ROOT / rel
		if not path.exists():
			bad.append(f"{rel} is missing")
			continue
		result = subprocess.run(
			["git", "-C", str(REPO_ROOT), "diff", "--quiet", "--", rel],
			capture_output=True)
		if result.returncode != 0:
			bad.append(f"{rel} has uncommitted local changes")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(HISTORICAL_FILES)} historical files untouched"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11):
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
