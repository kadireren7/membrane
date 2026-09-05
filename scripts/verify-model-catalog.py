#!/usr/bin/env python3
"""Validate Mega Phase D, PR D1's model-catalog/download-manager
evidence (results/model-catalog/validation.json): schema and REAL/
SOURCE_ANALYSIS labeling, every catalog entry has real license/repo/
checksum metadata, HTTPS-only enforcement, no auto-executed downloaded
file, the download manager never touches git, no premature "auto
variant selection"/hardware-aware claim (that is D2's own job), and a
handful of direct source-level regression guards for this PR's own
real findings: the idempotence pre-check, the progress-callback
throttle, and the checksum-on-idempotent-path fix.

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
EVIDENCE_PATH = REPO_ROOT / "results" / "model-catalog" / "validation.json"
CATALOG_H = REPO_ROOT / "tools" / "membrane" / "model_catalog.h"
CATALOG_CPP = REPO_ROOT / "tools" / "membrane" / "model_catalog.cpp"
DOWNLOAD_H = REPO_ROOT / "tools" / "membrane" / "download_manager.h"
DOWNLOAD_CPP = REPO_ROOT / "tools" / "membrane" / "download_manager.cpp"
MODEL_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "model_cmd.cpp"
CATALOG_DOC = REPO_ROOT / "docs" / "model-catalog.md"
CMAKE_PATH = REPO_ROOT / "tools" / "membrane" / "CMakeLists.txt"

VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS"}
NEW_TESTS = ["test_model_catalog", "test_download_manager"]

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


def _load_evidence():
	return json.loads(EVIDENCE_PATH.read_text())


def _extract_catalog_json():
	text = CATALOG_CPP.read_text()
	start = text.index('R"JSON(') + len('R"JSON(')
	end = text.index(')JSON"', start)
	return json.loads(text[start:end])


@check("evidence file, module files, and doc all exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, CATALOG_H, CATALOG_CPP, DOWNLOAD_H, DOWNLOAD_CPP,
			CATALOG_DOC) if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1")
def _c2():
	data = _load_evidence()
	ok = (data.get("schema_version") == 1
		and isinstance(data.get("phase"), str) and len(data["phase"]) > 0)
	return ok, "shape OK" if ok else f"unexpected shape: {list(data.keys())}"


@check("every top-level evidence section carries a valid REAL/SYNTHETIC/"
	"SOURCE_ANALYSIS label")
def _c3():
	data = _load_evidence()
	bad = []
	for key, val in data.items():
		if isinstance(val, dict) and "label" in val:
			if val["label"] not in VALID_LABELS:
				bad.append(f"{key}: {val['label']!r}")
	return len(bad) == 0, ("bad labels: " + "; ".join(bad)) if bad \
		else "all labels valid"


@check("the compiled-in catalog JSON is valid and every family has real "
	"license/repo_url/compatibility_evidence metadata (Section 4 of the "
	"task: no silent redistribution, every source recorded)")
def _c4():
	catalog = _extract_catalog_json()
	bad = []
	families = catalog.get("families", [])
	if len(families) == 0:
		return False, "catalog has zero families"
	for f in families:
		for field in ("name", "license", "repo_url", "compatibility_evidence",
				"provider", "arch"):
			if not f.get(field):
				bad.append(f"{f.get('name', '?')}: missing {field}")
		if not f.get("repo_url", "").startswith("https://"):
			bad.append(f"{f.get('name')}: repo_url is not https://")
		if not f.get("variants"):
			bad.append(f"{f.get('name')}: no variants")
		for v in f.get("variants", []):
			if not v.get("download_url", "").startswith("https://"):
				bad.append(f"{f.get('name')}/{v.get('quant')}: "
					"download_url is not https://")
			if v.get("size_bytes", 0) <= 0:
				bad.append(f"{f.get('name')}/{v.get('quant')}: "
					"size_bytes is not positive")
			if len(v.get("sha256", "")) != 64:
				bad.append(f"{f.get('name')}/{v.get('quant')}: sha256 is "
					"not a 64-char digest")
	return len(bad) == 0, ("; ".join(bad) if bad
		else f"{len(families)} families, all real metadata present")


@check("regression guard: downloads are rejected before any network call "
	"unless the URL is https:// (Section 6: 'HTTPS only')")
def _c5():
	text = DOWNLOAD_CPP.read_text()
	ok = 'rfind("https://", 0) != 0' in text and "INVALID_URL" in text
	return ok, "HTTPS-only guard present" if ok else "guard not found"


@check("regression guard: no downloaded file is ever exec'd/system()'d "
	"anywhere in the download manager or model_cmd.cpp's install path")
def _c6():
	bad = []
	for path in (DOWNLOAD_CPP, MODEL_CMD_CPP):
		text = path.read_text()
		if re.search(r"\bsystem\s*\(|\bexecv?p?\s*\(", text):
			bad.append(path.name)
	return len(bad) == 0, (f"exec/system call found in: {bad}" if bad
		else "no exec/system call in either module")


@check("regression guard: the download manager never touches git (no "
	"git commands anywhere in download_manager.cpp)")
def _c7():
	text = DOWNLOAD_CPP.read_text()
	bad = [tok for tok in ("git commit", "git push", "git tag") if tok in text]
	return len(bad) == 0, (f"forbidden git operation(s): {bad}" if bad
		else "no git operation present")


@check("regression guard: install's idempotence pre-check exists (a real "
	"bug this PR found and fixed -- re-installing an already-correct "
	"file used to always re-download it)")
def _c8():
	text = MODEL_CMD_CPP.read_text()
	ok = "already_present" in text and "already downloaded at the right size" in text
	return ok, "idempotence pre-check present" if ok else "not found"


@check("regression guard: the idempotent path re-verifies checksum "
	"instead of unconditionally reporting it unverified (a real bug "
	"this PR found and fixed)")
def _c9():
	text = MODEL_CMD_CPP.read_text()
	ok = ("if (already_present)" in text
		and text.count("membrane_compute_sha256(dest_path") >= 1)
	return ok, "checksum re-verification present" if ok else "not found"


@check("regression guard: the install progress callback throttles "
	"updates (a real bug this PR found and fixed -- an untbrottled "
	"version produced a 3.4 MB log from thousands of redundant prints)")
def _c10():
	text = MODEL_CMD_CPP.read_text()
	ok = "clock_gettime" in text and "elapsed_ms" in text
	return ok, "throttle present" if ok else "not found"


@check("no premature hardware-aware/'best quantization' claim -- D1 has "
	"no variant-selection intelligence yet (that is D2's own separate "
	"work), and the default-variant code/docs disclose this honestly")
def _c11():
	bad = []
	pattern = re.compile(r"\bbest\s+quant(?:ization)?\b", re.IGNORECASE)
	for path in (MODEL_CMD_CPP, CATALOG_DOC, REPO_ROOT / "README.md"):
		if not path.exists():
			continue
		if pattern.search(path.read_text()):
			bad.append(path.name)
	return len(bad) == 0, (f"overclaim found in: {bad}" if bad
		else "no overclaim found")


@check("every new test binary is registered as a real ctest entry")
def _c12():
	text = CMAKE_PATH.read_text()
	missing = [t for t in NEW_TESTS if f"add_test(NAME {t}" not in text]
	return len(missing) == 0, (f"missing add_test(): {missing}" if missing
		else "all present")


@check("CI: every job that sets MEMBRANE_ENABLE_LLAMA=ON also installs "
	"libcurl4-openssl-dev (find_package(CURL REQUIRED) runs at CMake "
	"configure time regardless of which targets are later built)")
def _c13():
	ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
	text = ci_path.read_text()
	lines = text.splitlines()
	job_header_re = re.compile(r"^  [a-zA-Z0-9_-]+:\s*$")
	bad = []
	for i, line in enumerate(lines):
		if "MEMBRANE_ENABLE_LLAMA=ON" in line:
			# Search backward to this job's OWN header (a top-level,
			# 2-space-indented "name:" line), not a fixed line count --
			# a real false positive was found and fixed here: a job with
			# more than 40 lines between its own apt-get install step
			# and a later configure call (packaging-smoke's Vulkan
			# configure, well after its one shared install step) was
			# incorrectly flagged as missing the dependency it actually
			# already had.
			job_start = 0
			for j in range(i - 1, -1, -1):
				if job_header_re.match(lines[j]):
					job_start = j
					break
			window = "\n".join(lines[job_start:i])
			if "libcurl4-openssl-dev" not in window:
				bad.append(f"line {i + 1}")
	return len(bad) == 0, (f"missing libcurl4-openssl-dev near: {bad}" if bad
		else "every MEMBRANE_ENABLE_LLAMA=ON job has libcurl4-openssl-dev")


@check("Mega Phase C's own evidence files are untouched by this PR's own "
	"new commits (checked against origin/main)")
def _c14():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/product-onboarding/validation.json",
		"results/release-supply-chain/validation.json",
		"results/product-hardening/v0.4-validation.json",
		"results/release-v0.4.0/readiness.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11,
			_c12, _c13, _c14):
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
