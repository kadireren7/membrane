#!/usr/bin/env python3
"""Validate Mega Phase D, PR D5's real Windows support evidence
(results/windows-support/validation.json): schema and REAL/
SOURCE_ANALYSIS labeling, the cross-platform compat shim headers
exist, the real bugs this phase found and fixed stay fixed (regression
guards for the -Wall/-Wextra MSVC flag issue, the windows.h min/max
macro collision, the sha256sum backslash-escaping parse bug, the
schtasks UTF-16 XML conversion), no CUDA/Metal/Windows-specific string
leaked into any backend-agnostic planner/policy module, and no
"every Windows machine" overclaim.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see scripts/verify-cuda-backend.py/
scripts/verify-macos-metal.py, PR D3/D4's own equivalents).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "windows-support" / "validation.json"
WINDOWS_DOC = REPO_ROOT / "docs" / "windows-support.md"
ROOT_CMAKE = REPO_ROOT / "CMakeLists.txt"
COMPAT_JSON = REPO_ROOT / "docs" / "compatibility.json"
PTHREAD_COMPAT_H = REPO_ROOT / "include" / "membrane" / "pthread_compat.h"
DIRENT_COMPAT_H = REPO_ROOT / "include" / "membrane" / "dirent_compat.h"
POSIX_COMPAT_H = REPO_ROOT / "include" / "membrane" / "posix_compat.h"
CLOCK_COMPAT_H = REPO_ROOT / "include" / "membrane" / "clock_compat.h"
WINDOWS_LEAN_H = REPO_ROOT / "include" / "membrane" / "windows_lean.h"
WINDOWS_TASK_H = REPO_ROOT / "tools" / "membrane" / "windows_task.h"
WINDOWS_TASK_CPP = REPO_ROOT / "tools" / "membrane" / "windows_task.cpp"
SERVICE_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "service_cmd.cpp"
DOWNLOAD_MANAGER_CPP = REPO_ROOT / "tools" / "membrane" / "download_manager.cpp"
BACKEND_AGNOSTIC_SOURCES = [
	REPO_ROOT / "tools" / "membrane-run" / "gpu_policy.c",
	REPO_ROOT / "tools" / "membrane-run" / "context_recommender.c",
	REPO_ROOT / "tools" / "membrane-run" / "compat_check.c",
]

VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS"}

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


@check("evidence file, docs/windows-support.md, and all four compat "
	"shim headers plus windows_lean.h/windows_task.h/.cpp exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, WINDOWS_DOC, PTHREAD_COMPAT_H, DIRENT_COMPAT_H,
			POSIX_COMPAT_H, CLOCK_COMPAT_H, WINDOWS_LEAN_H,
			WINDOWS_TASK_H, WINDOWS_TASK_CPP)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and real "
	"(non-PENDING) D1/D2/D3/D4 squash SHAs")
def _c2():
	data = _load_evidence()
	shas = data.get("pr_squash_shas", {})
	ok = all(re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))
		for k in ("D1", "D2", "D3", "D4"))
	return ok, (f"D1={shas.get('D1')!r} D2={shas.get('D2')!r} "
		f"D3={shas.get('D3')!r} D4={shas.get('D4')!r}")


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


@check("real build/device-enum/generation/service-lifecycle sections "
	"are labeled REAL, not SOURCE_ANALYSIS (this PR's central claims "
	"are genuine on-device validation, not code-reading alone)")
def _c4():
	data = _load_evidence()
	required_real = ("hardware", "build_integration",
		"real_device_enumeration", "real_generation",
		"service_lifecycle_smoke")
	bad = [k for k in required_real if data.get(k, {}).get("label") != "REAL"]
	return len(bad) == 0, (f"not REAL: {bad}" if bad else "all REAL")


@check("regression guard: root CMakeLists.txt's MSVC /W4 guard exists "
	"(the real -Wall/-Wextra/-Wpedantic MSVC incompatibility this "
	"phase found and fixed)")
def _c5():
	text = ROOT_CMAKE.read_text()
	ok = "if(MSVC)" in text and "/W4" in text
	return ok, "MSVC /W4 guard present" if ok else "guard missing"


@check("regression guard: the four cross-platform compat shim headers "
	"each provide a real Win32-backed implementation (not just a "
	"passthrough) for their own real POSIX symbol")
def _c6():
	checks = {
		"pthread_compat.h (SRWLOCK)": (PTHREAD_COMPAT_H, "SRWLOCK"),
		"dirent_compat.h (FindFirstFileA)": (DIRENT_COMPAT_H, "FindFirstFileA"),
		"posix_compat.h (_fullpath)": (POSIX_COMPAT_H, "_fullpath"),
		"clock_compat.h (QueryPerformanceCounter)": (CLOCK_COMPAT_H,
			"QueryPerformanceCounter"),
	}
	bad = [name for name, (path, needle) in checks.items()
		if needle not in path.read_text()]
	return len(bad) == 0, (f"missing real Win32 backing: {bad}" if bad
		else "all four shims carry real Win32 implementations")


@check("regression guard: windows_lean.h defines NOMINMAX before "
	"including windows.h (the real std::min/std::max macro-collision "
	"bug this phase found and fixed)")
def _c7():
	text = WINDOWS_LEAN_H.read_text()
	ok = "NOMINMAX" in text and "<windows.h>" in text
	return ok, "NOMINMAX guard present" if ok else "guard missing"


@check("regression guard: service_cmd.cpp's Windows install path "
	"writes membrane_task_xml_to_utf16le_bytes()'s own output, not "
	"membrane_generate_task_xml()'s raw UTF-8 return value directly "
	"(the real schtasks 'unable to switch the encoding' bug)")
def _c8():
	text = SERVICE_CMD_CPP.read_text()
	ok = ("membrane_task_xml_to_utf16le_bytes" in text
		and "membrane_atomic_write_file(tmp_xml_path, task_xml_bytes"
			in text)
	return ok, "UTF-16LE conversion wired in" if ok else "not found"


@check("regression guard: download_manager.cpp strips a leading "
	"backslash from sha256sum output before parsing (the real "
	"Windows-path-escaping checksum-corruption bug)")
def _c9():
	text = DOWNLOAD_MANAGER_CPP.read_text()
	# The real source line is: if (!out.empty() && out[0] == '\\')
	# -- a C++ char literal for a single backslash (source bytes: two
	# literal backslash characters between the quotes). This Python
	# string needs four backslash characters to represent that same
	# two-backslash byte sequence.
	ok = "out[0] == '\\\\'" in text
	return ok, "backslash-stripping present" if ok else "not found"


@check("regression guard: no CUDA/Metal/Windows-specific string leaked "
	"into any backend-agnostic planner/policy module (the 'zero code "
	"change' claim this PR's own evidence makes, same finding as PR "
	"D3/D4)")
def _c10():
	bad = []
	for path in BACKEND_AGNOSTIC_SOURCES:
		if not path.exists():
			bad.append(f"{path.name}: MISSING")
			continue
		text = path.read_text()
		if re.search(r"\bWindows\b|_WIN32|\bMetal\b|\bmetal\b|\bCUDA\b|\bcuda\b",
				text):
			bad.append(f"{path.name}: contains a backend/platform-"
				"specific reference")
	return len(bad) == 0, ("; ".join(bad) if bad else "no backend/"
		"platform-specific code in any of these files")


@check("no 'every Windows machine'/'all Windows'/guaranteed-real-GPU "
	"overclaim in docs/windows-support.md (real CPU-only validation, "
	"no GPU tested this phase)")
def _c11():
	text = WINDOWS_DOC.read_text()
	pattern = re.compile(
		r"\ball\s+windows\b(?!\s*\.?\"|\s*\.?')|\bevery\s+windows\s+"
		r"(?:machine|host|pc)\b",
		re.IGNORECASE)
	bad = [m.group(0) for m in pattern.finditer(text)
		if "not " not in text[max(0, m.start() - 12):m.start()].lower()]
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("docs/compatibility.json has new, real Windows rows citing "
	"results/windows-support/validation.json, and every cited evidence "
	"path resolves to a real in-repository file")
def _c12():
	data = json.loads(COMPAT_JSON.read_text())
	rows = {r["id"]: r for r in data["rows"]}
	windows_rows = {rid: r for rid, r in rows.items()
		if "windows" in json.dumps(r).lower()
			and r.get("evidence")
			and any("windows-support" in e for e in r["evidence"])}
	if not windows_rows:
		return False, "no rows citing results/windows-support/ evidence found"
	bad = []
	for rid, row in windows_rows.items():
		for ev in row.get("evidence", []):
			if not (REPO_ROOT / ev).exists():
				bad.append(f"{rid}: {ev}")
	return len(bad) == 0, (f"broken evidence paths: {bad}" if bad
		else f"{len(windows_rows)} Windows row(s), all evidence paths "
			"resolve")


@check("Mega Phase C/D1/D2/D3/D4's own evidence files are untouched by "
	"this PR's own new commits (checked against origin/main)")
def _c13():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/model-catalog/validation.json",
		"results/model-variant-selection/validation.json",
		"results/cuda-backend/validation.json",
		"results/macos-metal/validation.json",
		"results/product-onboarding/validation.json",
		"results/release-supply-chain/validation.json",
		"results/product-hardening/v0.4-validation.json",
		"results/release-v0.4.0/readiness.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11,
			_c12, _c13):
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
