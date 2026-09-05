#!/usr/bin/env python3
"""Validate Mega Phase C, PR C1's product-onboarding evidence
(results/product-onboarding/validation.json): schema and REAL/SYNTHETIC/
SOURCE_ANALYSIS labeling, README/docs mention the new `membrane setup`/
`membrane doctor` commands, no premature v1.0/CUDA/GUI claim, stable
release still says v0.3.0, the llama.cpp patch set is unchanged, and a
handful of direct source-level regression guards for this PR's own real
findings: the systemctl-availability check (never a shelled-out
`which`/`command -v`), the JSON-mode stdout-silencing fix, and the
verify-endpoint retry-poll fix.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see verify-background-service.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "product-onboarding" / "validation.json"
README_PATH = REPO_ROOT / "README.md"
MEMBRANE_DIR = REPO_ROOT / "tools" / "membrane"
DOCTOR_CMD_CPP_PATH = MEMBRANE_DIR / "doctor_cmd.cpp"
SETUP_CMD_CPP_PATH = MEMBRANE_DIR / "setup_cmd.cpp"
MEMBRANE_CMAKE_PATH = MEMBRANE_DIR / "CMakeLists.txt"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
PATCHES_DIR = REPO_ROOT / "patches"
EXPECTED_PATCHES = {
	"llama.cpp-membrane-kv-type-override.patch",
	"llama.cpp-membrane-kv-device-override.patch",
}

VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS"}
VALID_SEVERITIES = {"BLOCKING", "HIGH", "MEDIUM", "LOW"}
NEW_TESTS = ["test_doctor_cmd", "test_setup_cmd"]

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


@check("evidence file exists and README exists")
def _c1():
	missing = [str(p) for p in (EVIDENCE_PATH, README_PATH) if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and a phase name")
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


@check("every first-product-audit friction finding carries a valid "
	"BLOCKING/HIGH/MEDIUM/LOW severity (Section 2 of the task)")
def _c4():
	data = _load_evidence()
	findings = data.get("first_product_audit", {}).get("friction_found", [])
	bad = [f["finding"] for f in findings
		if f.get("severity") not in VALID_SEVERITIES]
	return (len(findings) > 0 and len(bad) == 0), (
		f"bad severities: {bad}" if bad
		else "no findings recorded" if len(findings) == 0
		else f"{len(findings)} findings, all validly classified")


@check("README documents `membrane setup` and `membrane doctor`")
def _c5():
	text = README_PATH.read_text()
	has_setup = "membrane setup" in text
	has_doctor = "membrane doctor" in text
	ok = has_setup and has_doctor
	return ok, f"has_setup={has_setup} has_doctor={has_doctor}"


@check("no premature v1.0/CUDA-support/GUI claim anywhere in README "
	"(Mega Phase C's own explicit scope boundaries) -- a disclaimer "
	"that CUDA/a GUI is NOT supported is fine and expected, only a "
	"positive claim of either is flagged")
def _c6():
	text = README_PATH.read_text()
	bad = []
	pattern = re.compile(
		r"\bv1\.0\.\d+\b|\bCUDA[- ]enabled\b|\bsupports? CUDA\b|"
		r"\bCUDA support\b(?!\s+is not)|\bhas a (?:graphical|native) "
		r"(?:user interface|GUI)\b", re.IGNORECASE)
	for m in pattern.finditer(text):
		bad.append(m.group(0))
	return len(bad) == 0, "; ".join(bad) if bad else "no scope-violating claim"


@check("stable release still says v0.3.0 (no version bump during Mega "
	"Phase C sub-PRs before C4)")
def _c7():
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.3.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("patch set is unchanged and matches the two real committed patch "
	"files")
def _c8():
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	return real == EXPECTED_PATCHES, f"real={real} expected={EXPECTED_PATCHES}"


@check("regression guard: systemctl availability is detected via a real "
	"subprocess exit-code check, never a shelled-out which/command -v "
	"(Section 43's own 'no shell injection' spirit)")
def _c9():
	text = DOCTOR_CMD_CPP_PATH.read_text()
	has_real_check = "exit_code == 127" in text
	has_shell_lookup = re.search(r'"command -v|"which ', text) is not None
	ok = has_real_check and not has_shell_lookup
	return ok, (f"has_real_check={has_real_check} "
		f"has_shell_lookup={has_shell_lookup}")


@check("regression guard: `membrane doctor`/`membrane setup` never "
	"perform generation (Section 9: 'Do NOT perform expensive "
	"generation')")
def _c10():
	bad = []
	for path in (DOCTOR_CMD_CPP_PATH, SETUP_CMD_CPP_PATH):
		text = path.read_text()
		if "membrane_session_generate" in text or "chat/completions" in text:
			bad.append(path.name)
	return len(bad) == 0, (f"generation-related calls found in: {bad}" if bad
		else "no generation call in either module")


@check("regression guard: setup's own JSON mode silences the reused "
	"model/service dispatch calls (a real bug this PR found and fixed) "
	"rather than mixing their human prose into the JSON stream")
def _c11():
	text = SETUP_CMD_CPP_PATH.read_text()
	ok = "stdout_silencer_t" in text and "dispatch_silently_if_json" in text
	return ok, ("stdout-silencing helper present" if ok
		else "stdout-silencing helper missing")


@check("regression guard: the endpoint-verification step uses a bounded "
	"retry poll (a real startup-timing race this PR found and fixed), "
	"never a single immediate check nor an unbounded wait")
def _c12():
	text = SETUP_CMD_CPP_PATH.read_text()
	ok = "verify_get_with_retry" in text and "attempt < 15" in text
	return ok, "bounded retry poll present" if ok else "not found"


@check("every new test binary is registered as a real ctest entry")
def _c13():
	text = MEMBRANE_CMAKE_PATH.read_text()
	missing = [t for t in NEW_TESTS if f"add_test(NAME {t}" not in text]
	return len(missing) == 0, (f"missing add_test(): {missing}" if missing
		else "all present")


@check("Mega Phase A/B's own evidence files are untouched by this PR's "
	"own new commits (checked against origin/main)")
def _c14():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/runtime-service/validation.json",
		"results/background-service/validation.json") if p in changed]
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
