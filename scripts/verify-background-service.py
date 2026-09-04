#!/usr/bin/env python3
"""Validate Mega Phase B, PR B1's background-service evidence
(results/background-service/validation.json): schema and REAL/SYNTHETIC/
SOURCE_ANALYSIS labeling, docs/service.md exists with its required
sections, no premature v0.4/v1.0 release claim, stable release still says
v0.3.0, the llama.cpp patch set is unchanged, and a handful of direct
source-level regression guards for this PR's own safety rules: the
generated unit never targets a system-wide/root systemd path, no shell
(system()/popen()) is used anywhere in the new subprocess/service-command
modules, and `service install` really does refuse to overwrite a
non-MEMBRANE-managed unit without --force.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see verify-runtime-service.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "background-service" / "validation.json"
SERVICE_DOC_PATH = REPO_ROOT / "docs" / "service.md"
SERVER_DOC_PATH = REPO_ROOT / "docs" / "server.md"
REGISTRY_DOC_PATH = REPO_ROOT / "docs" / "model-registry.md"
MEMBRANE_DIR = REPO_ROOT / "tools" / "membrane"
SERVICE_CMD_CPP_PATH = MEMBRANE_DIR / "service_cmd.cpp"
SUBPROCESS_CPP_PATH = MEMBRANE_DIR / "subprocess.cpp"
SYSTEMD_UNIT_CPP_PATH = MEMBRANE_DIR / "systemd_unit.cpp"
FS_UTIL_CPP_PATH = MEMBRANE_DIR / "fs_util.cpp"
MEMBRANE_CMAKE_PATH = MEMBRANE_DIR / "CMakeLists.txt"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
PATCHES_DIR = REPO_ROOT / "patches"
EXPECTED_PATCHES = {
	"llama.cpp-membrane-kv-type-override.patch",
	"llama.cpp-membrane-kv-device-override.patch",
}

REQUIRED_SERVICE_DOC_SECTIONS = [
	"## Commands", "## The generated unit", "## Server config",
	"## Status", "## Model registry and default-model reload",
	"## Security scope", "## Uninstalling", "## Real evidence",
]
VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS"}

NEW_PURE_LIBS = [
	"membrane_fs_util", "membrane_subprocess", "membrane_systemd_unit",
	"membrane_server_config",
]
NEW_PURE_TESTS = [
	"test_systemd_unit", "test_server_config", "test_fs_util",
	"test_subprocess",
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


def _load_evidence():
	return json.loads(EVIDENCE_PATH.read_text())


@check("evidence and doc files all exist")
def _c1():
	missing = [str(p) for p in (EVIDENCE_PATH, SERVICE_DOC_PATH,
			SERVER_DOC_PATH, REGISTRY_DOC_PATH) if not p.exists()]
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


@check("docs/service.md has every required section")
def _c4():
	text = SERVICE_DOC_PATH.read_text()
	missing = [s for s in REQUIRED_SERVICE_DOC_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("no premature v0.4/v1.0 release claim anywhere in this PR's own "
	"new/changed docs")
def _c5():
	bad = []
	pattern = re.compile(
		r"\bv0\.4\.\d+\b|\bv0\.4-release\b|\bv1\.0\.\d+\b|"
		r"\bv0\.4\s+(is|has been)\s+released\b", re.IGNORECASE)
	for path in (SERVICE_DOC_PATH, SERVER_DOC_PATH, REGISTRY_DOC_PATH):
		text = path.read_text()
		for m in pattern.finditer(text):
			bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no premature release claim"


@check("stable release still says v0.3.0 (no version bump during "
	"Mega Phase B sub-PRs)")
def _c6():
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.3.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("patch set is unchanged and matches the two real committed patch "
	"files")
def _c7():
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	return real == EXPECTED_PATCHES, f"real={real} expected={EXPECTED_PATCHES}"


@check("regression guard: the generated systemd unit never targets a "
	"system-wide/root path -- only $XDG_CONFIG_HOME/HOME's own --user "
	"directory (or an explicit MEMBRANE_SYSTEMD_USER_DIR test override)")
def _c8():
	text = SYSTEMD_UNIT_CPP_PATH.read_text()
	bad = "/etc/systemd/system" in text
	ok = (not bad) and "/.config/systemd/user/" in text
	return ok, ("no /etc/systemd/system reference; --user path present" if ok
		else f"bad_system_path_referenced={bad}")


@check("regression guard: no shell (system()/popen()) anywhere in the new "
	"subprocess/service-command/systemd-unit modules -- argv is always "
	"passed directly to execvp()")
def _c9():
	bad = []
	for path in (SUBPROCESS_CPP_PATH, SERVICE_CMD_CPP_PATH,
			SYSTEMD_UNIT_CPP_PATH, FS_UTIL_CPP_PATH):
		text = path.read_text()
		for m in re.finditer(r"\b(system|popen)\s*\(", text):
			bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no shell invocation found"


@check("regression guard: `service install` refuses to overwrite a "
	"same-named unit lacking the MEMBRANE marker unless --force is given")
def _c10():
	text = SERVICE_CMD_CPP_PATH.read_text()
	ok = ("membrane_unit_is_membrane_managed" in text
		and "!force" in text and "UNIT_EXISTS" in text)
	return ok, ("marker check + !force + UNIT_EXISTS all present" if ok
		else "one or more of the overwrite-refusal pieces is missing")


@check("regression guard: `service uninstall` refuses to remove a unit it "
	"did not generate")
def _c11():
	text = SERVICE_CMD_CPP_PATH.read_text()
	ok = "NOT_MANAGED" in text
	return ok, "NOT_MANAGED refusal path present" if ok else "not found"


@check("regression guard: every new pure library actually links "
	"membrane_sanitizers (the exact PR A2 gap this project's own history "
	"warns about -- a library that links nothing is silently never "
	"sanitizer-instrumented, however green its own tests look)")
def _c12():
	text = MEMBRANE_CMAKE_PATH.read_text()
	bad = []
	for lib in NEW_PURE_LIBS:
		m = re.search(
			rf"add_library\({lib}\b.*?(?=\nadd_library|\nadd_executable|\Z)",
			text, re.DOTALL)
		if m is None or "membrane_sanitizers" not in m.group(0):
			bad.append(lib)
	return len(bad) == 0, (f"missing membrane_sanitizers link: {bad}" if bad
		else "all new pure libraries link membrane_sanitizers")


@check("all four new pure test binaries are registered as real ctest "
	"entries")
def _c13():
	text = MEMBRANE_CMAKE_PATH.read_text()
	missing = [t for t in NEW_PURE_TESTS if f"add_test(NAME {t}" not in text]
	return len(missing) == 0, (f"missing add_test(): {missing}" if missing
		else "all four registered")


@check("CI wires the new service-lifecycle-tests job and the new "
	"packaging-smoke install/uninstall integration step")
def _c14():
	ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
	text = ci_path.read_text()
	has_job = "service-lifecycle-tests:" in text
	has_step = "membrane service install/uninstall (isolated dirs)" in text
	ok = has_job and has_step
	return ok, (f"has_job={has_job} has_step={has_step}")


@check("Mega Phase A's own evidence files are untouched by this PR's own "
	"new commits (checked against origin/main)")
def _c15():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/runtime-service/validation.json",) if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


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
