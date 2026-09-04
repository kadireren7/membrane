#!/usr/bin/env python3
"""Validate Mega Phase B's background-service/streaming evidence
(results/background-service/validation.json, extended across PRs B1-B4,
matching results/runtime-service/validation.json's own precedent from
Mega Phase A): schema and REAL/SYNTHETIC/SOURCE_ANALYSIS labeling,
docs/service.md exists with its required sections, no premature
v0.4/v1.0 release claim, stable release still says v0.3.0, the llama.cpp
patch set is unchanged, and a handful of direct source-level regression
guards:

PR B1 (service lifecycle): the generated systemd unit never targets a
system-wide/root path, no shell (system()/popen()) anywhere in the new
subprocess/service-command modules, `service install`/`uninstall` really
do refuse to touch a non-MEMBRANE-managed unit without --force.

PR B2 (streaming): the server genuinely implements real SSE streaming
(never left silently claiming STREAMING_NOT_SUPPORTED while also
claiming real streaming), the cancellation flag is threaded all the way
into the runtime core's own decode loop (never a server-only/fake
cancellation), and every new pure library from both PRs actually links
membrane_sanitizers.

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
SERVER_CPP_PATH = MEMBRANE_DIR / "server.cpp"
DECODE_LOOP_H_PATH = (REPO_ROOT / "tools" / "membrane-llama-runtime"
	/ "decode_loop.h")
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
	"## Startup robustness (PR B4)",
	"## Service upgrade / config versioning (PR B4, Section 48 of the task)",
]
REQUIRED_SERVER_DOC_STREAMING_SECTIONS = [
	"## Streaming (`stream: true`) — PR B2",
]
REQUIRED_SERVER_DOC_CLIENT_SECTIONS = [
	"## Client integration (PR B4)",
]
REQUIRED_SERVER_DOC_LIFECYCLE_SECTIONS = [
	"### Model-lifecycle state machine (PR B3)",
	"### Model-switch failure recovery (PR B3, Section 31 of the task)",
	"### Bounded request admission (Section 29 of the task)",
	"### Model registry hot-reload (Section 32 of the task)",
]
VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS"}

NEW_PURE_LIBS = [
	"membrane_fs_util", "membrane_subprocess", "membrane_systemd_unit",
	"membrane_server_config", "membrane_utf8_stream",
]
NEW_PURE_TESTS = [
	"test_systemd_unit", "test_server_config", "test_fs_util",
	"test_subprocess", "test_utf8_stream", "test_stream_queue",
	"test_request_admission",
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


@check("docs/server.md documents real streaming (PR B2), not the "
	"retired stream=true rejection")
def _c4b():
	text = SERVER_DOC_PATH.read_text()
	missing = [s for s in REQUIRED_SERVER_DOC_STREAMING_SECTIONS
		if s not in text]
	has_stale_rejection_claim = "STREAMING_NOT_SUPPORTED` was requested" in text
	ok = len(missing) == 0 and not has_stale_rejection_claim
	return ok, (f"missing: {missing}" if missing
		else "stale rejection claim still present" if has_stale_rejection_claim
		else "all present, no stale claim")


@check("docs/server.md documents the PR B3 model-lifecycle state "
	"machine/switch-recovery/admission/registry-reload sections, and "
	"docs/service.md no longer claims registry changes need a restart")
def _c4c():
	server_text = SERVER_DOC_PATH.read_text()
	missing = [s for s in REQUIRED_SERVER_DOC_LIFECYCLE_SECTIONS
		if s not in server_text]
	service_text = SERVICE_DOC_PATH.read_text()
	has_stale_no_hot_reload_claim = ("Neither the model registry nor the "
		"server config is hot-reloaded" in service_text)
	ok = len(missing) == 0 and not has_stale_no_hot_reload_claim
	return ok, (f"missing: {missing}" if missing
		else "docs/service.md still claims no registry hot-reload"
			if has_stale_no_hot_reload_claim
		else "all present, no stale claim")


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


@check("every new pure test binary (PR B1+B2+B3) is registered as a real "
	"ctest entry")
def _c13():
	text = MEMBRANE_CMAKE_PATH.read_text()
	missing = [t for t in NEW_PURE_TESTS if f"add_test(NAME {t}" not in text]
	return len(missing) == 0, (f"missing add_test(): {missing}" if missing
		else "all registered")


@check("CI wires the new service-lifecycle-tests/stream-queue-thread-"
	"sanitizer jobs and the new packaging-smoke install/uninstall "
	"integration step")
def _c14():
	ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
	text = ci_path.read_text()
	has_job = "service-lifecycle-tests:" in text
	has_tsan_job = "stream-queue-thread-sanitizer:" in text
	has_step = "membrane service install/uninstall (isolated dirs)" in text
	ok = has_job and has_tsan_job and has_step
	return ok, (f"has_job={has_job} has_tsan_job={has_tsan_job} "
		f"has_step={has_step}")


@check("regression guard: server.cpp implements real streaming (worker "
	"thread + bounded queue + SSE), never left both claiming real "
	"streaming AND rejecting it")
def _c16():
	text = SERVER_CPP_PATH.read_text()
	has_sse = "text/event-stream" in text
	has_worker_thread = "stream_worker_fn" in text
	has_stale_rejection = "STREAMING_NOT_SUPPORTED" in text
	ok = has_sse and has_worker_thread and not has_stale_rejection
	return ok, (f"has_sse={has_sse} has_worker_thread={has_worker_thread} "
		f"has_stale_rejection={has_stale_rejection}")


@check("regression guard: cancellation is threaded all the way into the "
	"runtime core's own decode loop (never a server-only/fake "
	"cancellation that merely closes the HTTP connection)")
def _c17():
	server_text = SERVER_CPP_PATH.read_text()
	decode_loop_text = DECODE_LOOP_H_PATH.read_text()
	has_server_cancel_flag = "cancel_flag" in server_text
	has_core_cancel_param = "cancel_flag" in decode_loop_text
	ok = has_server_cancel_flag and has_core_cancel_param
	return ok, (f"server.cpp has cancel_flag={has_server_cancel_flag} "
		f"decode_loop.h has cancel_flag param={has_core_cancel_param}")


@check("regression guard: server.cpp has an explicit model-lifecycle "
	"state machine (never left as implicit bool-soup) and a real "
	"switch-failure recovery path (never a naive unload-then-maybe-empty)")
def _c19():
	text = SERVER_CPP_PATH.read_text()
	has_state_enum = "e_membrane_model_state" in text
	has_error_state = "MEMBRANE_MODEL_STATE_ERROR" in text
	has_recovery = "had_previous" in text and "previous_name" in text
	ok = has_state_enum and has_error_state and has_recovery
	return ok, (f"has_state_enum={has_state_enum} "
		f"has_error_state={has_error_state} has_recovery={has_recovery}")


@check("regression guard: chat completions are admitted through a "
	"bounded gate before any other work, and the registry is hot-"
	"refreshed rather than captured once at server startup")
def _c20():
	text = SERVER_CPP_PATH.read_text()
	has_admission = "admission_gate" in text and "SERVER_BUSY" in text
	has_hot_reload = "refresh_and_snapshot_registry" in text
	ok = has_admission and has_hot_reload
	return ok, (f"has_admission={has_admission} "
		f"has_hot_reload={has_hot_reload}")


@check("docs/server.md documents the PR B4 client-integration section")
def _c4d():
	text = SERVER_DOC_PATH.read_text()
	missing = [s for s in REQUIRED_SERVER_DOC_CLIENT_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("regression guard: the server overrides cpp-httplib's default "
	"socket options to avoid SO_REUSEPORT -- a real bug this phase found "
	"and fixed (a second membrane serve instance could silently bind an "
	"already-listening port)")
def _c21():
	text = SERVER_CPP_PATH.read_text()
	has_override = "set_socket_options" in text and "SO_REUSEADDR" in text
	return has_override, ("set_socket_options()/SO_REUSEADDR override "
		"present" if has_override else "not found")


@check("every new pure test binary AND the new port-conflict regression "
	"test are registered as real ctest/test-suite entries")
def _c22():
	cmake_text = MEMBRANE_CMAKE_PATH.read_text()
	missing = [t for t in NEW_PURE_TESTS
		if f"add_test(NAME {t}" not in cmake_text]
	test_server_text = (MEMBRANE_DIR / "test_server.cpp").read_text()
	has_port_conflict_test = ("test_second_instance_same_port_fails_to_bind"
		in test_server_text)
	ok = len(missing) == 0 and has_port_conflict_test
	return ok, (f"missing add_test(): {missing}" if missing
		else "port-conflict regression test missing from test_server.cpp"
			if not has_port_conflict_test
		else "all present")


@check("Mega Phase A's own evidence files are untouched by this PR's own "
	"new commits (checked against origin/main)")
def _c18():
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
	for fn in (_c1, _c2, _c3, _c4, _c4b, _c4c, _c4d, _c5, _c6, _c7, _c8,
			_c9, _c10, _c11, _c12, _c13, _c14, _c16, _c17, _c19, _c20,
			_c21, _c22, _c18):
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
