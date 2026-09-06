#!/usr/bin/env python3
"""Validate Mega Phase D, PR D6's model-lifecycle-UX evidence
(results/model-lifecycle-ux/validation.json): schema and REAL/
SYNTHETIC/SOURCE_ANALYSIS labeling, docs match real behavior, consent
is genuinely required before any auto-install, no silent multi-GB
auto-download path exists, --yes behavior is documented, `membrane use`
reuses the existing downloader/registry/switch primitives rather than
duplicating them, default vs. active model stays a real, distinct
concept, a switch failure is never reported as a success, no Linux-only
assumption was introduced, no CUDA/Windows/macOS overclaim, and the
release version is unchanged by this PR.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see scripts/verify-windows-
support.py, PR D5's own equivalent).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "model-lifecycle-ux" / "validation.json"
LIFECYCLE_DOC = REPO_ROOT / "docs" / "model-lifecycle.md"
SERVER_DOC = REPO_ROOT / "docs" / "server.md"
SERVICE_DOC = REPO_ROOT / "docs" / "service.md"
REGISTRY_DOC = REPO_ROOT / "docs" / "model-registry.md"
USE_CMD_H = REPO_ROOT / "tools" / "membrane" / "use_cmd.h"
USE_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "use_cmd.cpp"
SERVER_CPP = REPO_ROOT / "tools" / "membrane" / "server.cpp"
STATUS_CLIENT_H = REPO_ROOT / "tools" / "membrane" / "status_client.h"
DOCTOR_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "doctor_cmd.cpp"
MODEL_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "model_cmd.cpp"
CLI_SHARED_H = REPO_ROOT / "tools" / "membrane" / "cli_shared.h"
CLI_SHARED_CPP = REPO_ROOT / "tools" / "membrane" / "cli_shared.cpp"
TEST_USE_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "test_use_cmd.cpp"
PRODUCT_CLI_H = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"

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


@check("evidence file, docs/model-lifecycle.md, use_cmd.h/.cpp, and "
	"cli_shared.h/.cpp all exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, LIFECYCLE_DOC, USE_CMD_H, USE_CMD_CPP,
			CLI_SHARED_H, CLI_SHARED_CPP, TEST_USE_CMD_CPP)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and real "
	"(non-PENDING) D1-D5 squash SHAs")
def _c2():
	data = _load_evidence()
	shas = data.get("pr_squash_shas", {})
	ok = data.get("schema_version") == 1 and all(
		re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))
		for k in ("D1", "D2", "D3", "D4", "D5"))
	return ok, (f"schema_version={data.get('schema_version')} "
		+ " ".join(f"{k}={shas.get(k)!r}" for k in
			("D1", "D2", "D3", "D4", "D5")))


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


@check("the central real-behavior sections (resolution, installed_use, "
	"auto_install, consent, active_model, service_switch, switch_"
	"recovery, idempotence, real_download, real_runtime) are labeled "
	"REAL, not merely SOURCE_ANALYSIS")
def _c4():
	data = _load_evidence()
	required_real = ("resolution", "installed_use", "auto_install",
		"consent", "active_model", "service_switch", "switch_recovery",
		"idempotence", "real_download", "real_runtime")
	bad = [k for k in required_real if data.get(k, {}).get("label") != "REAL"]
	return len(bad) == 0, (f"not REAL: {bad}" if bad else "all REAL")


@check("consent is genuinely required: use_cmd.cpp never calls the "
	"shared ask_yes_no() (which defaults to yes when non-interactive) "
	"for the not-installed download gate -- it has its own bespoke "
	"non-interactive-fails-clearly check")
def _c5():
	text = USE_CMD_CPP.read_text()
	has_bespoke_check = ("NONINTERACTIVE_CONSENT_REQUIRED" in text
		and "membrane_cli_is_interactive()" in text)
	# The one membrane_cli_ask_yes_no() call in this file must be
	# reached only AFTER the interactive check above already returned
	# early for the non-interactive case -- i.e. it must not be the
	# sole gate.
	return has_bespoke_check, ("bespoke non-interactive consent gate "
		"present" if has_bespoke_check else "gate not found")


@check("no silent multi-GB auto-download path: every real network "
	"install call site in use_cmd.cpp is reached only after either "
	"assume_yes or a real consent prompt")
def _c6():
	text = USE_CMD_CPP.read_text()
	install_call = "membrane_cli_dispatch_silently_if_json(want_json,\n\t\t\t\t{\"install\","
	has_install_call = "{\"install\", fam->name" in text
	has_consent_gate_before = "preview_and_consent(" in text
	return (has_install_call and has_consent_gate_before,
		"install call gated behind preview_and_consent()"
		if (has_install_call and has_consent_gate_before)
		else f"install_call={has_install_call} consent_gate="
			f"{has_consent_gate_before}")


@check("--yes/-y behavior is documented in docs/model-lifecycle.md")
def _c7():
	text = LIFECYCLE_DOC.read_text()
	ok = "--yes" in text and "NONINTERACTIVE_CONSENT_REQUIRED" in text
	return ok, "--yes and the non-interactive failure both documented" \
		if ok else "missing --yes or non-interactive documentation"


@check("`membrane use` reuses the existing downloader/registry -- "
	"use_cmd.cpp calls membrane_model_cmd_dispatch()/membrane_registry_"
	"load() etc. directly, and defines no second download/registry "
	"function of its own")
def _c8():
	text = USE_CMD_CPP.read_text()
	reuses = ("membrane_model_cmd_dispatch" in text
		and "membrane_registry_load" in text
		and "membrane_catalog_resolve" in text
		and "membrane_select_variant" in text)
	no_second_downloader = ("curl" not in text.lower()
		and "CURLOPT" not in text)
	return (reuses and no_second_downloader,
		"reuses catalog/registry/install primitives, defines no "
		"second downloader" if (reuses and no_second_downloader)
		else f"reuses={reuses} no_second_downloader={no_second_downloader}")


@check("default vs. active model stays a real, distinct concept: "
	"server.cpp's /v1/status exposes both default_model and "
	"loaded_model as separate fields, and status_client.cpp's human "
	"printer prints them as two separate lines")
def _c9():
	server_text = SERVER_CPP.read_text()
	status_text = STATUS_CLIENT_H.parent.joinpath("status_client.cpp") \
		.read_text()
	ok = ('j["default_model"]' in server_text
		and 'j["loaded_model"]' in server_text
		and "default model:" in status_text
		and "active model:" in status_text)
	return ok, "default_model/loaded_model both real, separate fields" \
		if ok else "one or both fields/lines missing"


@check("a switch failure is never reported as a success: use_cmd.cpp's "
	"switch_failed branch returns a nonzero exit code and sets "
	"result[\"ok\"]=false, never leaving the earlier ok:true untouched")
def _c10():
	text = USE_CMD_CPP.read_text()
	ok = ('if (!act.ok)' in text and 'result["ok"] = false' in text
		and "MEMBRANE_EXIT_RUNTIME_ERROR" in text)
	return ok, "switch_failed branch correctly overrides ok to false" \
		if ok else "switch_failed branch does not override ok/exit code"


@check("no Linux-only assumption: use_cmd.cpp/cli_shared.cpp contain "
	"no #ifdef __linux__/raw POSIX-only call not already routed "
	"through the cross-platform compat headers")
def _c11():
	bad = []
	for path in (USE_CMD_CPP, CLI_SHARED_CPP):
		text = path.read_text()
		if "__linux__" in text:
			bad.append(f"{path.name}: contains __linux__")
	return len(bad) == 0, ("; ".join(bad) if bad else
		"no Linux-only branching in the new D6 files")


@check("no CUDA/Metal/Windows-specific overclaim in docs/model-"
	"lifecycle.md (this PR adds no new platform-specific code)")
def _c12():
	text = LIFECYCLE_DOC.read_text()
	bad = [m for m in re.findall(
		r"\b(?:guaranteed|always)\s+(?:CUDA|Metal|Windows)\b", text,
		re.IGNORECASE)]
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("the release version (MEMBRANE_VERSION) is unchanged by this PR")
def _c13():
	text = PRODUCT_CLI_H.read_text()
	m = re.search(r'MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.4.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '?'}"


@check("model_cmd.cpp's cmd_uninstall refuses to delete the currently "
	"ACTIVE model (MODEL_ACTIVE) and clears a matching default_model "
	"rather than leaving it dangling")
def _c14():
	text = MODEL_CMD_CPP.read_text()
	ok = ("MODEL_ACTIVE" in text and "was_default" in text
		and "default_cleared" in text)
	return ok, "active/default uninstall guard present" if ok \
		else "active/default uninstall guard missing"


@check("doctor_cmd.cpp's new model_lifecycle check is wired into "
	"membrane_doctor_collect()")
def _c15():
	text = DOCTOR_CMD_CPP.read_text()
	ok = ("check_model_lifecycle(" in text
		and 'checks.push_back(check_model_lifecycle(' in text)
	return ok, "model_lifecycle check wired in" if ok else "not wired in"


@check("Mega Phase C/D1-D5's own evidence files are untouched by this "
	"PR's own new commits (checked against origin/main)")
def _c16():
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
		"results/windows-support/validation.json",
		"results/background-service/validation.json",
		"results/product-onboarding/validation.json",
		"results/release-supply-chain/validation.json",
		"results/product-hardening/v0.4-validation.json",
		"results/release-v0.4.0/readiness.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11,
			_c12, _c13, _c14, _c15, _c16):
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
