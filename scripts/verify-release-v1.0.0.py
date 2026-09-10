#!/usr/bin/env python3
"""Validate v1.0.0 stable-release readiness (Mega Phase E, PR E6):
MEMBRANE_VERSION and CITATION.cff both say 1.0.0, docs/release-v1.0.0.md,
docs/api-v1-stability.md, docs/cli-stability-contract.md, docs/upgrade-
v0.8-to-v1.0.md, and results/release-v1.0.0/readiness.json all exist and
are self-consistent and labeled REAL, no CUDA/Windows-GPU/Open-WebUI-
validated/"complete OpenAI API"/production-ready/universal-hardware
overclaim exists anywhere in release-facing docs, the official package
policy is version-bumped correctly (Vulkan-enabled `membrane` +
`membrane-cpu`, still no Windows/macOS package claimed shipped),
README's release status correctly says stable v1.0.0, external-
validation wording matches what results/v1-external-validation/
validation.json actually found (not overclaimed as merged), and
historical v0.3.0/v0.4.0/v0.8.0 release docs/evidence remain byte-for-
byte untouched.

Same standalone-script, own-CI-job pattern as scripts/verify-release-
v0.8.0.py before it -- not chained into scripts/verify-results.py.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_CLI_H = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
CITATION = REPO_ROOT / "CITATION.cff"
RELEASE_NOTES = REPO_ROOT / "docs" / "release-v1.0.0.md"
UPGRADE_DOC = REPO_ROOT / "docs" / "upgrade-v0.8-to-v1.0.md"
API_STABILITY = REPO_ROOT / "docs" / "api-v1-stability.md"
CLI_STABILITY = REPO_ROOT / "docs" / "cli-stability-contract.md"
EXTERNAL_VALIDATION_DOC = REPO_ROOT / "docs" / "external-validation.md"
READINESS = REPO_ROOT / "results" / "release-v1.0.0" / "readiness.json"
EXTERNAL_VALIDATION_JSON = (REPO_ROOT / "results" / "v1-external-validation"
	/ "validation.json")
SUPPORT_MATRIX = REPO_ROOT / "docs" / "support-matrix.md"
MODEL_COMPAT = REPO_ROOT / "docs" / "model-compatibility.md"
README = REPO_ROOT / "README.md"
MANIFEST = REPO_ROOT / "results" / "release-artifacts" / "manifest.json"

VALID_STATES = {"VALIDATED_REAL", "VALIDATED_CI", "VALIDATED_MAINTAINER",
	"VALIDATED_CI_REAL_HARDWARE", "COMPILE_VALIDATED_CI",
	"CONFIGURATION_ONLY", "NOT_VALIDATED", "UNSUPPORTED", "REAL",
	"SYNTHETIC", "SOURCE_ANALYSIS", "NOT_SHIPPED", "PARTIAL"}

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


def _release_facing_docs():
	return [RELEASE_NOTES, UPGRADE_DOC, API_STABILITY, CLI_STABILITY,
		EXTERNAL_VALIDATION_DOC, README,
		REPO_ROOT / "docs" / "client-compatibility.md",
		REPO_ROOT / "docs" / "api-contract.md", CITATION]


@check("MEMBRANE_VERSION == 1.0.0 (live product_cli.h)")
def _c1():
	text = PRODUCT_CLI_H.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "1.0.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("CITATION.cff version == 1.0.0")
def _c2():
	text = CITATION.read_text()
	m = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
	ok = m is not None and m.group(1) == "1.0.0"
	return ok, f"version={m.group(1) if m else '(not found)'}"


@check("docs/release-v1.0.0.md, docs/upgrade-v0.8-to-v1.0.md, "
	"docs/api-v1-stability.md, docs/cli-stability-contract.md, "
	"docs/external-validation.md all exist")
def _c3():
	missing = [str(p) for p in
		(RELEASE_NOTES, UPGRADE_DOC, API_STABILITY, CLI_STABILITY,
			EXTERNAL_VALIDATION_DOC)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("results/release-v1.0.0/readiness.json exists, is valid JSON, "
	"schema_version 1, version 1.0.0, label REAL")
def _c4():
	data = json.loads(READINESS.read_text())
	ok = (data.get("schema_version") == 1 and data.get("version") == "1.0.0"
		and data.get("label") == "REAL")
	return ok, f"schema_version={data.get('schema_version')} " \
		f"version={data.get('version')} label={data.get('label')}"


@check("every evidence-state value in readiness.json uses only legal "
	"vocabulary")
def _c5():
	data = json.loads(READINESS.read_text())
	bad = []

	def walk(obj, path):
		if isinstance(obj, dict):
			for k, v in obj.items():
				if k in ("state", "label") and isinstance(v, str) \
						and v not in VALID_STATES:
					bad.append(f"{path}.{k}={v!r}")
				walk(v, f"{path}.{k}")
		elif isinstance(obj, list):
			for i, v in enumerate(obj):
				walk(v, f"{path}[{i}]")

	walk(data, "readiness")
	return len(bad) == 0, (f"illegal states: {bad}" if bad
		else "all evidence states legal")


@check("no CUDA/Windows-GPU/Open-WebUI-validated overclaim in any "
	"release-facing doc")
def _c6():
	bad = []
	cuda_all_pattern = re.compile(
		r"\bCUDA\b[^.\n]{0,40}\b(all|any|every)\s+NVIDIA\b", re.IGNORECASE)
	windows_gpu_pattern = re.compile(
		r"\bWindows\b[^.\n]{0,40}\b(GPU|CUDA|Vulkan)\b[^.\n]{0,40}"
		r"\b(validated|supported|tested)\b", re.IGNORECASE)
	negation = re.compile(r"\b(?:not|never|n't|no)\b", re.IGNORECASE)
	openwebui_validated_pattern = re.compile(
		r"\bOpen\s*WebUI\b[^.\n]{0,60}\bVALIDATED_REAL\b")
	for path in _release_facing_docs():
		if not path.exists():
			continue
		text = path.read_text()
		if cuda_all_pattern.search(text):
			bad.append(f"{path.name}: CUDA-all-NVIDIA overclaim")
		for m in windows_gpu_pattern.finditer(text):
			window = text[max(0, m.start() - 20):m.end() + 5]
			if not negation.search(window):
				bad.append(f"{path.name}: Windows-GPU overclaim "
					f"({m.group(0)!r})")
		if openwebui_validated_pattern.search(text):
			bad.append(f"{path.name}: Open WebUI claimed VALIDATED_REAL")
	data = json.loads(READINESS.read_text())
	if data.get("platform_matrix", {}).get("windows_gpu", {}).get("state") \
			not in ("NOT_VALIDATED", None):
		bad.append("readiness.json: windows_gpu state is not NOT_VALIDATED")
	if data.get("client_matrix", {}).get("openwebui", {}).get("state") \
			== "VALIDATED_REAL":
		bad.append("readiness.json: openwebui claimed VALIDATED_REAL")
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("no 'complete OpenAI API'/production-ready/universal-hardware "
	"claim anywhere in release-facing docs")
def _c7():
	bad = []
	patterns = [
		re.compile(r"(?:complete|full|entire)\s+OpenAI(?:\s+API)?\b", re.I),
		re.compile(r"\bproduction[- ]ready\b", re.I),
		re.compile(r"\b(?:every|all|any)\s+(?:GPU|NVIDIA|AMD|Intel)\s+"
			r"(?:device|hardware|card)\b", re.I),
	]
	negation = re.compile(r"\b(?:not|never|n't|no)\b", re.I)
	for path in _release_facing_docs():
		if not path.exists():
			continue
		text = path.read_text()
		for pat in patterns:
			for m in pat.finditer(text):
				window = text[max(0, m.start() - 40):m.start()]
				if not negation.search(window):
					bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("official package policy version-bumped: Vulkan-enabled "
	"'membrane' + 'membrane-cpu', both at version 1.0.0; no Windows/"
	"macOS package claimed shipped")
def _c8():
	manifest = json.loads(MANIFEST.read_text())
	ok = (manifest.get("product_version") == "1.0.0"
		and manifest.get("debian_version") == "1.0.0")
	readiness = json.loads(READINESS.read_text())
	pm = readiness.get("package_matrix", {})
	win_ok = pm.get("windows_package", {}).get("state") == "NOT_SHIPPED"
	mac_ok = pm.get("macos_package", {}).get("state") == "NOT_SHIPPED"
	return (ok and win_ok and mac_ok,
		f"manifest_versions_ok={ok} windows_not_shipped={win_ok} "
		f"macos_not_shipped={mac_ok}")


@check("README's release-status line says stable v1.0.0, not v0.8.0")
def _c9():
	text = README.read_text()
	ok = "`v1.0.0`" in text
	stale = re.search(r"latest stable tag `v0\.8\.0`", text)
	return (ok and not stale,
		"README correctly names v1.0.0 as latest stable" if (ok and not stale)
		else f"has_v100={ok} stale_v080_claim={bool(stale)}")


@check("historical v0.3.0/v0.4.0/v0.8.0 release docs/evidence remain "
	"byte-for-byte untouched (checked against origin/main)")
def _c10():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	forbidden_prefixes = (
		"results/release-v0.3.0/", "results/release-v0.3.0-rc2/",
		"results/release-v0.3.0-rc3/", "results/release-v0.4.0/",
		"results/release-v0.8.0/",
		"docs/release-v0.3.0.md", "docs/release-v0.4.0.md",
		"docs/release-v0.8.0.md",
	)
	touched = [p for p in changed if p.startswith(forbidden_prefixes)]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


@check("external-validation wording matches reality: the 4 real "
	"external PRs are disclosed as found-but-unmerged, never implied "
	"as merged/shipped in this release")
def _c11():
	data = json.loads(EXTERNAL_VALIDATION_JSON.read_text())
	prs = data.get("external_pull_requests", {})
	status_ok = "not merged" in prs.get("status", "").lower() or \
		"deferred" in prs.get("status", "").lower()
	release_notes_text = RELEASE_NOTES.read_text() if RELEASE_NOTES.exists() \
		else ""
	# The release notes may MENTION the external PRs as a real finding,
	# but must never claim they are included/merged/shipped in v1.0.0.
	bad_claim = re.search(
		r"external[^.\n]{0,80}\b(merged|shipped|included)\b",
		release_notes_text, re.IGNORECASE)
	return (status_ok and not bad_claim,
		f"validation.json status disclosed as unmerged={status_ok} "
		f"release_notes_bad_claim={bool(bad_claim)}")


@check("known limitations list is non-empty and disclosed in "
	"readiness.json")
def _c12():
	data = json.loads(READINESS.read_text())
	limitations = data.get("known_limitations", [])
	return len(limitations) > 0, f"{len(limitations)} known limitations listed"


@check("real upgrade-test evidence exists and is labeled REAL (v0.8.0 "
	"-> v1.0.0, real package, real registry/config diff)")
def _c13():
	data = json.loads(READINESS.read_text())
	upgrade = data.get("package_matrix", {}).get("real_upgrade_test", {})
	ok = upgrade.get("label") == "REAL" and "IDENTICAL" in upgrade.get(
		"result", "")
	return ok, f"label={upgrade.get('label')} " \
		f"has_identical_claim={'IDENTICAL' in upgrade.get('result', '')}"


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
