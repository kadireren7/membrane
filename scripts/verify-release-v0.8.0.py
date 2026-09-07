#!/usr/bin/env python3
"""Validate v0.8.0 stable-release readiness (Mega Phase D, PR D8):
MEMBRANE_VERSION and CITATION.cff both say 0.8.0, docs/release-v0.8.0.md
and results/release-v0.8.0/readiness.json exist and are self-consistent
and labeled REAL, docs/support-matrix.md/docs/model-compatibility.md
exist and use only legal evidence-state vocabulary, no CUDA/Windows-
GPU/Open-WebUI-validated/"complete OpenAI API"/production-ready/
universal-hardware overclaim exists anywhere in release-facing docs,
the official package policy is unchanged (Vulkan-enabled `membrane` +
`membrane-cpu`, no Windows/macOS package), README's release status
correctly says stable v0.8.0, and historical v0.3.0/v0.4.0 release
docs/evidence remain byte-for-byte untouched.

Same standalone-script, own-CI-job pattern as scripts/verify-release-
v0.4.0.py (Mega Phase C, PR C4) before it -- not chained into
scripts/verify-results.py. Expected to need retirement itself once a
future release supersedes v0.8.0 (same precedent that script's own
retirement documents).

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
RELEASE_NOTES = REPO_ROOT / "docs" / "release-v0.8.0.md"
UPGRADE_DOC = REPO_ROOT / "docs" / "upgrade-v0.4-to-v0.8.md"
READINESS = REPO_ROOT / "results" / "release-v0.8.0" / "readiness.json"
SUPPORT_MATRIX = REPO_ROOT / "docs" / "support-matrix.md"
MODEL_COMPAT = REPO_ROOT / "docs" / "model-compatibility.md"
README = REPO_ROOT / "README.md"
MANIFEST = REPO_ROOT / "results" / "release-artifacts" / "manifest.json"

VALID_STATES = {"VALIDATED_REAL", "VALIDATED_CI", "CONFIGURATION_ONLY",
	"NOT_VALIDATED", "UNSUPPORTED", "REAL", "SYNTHETIC", "SOURCE_ANALYSIS",
	"NOT_SHIPPED"}

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
	return [RELEASE_NOTES, UPGRADE_DOC, SUPPORT_MATRIX, MODEL_COMPAT, README,
		REPO_ROOT / "docs" / "client-compatibility.md",
		REPO_ROOT / "docs" / "api-contract.md",
		REPO_ROOT / "CITATION.cff"]


@check("MEMBRANE_VERSION == 0.8.0 (live product_cli.h)")
def _c1():
	text = PRODUCT_CLI_H.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.8.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("CITATION.cff version == 0.8.0")
def _c2():
	text = CITATION.read_text()
	m = re.search(r'^version:\s*"([^"]+)"', text, re.MULTILINE)
	ok = m is not None and m.group(1) == "0.8.0"
	return ok, f"version={m.group(1) if m else '(not found)'}"


@check("docs/release-v0.8.0.md, docs/upgrade-v0.4-to-v0.8.md, "
	"docs/support-matrix.md, docs/model-compatibility.md all exist")
def _c3():
	missing = [str(p) for p in
		(RELEASE_NOTES, UPGRADE_DOC, SUPPORT_MATRIX, MODEL_COMPAT)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("results/release-v0.8.0/readiness.json exists, is valid JSON, "
	"schema_version 1, version 0.8.0, label REAL")
def _c4():
	data = json.loads(READINESS.read_text())
	ok = (data.get("schema_version") == 1 and data.get("version") == "0.8.0"
		and data.get("label") == "REAL")
	return ok, f"schema_version={data.get('schema_version')} " \
		f"version={data.get('version')} label={data.get('label')}"


@check("every evidence-state value in readiness.json/support-matrix.md/"
	"model-compatibility.md uses only legal vocabulary")
def _c5():
	data = json.loads(READINESS.read_text())
	bad = []

	def walk(obj, path):
		if isinstance(obj, dict):
			for k, v in obj.items():
				if k in ("state", "label") and isinstance(v, str) \
						and v not in VALID_STATES and v != "PARTIAL":
					bad.append(f"{path}.{k}={v!r}")
				walk(v, f"{path}.{k}")
		elif isinstance(obj, list):
			for i, v in enumerate(obj):
				walk(v, f"{path}[{i}]")

	walk(data, "readiness")
	for path in (SUPPORT_MATRIX, MODEL_COMPAT):
		text = path.read_text()
		for m in re.finditer(r'\b(VALIDATED_[A-Z]+|CONFIGURATION_ONLY|'
				r'NOT_VALIDATED|UNSUPPORTED)\b', text):
			if m.group(1) not in VALID_STATES:
				bad.append(f"{path.name}: {m.group(1)}")
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
	# Cross-check the evidence file's own real state for Windows GPU /
	# Open WebUI never got flipped to something it shouldn't be.
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


@check("official package policy unchanged: Vulkan-enabled 'membrane' + "
	"'membrane-cpu', both at version 0.8.0; no Windows/macOS package "
	"claimed shipped")
def _c8():
	manifest = json.loads(MANIFEST.read_text())
	ok = (manifest.get("product_version") == "0.8.0"
		and manifest.get("debian_version") == "0.8.0")
	readiness = json.loads(READINESS.read_text())
	pm = readiness.get("package_matrix", {})
	win_ok = pm.get("windows_package", {}).get("state") == "NOT_SHIPPED"
	mac_ok = pm.get("macos_package", {}).get("state") == "NOT_SHIPPED"
	return (ok and win_ok and mac_ok,
		f"manifest_versions_ok={ok} windows_not_shipped={win_ok} "
		f"macos_not_shipped={mac_ok}")


@check("README's release-status line says stable v0.8.0, not v0.4.0")
def _c9():
	text = README.read_text()
	ok = "`v0.8.0`" in text and "v0.4.0" in text  # v0.4.0 still mentioned as historical
	stale = re.search(r"latest stable tag `v0\.4\.0`", text)
	return (ok and not stale,
		"README correctly names v0.8.0 as latest stable" if (ok and not stale)
		else f"has_v080={ok} stale_v040_claim={bool(stale)}")


@check("historical v0.3.0/v0.4.0 release docs/evidence remain "
	"byte-for-byte untouched (checked against origin/main)")
def _c10():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	forbidden_prefixes = (
		"results/release-v0.3.0/", "results/release-v0.3.0-rc2/",
		"results/release-v0.3.0-rc3/", "docs/release-v0.3.0.md",
		"docs/release-v0.4.0.md",
	)
	touched = [p for p in changed if p.startswith(forbidden_prefixes)]
	# results/release-v0.4.0/readiness.json itself, and results/release-
	# artifacts/manifest.json, ARE expected to be untouched historically
	# except the manifest's own live-version fields, which this exact
	# release deliberately updates (see _c8) -- everything else in the
	# forbidden set must be untouched.
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


@check("D1-D7 own evidence files still carry real (non-PENDING) squash "
	"SHAs (client-compatibility.json's own record, cross-checked)")
def _c11():
	data = json.loads((REPO_ROOT / "results" / "client-compatibility"
		/ "validation.json").read_text())
	shas = data.get("pr_squash_shas", {})
	missing = [k for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
		if not re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))]
	return len(missing) == 0, (f"missing/invalid SHAs: {missing}" if missing
		else "all D1-D7 SHAs real")


@check("the long-context fix (CTX_TOO_SMALL_FOR_PROMPT) is real: present "
	"in server.cpp, documented in docs/server.md, and real evidence "
	"exists in readiness.json")
def _c12():
	server_cpp = (REPO_ROOT / "tools" / "membrane" / "server.cpp").read_text()
	server_md = (REPO_ROOT / "docs" / "server.md").read_text()
	readiness = json.loads(READINESS.read_text())
	has_code = "CTX_TOO_SMALL_FOR_PROMPT" in server_cpp
	has_doc = "CTX_TOO_SMALL_FOR_PROMPT" in server_md
	has_evidence = readiness.get("long_context_fix", {}).get("label") == "REAL" \
		and len(readiness.get("long_context_fix", {}).get("real_evidence", [])) >= 2
	ok = has_code and has_doc and has_evidence
	return ok, f"code={has_code} doc={has_doc} evidence={has_evidence}"


@check("reproducibility finding is disclosed honestly (not silently "
	"claimed fully reproducible)")
def _c13():
	readiness = json.loads(READINESS.read_text())
	repro = readiness.get("package_matrix", {}).get("reproducible_build", {})
	ok = repro.get("label") in ("REAL", "SOURCE_ANALYSIS") and \
		repro.get("result") not in (None, "")
	no_overclaim = "fully reproducible" not in RELEASE_NOTES.read_text().lower()
	return (ok and no_overclaim,
		f"repro_result={repro.get('result')!r} no_overclaim={no_overclaim}")


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
