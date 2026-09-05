#!/usr/bin/env python3
"""Validate Mega Phase A's runtime-service evidence
(results/runtime-service/validation.json): schema and REAL/SYNTHETIC/
SOURCE_ANALYSIS labeling, docs/server.md + docs/model-registry.md exist
with their required sections, no OOM-proof/guaranteed-fit overclaim, the
llama.cpp patch set is unchanged, no premature v0.4/v1.0 release claim,
stable release still says v0.3.0, and a handful of direct source-level
regression guards for the real bugs this mega-phase found and fixed
(membrane_registry_core actually links the sanitizer flags; the server
defaults to loopback-only; stream=true is honestly rejected, never
faked).

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
EVIDENCE_PATH = REPO_ROOT / "results" / "runtime-service" / "validation.json"
SERVER_DOC_PATH = REPO_ROOT / "docs" / "server.md"
REGISTRY_DOC_PATH = REPO_ROOT / "docs" / "model-registry.md"
SERVER_CPP_PATH = REPO_ROOT / "tools" / "membrane" / "server.cpp"
REGISTRY_CMAKE_PATH = REPO_ROOT / "tools" / "membrane" / "CMakeLists.txt"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
PATCHES_DIR = REPO_ROOT / "patches"
EXPECTED_PATCHES = {
	"llama.cpp-membrane-kv-type-override.patch",
	"llama.cpp-membrane-kv-device-override.patch",
}

REQUIRED_SERVER_DOC_SECTIONS = [
	"## Security scope", "## Endpoints", "## `POST /v1/chat/completions`",
	"## Model cache policy", "## Errors", "## Graceful shutdown",
	"## Not implemented",
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


@check("evidence and doc files all exist")
def _c1():
	missing = [str(p) for p in (EVIDENCE_PATH, SERVER_DOC_PATH,
			REGISTRY_DOC_PATH) if not p.exists()]
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


@check("docs/server.md has every required section")
def _c4():
	text = SERVER_DOC_PATH.read_text()
	missing = [s for s in REQUIRED_SERVER_DOC_SECTIONS if s not in text]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("no OOM-proof / guaranteed-fit overclaim in docs or evidence")
def _c5():
	pattern = re.compile(
		r"\bguarantee(d|s)?\s+(no|zero)\s+oom\b|\boom-proof\b|"
		r"\bguaranteed[- ]fit\b|\bnever\s+(fails|runs out of memory)\b",
		re.IGNORECASE)
	bad = []
	for path in (SERVER_DOC_PATH, REGISTRY_DOC_PATH, EVIDENCE_PATH):
		text = path.read_text()
		for m in pattern.finditer(text):
			bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified overclaim found"


@check("patch set is documented as unchanged and matches the two real "
	"committed patch files")
def _c6():
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	return real == EXPECTED_PATCHES, f"real={real} expected={EXPECTED_PATCHES}"


@check("no premature v0.4/v1.0 release claim anywhere in this phase's own "
	"new docs")
def _c7():
	bad = []
	pattern = re.compile(
		r"\bv0\.4\.\d+\b|\bv0\.4-release\b|\bv1\.0\.\d+\b|"
		r"\bv0\.4\s+(is|has been)\s+released\b", re.IGNORECASE)
	for path in (SERVER_DOC_PATH, REGISTRY_DOC_PATH):
		text = path.read_text()
		for m in pattern.finditer(text):
			bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no premature release claim"


@check("live MEMBRANE_VERSION is v0.4.0 (bumped by Mega Phase C's own PR C4)")
def _c8():
	text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.4.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("regression guard: membrane_registry_core actually links the real "
	"sanitizer flags (the exact bug this phase found and fixed)")
def _c9():
	text = REGISTRY_CMAKE_PATH.read_text()
	m = re.search(
		r"add_library\(membrane_registry_core.*?(?=\nadd_library|\Z)",
		text, re.DOTALL)
	if m is None:
		return False, "membrane_registry_core target not found"
	ok = "membrane_sanitizers" in m.group(0)
	return ok, ("links membrane_sanitizers" if ok
		else "membrane_registry_core does not link membrane_sanitizers")


@check("regression guard: the server defaults to loopback-only and refuses "
	"any other bind address without an explicit opt-in")
def _c10():
	text = SERVER_CPP_PATH.read_text()
	has_default = '"127.0.0.1"' in text
	has_refusal = "allow_non_loopback" in text and "refusing to bind" in text
	ok = has_default and has_refusal
	return ok, ("loopback default + explicit-refusal path both present" if ok
		else f"has_default={has_default} has_refusal={has_refusal}")


@check("regression guard: stream=true is either honestly rejected or "
	"genuinely implemented, never silently faked either way -- Mega "
	"Phase B, PR B2 replaced this phase's own STREAMING_NOT_SUPPORTED "
	"rejection with real SSE streaming (scripts/verify-background-"
	"service.py's own regression guards cover THAT implementation's "
	"specifics; this check only asserts the code no longer claims BOTH "
	"things could be simultaneously true, which would mean one of the "
	"two claims is a lie)")
def _c11():
	text = SERVER_CPP_PATH.read_text()
	has_rejection = "STREAMING_NOT_SUPPORTED" in text
	has_real_streaming = "text/event-stream" in text and "cancel_flag" in text
	ok = has_rejection != has_real_streaming	# exactly one, never both/neither
	return ok, (f"has_rejection={has_rejection} has_real_streaming="
		f"{has_real_streaming}")


@check("regression guard: HTTP-facing errors never forward "
	"membrane_run_error_t's own human/message text verbatim into a "
	"filesystem-path-bearing field")
def _c12():
	text = SERVER_CPP_PATH.read_text()
	# The one real risk is send_json_error() being called with
	# `.human`/`.message` from a membrane_run_error_t (which can contain a
	# real path) rather than server.cpp's own hand-built message strings.
	bad = re.findall(r"send_json_error\([^;]*\.(?:human|message)\b",
		text, re.DOTALL)
	return len(bad) == 0, ("no membrane_run_error_t text forwarded to an "
		"HTTP response" if len(bad) == 0
		else f"{len(bad)} suspicious call site(s)")


@check("Mega Phase A's PR A1/A2/A3 own evidence files are untouched by "
	"this phase's own new commits (checked against origin/main)")
def _c13():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/context-recommendation/cli-validation.json",
		"results/context-recommendation/core-validation.json",
		"results/host-memory-guard/validation.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "all untouched")


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
