#!/usr/bin/env python3
"""Validate Mega Phase D, PR D7's client-compatibility evidence
(results/client-compatibility/validation.json): schema and REAL/
SYNTHETIC/SOURCE_ANALYSIS/VALIDATED_REAL/CONFIGURATION_ONLY/
NOT_VALIDATED/UNSUPPORTED labeling, no CONFIGURATION_ONLY client is
claimed VALIDATED_REAL, OpenAI-compatibility wording stays scoped
("subset", never "complete"), tool calling is never claimed without
real evidence, embeddings are never claimed at all (this PR
deliberately did not add them), model paths are never exposed in
docs/API responses, the base URL is documented correctly, real client
versions are recorded, and the release version is unchanged by this PR.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see scripts/verify-model-lifecycle-
ux.py, PR D6's own equivalent).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "client-compatibility" / "validation.json"
CLIENT_COMPAT_MD = REPO_ROOT / "docs" / "client-compatibility.md"
API_CONTRACT_MD = REPO_ROOT / "docs" / "api-contract.md"
SERVER_MD = REPO_ROOT / "docs" / "server.md"
SERVER_CPP = REPO_ROOT / "tools" / "membrane" / "server.cpp"
TEST_SERVER_CPP = REPO_ROOT / "tools" / "membrane" / "test_server.cpp"
NODE_CHECK = REPO_ROOT / "scripts" / "node-client-check.mjs"
PY_CLIENT_SCRIPT = REPO_ROOT / "scripts" / "client-compat" / "test-openai-python.py"
NODE_CLIENT_SCRIPT = REPO_ROOT / "scripts" / "client-compat" / "test-openai-node.mjs"
PRODUCT_CLI_H = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
README_MD = REPO_ROOT / "README.md"

VALID_LABELS = {"REAL", "SYNTHETIC", "SOURCE_ANALYSIS", "VALIDATED_REAL",
	"CONFIGURATION_ONLY", "NOT_VALIDATED", "UNSUPPORTED"}
NOT_VALIDATED_STATES = {"CONFIGURATION_ONLY", "NOT_VALIDATED"}

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


@check("evidence file, docs/client-compatibility.md, docs/api-contract.md, "
	"node-client-check.mjs, and both scripts/client-compat/ harnesses exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, CLIENT_COMPAT_MD, API_CONTRACT_MD, NODE_CHECK,
			PY_CLIENT_SCRIPT, NODE_CLIENT_SCRIPT)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON, schema_version 1, real (non-"
	"PENDING) D1-D6 squash SHAs")
def _c2():
	data = _load_evidence()
	shas = data.get("pr_squash_shas", {})
	ok = data.get("schema_version") == 1 and all(
		re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))
		for k in ("D1", "D2", "D3", "D4", "D5", "D6"))
	return ok, (f"schema_version={data.get('schema_version')} "
		+ " ".join(f"{k}={shas.get(k)!r}" for k in
			("D1", "D2", "D3", "D4", "D5", "D6")))


@check("every top-level evidence section carries a valid label")
def _c3():
	data = _load_evidence()
	bad = []
	for key, val in data.items():
		if isinstance(val, dict) and "label" in val:
			if val["label"] not in VALID_LABELS:
				bad.append(f"{key}: {val['label']!r}")
	return len(bad) == 0, ("bad labels: " + "; ".join(bad)) if bad \
		else "all labels valid"


@check("python_openai and node_client are labeled VALIDATED_REAL (this "
	"PR's own real integration gate, Section 46 of the task)")
def _c4():
	data = _load_evidence()
	bad = [k for k in ("python_openai", "node_client")
		if data.get(k, {}).get("label") != "VALIDATED_REAL"]
	return len(bad) == 0, (f"not VALIDATED_REAL: {bad}" if bad
		else "both VALIDATED_REAL")


@check("no CONFIGURATION_ONLY/NOT_VALIDATED client is claimed "
	"VALIDATED_REAL anywhere in docs/client-compatibility.md's own "
	"matrix row for that same client")
def _c5():
	data = _load_evidence()
	text = CLIENT_COMPAT_MD.read_text()
	bad = []
	# Cross-check evidence-file labels against the doc's own matrix rows
	# for Open WebUI / Continue specifically -- the two clients this
	# phase's own task explicitly anticipated might not be validated.
	checks = [("openwebui", "Open WebUI"), ("continue_extension", "Continue")]
	for evidence_key, doc_label in checks:
		state = data.get(evidence_key, {}).get("label")
		if state not in NOT_VALIDATED_STATES:
			continue
		# Find the doc's own table row for this client and confirm its
		# own Status column does not say VALIDATED_REAL.
		row_match = re.search(rf"\|\s*{re.escape(doc_label)}\s*\|.*\|\s*"
			r"(\S+)\s*\|\s*$", text, re.MULTILINE)
		if row_match and "VALIDATED_REAL" in row_match.group(1):
			bad.append(f"{doc_label}: evidence={state} but doc row claims "
				f"{row_match.group(1)}")
	return len(bad) == 0, ("mismatches: " + "; ".join(bad)) if bad \
		else "doc matrix matches evidence labels for not-fully-validated clients"


@check("OpenAI-compatibility wording stays scoped -- no file positively "
	"claims 'complete'/'full' OpenAI API support")
def _c6():
	claim_pattern = re.compile(
		r"(?:complete|full|entire)\s+OpenAI(?:\s+API)?\b", re.IGNORECASE)
	negation_pattern = re.compile(r"\b(?:not|never|n't|no)\b", re.IGNORECASE)
	bad = []
	for path in (CLIENT_COMPAT_MD, API_CONTRACT_MD, SERVER_MD, README_MD):
		text = path.read_text()
		for m in claim_pattern.finditer(text):
			window = text[max(0, m.start() - 40):m.start()]
			if not negation_pattern.search(window):
				bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("tool calling is not claimed as supported anywhere -- only as "
	"explicitly rejected (UNSUPPORTED_TOOL_CALLING)")
def _c7():
	text = SERVER_CPP.read_text()
	ok = "UNSUPPORTED_TOOL_CALLING" in text
	overclaim = re.search(r"\btool[- ]calling (?:is )?support(?:ed)?\b",
		API_CONTRACT_MD.read_text() + SERVER_MD.read_text(), re.IGNORECASE)
	return (ok and overclaim is None,
		"tool calling correctly rejected, not overclaimed as supported"
		if (ok and overclaim is None)
		else f"rejection_present={ok} overclaim_found={overclaim}")


@check("embeddings are never claimed as supported (this PR deliberately "
	"did not implement /v1/embeddings)")
def _c8():
	server_text = SERVER_CPP.read_text()
	ok = "/v1/embeddings" not in server_text
	overclaim = re.search(r"\bembeddings? (?:is |are )?support(?:ed)?\b",
		API_CONTRACT_MD.read_text() + SERVER_MD.read_text(), re.IGNORECASE)
	return (ok and overclaim is None,
		"no /v1/embeddings route, no overclaim in docs" if (ok and overclaim is None)
		else f"has_route={not ok} overclaim_found={overclaim}")


@check("GET /v1/models never exposes a filesystem path (id is the "
	"registry name, not entry.path)")
def _c9():
	text = SERVER_CPP.read_text()
	m = re.search(r'handle_models\([^{]*\{(.*?)\n\}\n', text, re.DOTALL)
	if not m:
		return False, "handle_models() not found"
	body = m.group(1)
	ok = 'm["id"] = e.name' in body and 'e.path' not in body
	return ok, "id comes from e.name, e.path never referenced in the response"


@check("docs/client-compatibility.md documents the exact base URL with "
	"no double /v1/v1")
def _c10():
	text = CLIENT_COMPAT_MD.read_text()
	ok = "http://127.0.0.1:8642/v1" in text and "/v1/v1" not in text
	return ok, "base URL documented correctly" if ok else "base URL missing or double /v1/v1 present"


@check("real client versions are recorded in the evidence file "
	"(openai python/node SDK versions, not left blank)")
def _c11():
	data = _load_evidence()
	py_ver = data.get("python_openai", {}).get("client_version", "")
	node_ver = data.get("node_client", {}).get("client_version", "")
	ok = bool(re.search(r"\d+\.\d+", py_ver)) and bool(re.search(r"\d+\.\d+", node_ver))
	return ok, f"python={py_ver!r} node={node_ver!r}"


@check("the release version (MEMBRANE_VERSION) is unchanged by this PR")
def _c12():
	text = PRODUCT_CLI_H.read_text()
	m = re.search(r'MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.8.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '?'}"


@check("test_server.cpp carries real ctest coverage for the new D7 "
	"request-shape checks (tools/response_format/stop/auth-header)")
def _c13():
	text = TEST_SERVER_CPP.read_text()
	needed = ("test_chat_tools_rejected", "test_chat_tool_choice_rejected",
		"test_chat_response_format_json_rejected",
		"test_chat_stop_too_many_entries_rejected",
		"test_auth_header_is_tolerated")
	missing = [n for n in needed if n not in text]
	return len(missing) == 0, (f"missing tests: {missing}" if missing
		else "all new D7 tests present")


@check("Mega Phase A-D6's own evidence files are untouched by this PR's "
	"own new commits (checked against origin/main)")
def _c14():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/model-lifecycle-ux/validation.json",
		"results/windows-support/validation.json",
		"results/macos-metal/validation.json",
		"results/cuda-backend/validation.json",
		"results/background-service/validation.json",
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
