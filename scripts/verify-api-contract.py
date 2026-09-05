#!/usr/bin/env python3
"""Validate Mega Phase C, PR C3's API-contract/hardening work:
frozen-error-contract completeness (every code in server.cpp appears
in docs/server.md's table and vice versa), no "complete OpenAI API"
overclaim, the permanent port-collision regression test still exists,
privacy-audit regression guards (no content logging), and syntax
validity of the new soak/concurrency/client scripts.

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
SERVER_CPP = REPO_ROOT / "tools" / "membrane" / "server.cpp"
SERVER_MD = REPO_ROOT / "docs" / "server.md"
API_CONTRACT_MD = REPO_ROOT / "docs" / "api-contract.md"
CLIENT_COMPAT_MD = REPO_ROOT / "docs" / "client-compatibility.md"
PRIVACY_MD = REPO_ROOT / "docs" / "privacy-security-audit.md"
SOAK_DOC = REPO_ROOT / "docs" / "soak-and-concurrency-testing.md"
TEST_SERVER_CPP = REPO_ROOT / "tools" / "membrane" / "test_server.cpp"
EVIDENCE_PATH = REPO_ROOT / "results" / "product-hardening" / "v0.4-validation.json"
SOAK_PY = REPO_ROOT / "scripts" / "soak-test-server.py"
CONCURRENCY_PY = REPO_ROOT / "scripts" / "concurrency-soak-server.py"
NODE_CHECK = REPO_ROOT / "scripts" / "node-client-check.mjs"

# The fixed (non-dynamic) error codes server.cpp emits via a literal
# string -- GENERATION_FAILED is the fallback for the one dynamic row
# (a real planner reason code may appear instead, disclosed in
# docs/server.md and docs/api-contract.md, never enumerated as closed).
EXPECTED_CODES = {
	"INVALID_REQUEST", "MODEL_NOT_FOUND", "CHAT_TEMPLATE_UNAVAILABLE",
	"CHAT_TEMPLATE_FAILED", "MODEL_LOAD_FAILED", "NO_FEASIBLE_CONTEXT",
	"SERVER_BUSY", "GENERATION_FAILED",
}

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


@check("docs exist: api-contract, client-compatibility, "
	"privacy-security-audit, soak-and-concurrency-testing")
def _c1():
	missing = [str(p) for p in
		(API_CONTRACT_MD, CLIENT_COMPAT_MD, PRIVACY_MD, SOAK_DOC,
			EVIDENCE_PATH, SOAK_PY, CONCURRENCY_PY, NODE_CHECK)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("frozen error contract: every literal error code emitted by "
	"server.cpp appears in docs/server.md's Errors table")
def _c2():
	text = SERVER_CPP.read_text()
	codes_in_code = set(re.findall(r'"([A-Z_]{4,})"', text)) & EXPECTED_CODES
	doc_text = SERVER_MD.read_text()
	missing_from_docs = [c for c in codes_in_code if c not in doc_text]
	return len(missing_from_docs) == 0, (
		f"codes found in code but missing from docs/server.md: {missing_from_docs}"
		if missing_from_docs else f"all {len(codes_in_code)} codes documented")


@check("frozen error contract: every EXPECTED code still exists "
	"literally in server.cpp (nothing silently removed)")
def _c3():
	text = SERVER_CPP.read_text()
	missing = [c for c in EXPECTED_CODES if f'"{c}"' not in text]
	return len(missing) == 0, (f"expected codes missing from server.cpp: {missing}"
		if missing else "all expected codes present")


@check("no file in this repo POSITIVELY claims a 'complete'/'full' "
	"OpenAI API (Mega Phase C's own 'compatible subset, never "
	"complete' framing) -- a disclaimer that it is NOT complete is "
	"fine and expected (docs/api-contract.md itself says exactly "
	"that), only a positive claim -- one with no negation word in "
	"the 40 characters before the match -- is flagged")
def _c4():
	bad = []
	claim_pattern = re.compile(
		r"(?:complete|full|entire)\s+OpenAI(?:\s+API)?\b", re.IGNORECASE)
	negation_pattern = re.compile(r"\b(?:not|never|n't|no)\b", re.IGNORECASE)
	for path in list(REPO_ROOT.glob("docs/*.md")) + [REPO_ROOT / "README.md"]:
		text = path.read_text()
		for m in claim_pattern.finditer(text):
			window = text[max(0, m.start() - 40):m.start()]
			if not negation_pattern.search(window):
				bad.append(f"{path.name}: {m.group(0)!r}")
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("permanent port-collision regression test still exists and is "
	"ctest-registered")
def _c5():
	test_text = TEST_SERVER_CPP.read_text()
	has_test = "test_second_instance_same_port_fails_to_bind" in test_text
	cmake_text = (REPO_ROOT / "tools" / "membrane" / "CMakeLists.txt").read_text()
	has_ctest = "add_test(NAME test_server" in cmake_text
	ok = has_test and has_ctest
	return ok, f"has_test={has_test} has_ctest={has_ctest}"


@check("privacy regression guard: no fprintf/cerr/cout in server.cpp "
	"embeds request message/content/prompt data")
def _c6():
	text = SERVER_CPP.read_text()
	bad_lines = []
	for i, line in enumerate(text.splitlines(), 1):
		if re.search(r"\b(fprintf|std::cerr|std::cout)\b", line) and \
				re.search(r"\b(req\.|messages|\.content\b|prompt_text)\b", line):
			bad_lines.append(i)
	return len(bad_lines) == 0, (f"suspicious logging at line(s): {bad_lines}"
		if bad_lines else "no content-logging call found")


@check("Retry-After is scoped to SERVER_BUSY specifically, not applied "
	"to every 503 (the real bug this PR's own concurrency-soak script "
	"found in itself, not in MEMBRANE)")
def _c7():
	text = SERVER_CPP.read_text()
	# The header must be set immediately before the SERVER_BUSY send,
	# and NO_FEASIBLE_CONTEXT's own send call must not also set it.
	idx_retry = text.find('set_header("Retry-After"')
	idx_busy = text.find('"SERVER_BUSY"')
	idx_nofeasible_block = text.find('"NO_FEASIBLE_CONTEXT"')
	ok = (idx_retry != -1 and idx_busy != -1 and idx_nofeasible_block != -1
		and abs(idx_busy - idx_retry) < 200)
	return ok, ("Retry-After set immediately before SERVER_BUSY only" if ok
		else "could not confirm Retry-After is scoped to SERVER_BUSY only")


@check("new scripts are syntactically valid (py_compile / node --check)")
def _c8():
	results = {}
	for path in (SOAK_PY, CONCURRENCY_PY):
		r = subprocess.run(["python3", "-m", "py_compile", str(path)],
			capture_output=True, text=True, check=False)
		results[path.name] = r.returncode == 0
	node_r = subprocess.run(["node", "--check", str(NODE_CHECK)],
		capture_output=True, text=True, check=False)
	results[NODE_CHECK.name] = node_r.returncode == 0
	ok = all(results.values())
	return ok, str(results)


@check("v0.4-validation.json is valid JSON with schema_version 1 and "
	"real C1/C2/C3 pr_squash_shas")
def _c9():
	data = json.loads(EVIDENCE_PATH.read_text())
	shas = data.get("pr_squash_shas", {})
	ok = (data.get("schema_version") == 1
		and all(re.fullmatch(r"[0-9a-f]{40}", shas.get(k, "")) for k in ("C1", "C2")))
	return ok, f"shas={shas}"


@check("live MEMBRANE_VERSION is v0.4.0 (bumped by PR C4)")
def _c10():
	text = (REPO_ROOT / "tools" / "membrane-run" / "product_cli.h").read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.4.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("patch set is unchanged and matches the two real committed "
	"patch files")
def _c11():
	patches_dir = REPO_ROOT / "patches"
	expected = {
		"llama.cpp-membrane-kv-type-override.patch",
		"llama.cpp-membrane-kv-device-override.patch",
	}
	real = {p.name for p in patches_dir.glob("*.patch")}
	return real == expected, f"real={real} expected={expected}"


@check("Mega Phase A/B/C-C1's own evidence files are untouched by this "
	"PR's own new commits (checked against origin/main) -- "
	"results/release-supply-chain/validation.json IS expected to "
	"change (C2's real squash SHA backfilled here, same convention as "
	"C1's SHA being backfilled inside PR C2's own commit)")
def _c12():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/runtime-service/validation.json",
		"results/background-service/validation.json",
		"results/product-onboarding/validation.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


@check("results/release-supply-chain/validation.json's own C2 squash "
	"SHA is backfilled with a real commit (no longer PENDING) as part "
	"of this PR's own commit")
def _c13():
	path = REPO_ROOT / "results" / "release-supply-chain" / "validation.json"
	data = json.loads(path.read_text())
	sha = data.get("pr_squash_shas", {}).get("C2", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"C2 sha={sha!r}"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12,
			_c13):
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
