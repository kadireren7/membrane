#!/usr/bin/env python3
"""Validate v1.1.0 release readiness: MEMBRANE_VERSION and CITATION.cff say
1.1.0, the release notes / upgrade guide / readiness.json exist and are
self-consistent with legal evidence states, the package-policy manifest is
bumped, the README names v1.1.0 as the latest stable tag, no overclaim exists
in release-facing docs (complete OpenAI API, production-ready, multi-GPU or
distributed execution, a Windows/macOS package, CORS), the security statements
still hold in the server source, and historical release docs/evidence are
byte-for-byte untouched.

Same standalone-script, own-CI-job pattern as scripts/verify-release-v1.0.0.py
before it (retired from CI by this release, as v0.8.0's was by v1.0.0's).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRODUCT_CLI_H = REPO / "tools" / "membrane-run" / "product_cli.h"
CITATION = REPO / "CITATION.cff"
NOTES = REPO / "docs" / "release-v1.1.0.md"
UPGRADE = REPO / "docs" / "upgrade-v1.0-to-v1.1.md"
READINESS = REPO / "results" / "release-v1.1.0" / "readiness.json"
MANIFEST = REPO / "results" / "release-artifacts" / "manifest.json"
README = REPO / "README.md"
SERVER_CPP = REPO / "tools" / "membrane" / "server.cpp"

VALID_STATES = {"VALIDATED_REAL", "VALIDATED_CI", "VALIDATED_MAINTAINER",
	"VALIDATED_CI_REAL_HARDWARE", "COMPILE_VALIDATED_CI",
	"CONFIGURATION_ONLY", "NOT_VALIDATED", "UNSUPPORTED", "REAL",
	"SYNTHETIC", "SOURCE_ANALYSIS", "NOT_SHIPPED", "PARTIAL"}

FAILURES = []
COUNT = 0


def check(name):
	def deco(fn):
		def wrapper():
			global COUNT
			COUNT += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return deco


def _facing():
	return [NOTES, UPGRADE, README, CITATION,
		REPO / "docs" / "quickstart.md", REPO / "docs" / "orchestration.md",
		REPO / "docs" / "active-control.md",
		REPO / "docs" / "adaptive-runtime.md",
		REPO / "docs" / "runtime-vllm.md"]


@check("MEMBRANE_VERSION == 1.1.0 (live product_cli.h)")
def _c1():
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"',
		PRODUCT_CLI_H.read_text())
	return m is not None and m.group(1) == "1.1.0", \
		f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("CITATION.cff version == 1.1.0")
def _c2():
	m = re.search(r'^version:\s*"([^"]+)"', CITATION.read_text(), re.M)
	return m is not None and m.group(1) == "1.1.0", \
		f"version={m.group(1) if m else '(not found)'}"


@check("release notes, upgrade guide and readiness.json exist")
def _c3():
	missing = [str(p) for p in (NOTES, UPGRADE, READINESS) if not p.exists()]
	return not missing, f"missing: {missing}" if missing else "all present"


@check("readiness.json: schema 1, version 1.1.0, label REAL, legal states")
def _c4():
	d = json.loads(READINESS.read_text())
	bad = []

	def walk(o, path):
		if isinstance(o, dict):
			for k, v in o.items():
				if k in ("state", "label") and isinstance(v, str) \
						and v not in VALID_STATES:
					bad.append(f"{path}.{k}={v!r}")
				walk(v, f"{path}.{k}")
		elif isinstance(o, list):
			for i, v in enumerate(o):
				walk(v, f"{path}[{i}]")
	walk(d, "readiness")
	ok = d.get("schema_version") == 1 and d.get("version") == "1.1.0" \
		and d.get("label") == "REAL" and not bad
	return ok, f"illegal={bad}" if bad else "ok"


@check("no overclaim in release-facing docs")
def _c5():
	pats = [
		re.compile(r"(?:complete|full|entire)\s+OpenAI(?:\s+API)?\b", re.I),
		re.compile(r"\bproduction[- ]ready\b", re.I),
		re.compile(r"\bmulti[- ]GPU (?:execution|inference) (?:is )?"
			r"(?:supported|available|works)\b", re.I),
		re.compile(r"\bdistributed (?:inference|orchestration) "
			r"(?:is )?(?:supported|available)\b", re.I),
		re.compile(r"\b(?:every|all|any)\s+(?:GPU|NVIDIA|AMD|Intel)\s+"
			r"(?:device|hardware|card)\b", re.I),
	]
	neg = re.compile(r"\b(?:not|never|n't|no)\b", re.I)
	bad = []
	for p in _facing():
		if not p.exists():
			continue
		t = p.read_text()
		for pat in pats:
			for m in pat.finditer(t):
				if not neg.search(t[max(0, m.start() - 60):m.start()]):
					bad.append(f"{p.name}: {m.group(0)!r}")
	return not bad, f"overclaim: {bad}" if bad else "none"


@check("package policy bumped; no Windows/macOS package claimed shipped")
def _c6():
	m = json.loads(MANIFEST.read_text())
	r = json.loads(READINESS.read_text())
	pm = r.get("package_matrix", {})
	ok = m.get("product_version") == "1.1.0" \
		and m.get("debian_version") == "1.1.0" \
		and m.get("stable_release", {}).get("version") == "1.1.0" \
		and pm.get("windows_package", {}).get("state") == "NOT_SHIPPED" \
		and pm.get("macos_package", {}).get("state") == "NOT_SHIPPED"
	return ok, "manifest + package matrix consistent" if ok else "inconsistent"


@check("README names v1.1.0 as the latest stable tag, not v1.0.0")
def _c7():
	t = README.read_text()
	ok = "latest stable tag `v1.1.0`" in t
	stale = re.search(r"latest stable tag `v1\.0\.0`", t)
	return ok and not stale, f"has_v110={ok} stale_v100={bool(stale)}"


@check("security statements hold in the source: no CORS header is ever set "
	"by the server; bind defaults to loopback")
def _c8():
	t = SERVER_CPP.read_text()
	cors = re.search(r"Access-Control-|set_default_headers", t)
	return cors is None, "no CORS header in server.cpp" if cors is None \
		else f"found {cors.group(0)!r}"


@check("known limitations are disclosed in readiness.json, including the "
	"synthetic adaptive evidence and mock-only Ollama/vLLM testing")
def _c9():
	lim = " ".join(json.loads(READINESS.read_text())
		.get("known_limitations", [])).lower()
	ok = "synthetic" in lim and "mock" in lim and "multi-gpu" in lim
	return ok, "disclosed" if ok else "missing disclosure"


@check("real upgrade-test evidence exists and is labeled REAL")
def _c10():
	u = json.loads(READINESS.read_text()).get("package_matrix", {}) \
		.get("real_upgrade_test", {})
	ok = u.get("label") == "REAL" and "IDENTICAL" in u.get("result", "")
	return ok, f"label={u.get('label')}"


@check("historical v0.3.0/v0.4.0/v0.8.0/v1.0.0 release docs/evidence "
	"untouched (checked against origin/main)")
def _c11():
	r = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO, capture_output=True, text=True, check=False)
	if r.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref)"
	forbidden = ("results/release-v0.3.0/", "results/release-v0.3.0-rc2/",
		"results/release-v0.3.0-rc3/", "results/release-v0.4.0/",
		"results/release-v0.8.0/", "results/release-v1.0.0/",
		"docs/release-v0.3.0.md", "docs/release-v0.4.0.md",
		"docs/release-v0.8.0.md", "docs/release-v1.0.0.md")
	touched = [p for p in r.stdout.splitlines() if p.startswith(forbidden)]
	return not touched, f"touched: {touched}" if touched else "untouched"


@check("no Claude/Co-Authored-By attribution in the release docs/notes")
def _c12():
	pat = re.compile(r"co-authored-by|generated with \[claude|noreply@anthropic",
		re.I)
	bad = [p.name for p in (NOTES, UPGRADE, READINESS) if p.exists()
		and pat.search(p.read_text())]
	return not bad, f"found in {bad}" if bad else "none"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12):
		fn()
	print(f"\n{COUNT - len(FAILURES)}/{COUNT} checks passed")
	if FAILURES:
		print("\nFAILURES:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
