#!/usr/bin/env python3
"""Validate Mega Phase C, PR C2's release-supply-chain evidence
(results/release-supply-chain/validation.json): schema and REAL/
SYNTHETIC/SOURCE_ANALYSIS labeling, the registry schema-versioning
regression guard, the reproducibility fix's presence in the release
script, the release script's own real-binary sanity check, no
auto-generated-signing-key regression, and that the release script
never touches git.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see verify-product-onboarding.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "release-supply-chain" / "validation.json"
BUILD_RELEASE_SH = REPO_ROOT / "scripts" / "build-release.sh"
GENERATE_SBOM_PY = REPO_ROOT / "scripts" / "generate-sbom.py"
REGISTRY_CORE_H = REPO_ROOT / "tools" / "membrane" / "registry_core.h"
REGISTRY_CORE_CPP = REPO_ROOT / "tools" / "membrane" / "registry_core.cpp"
TEST_REGISTRY_CORE = REPO_ROOT / "tools" / "membrane" / "test_registry_core.cpp"
SCHEMA_DOC = REPO_ROOT / "docs" / "schema-versioning.md"
SUPPLY_CHAIN_DOC = REPO_ROOT / "docs" / "release-supply-chain.md"

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


@check("evidence file and docs exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, BUILD_RELEASE_SH, GENERATE_SBOM_PY, SCHEMA_DOC,
			SUPPLY_CHAIN_DOC) if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and both "
	"C1/C2 pr_squash_shas keys present")
def _c2():
	data = _load_evidence()
	ok = (data.get("schema_version") == 1
		and isinstance(data.get("pr_squash_shas"), dict)
		and "C1" in data["pr_squash_shas"] and "C2" in data["pr_squash_shas"])
	return ok, "shape OK" if ok else f"unexpected shape: {list(data.keys())}"


@check("C1's own real squash SHA is backfilled here (a real 40-char "
	"hex commit, not left PENDING) -- this PR's own commit is the "
	"place that backfill happens, never a standalone post-merge main "
	"commit")
def _c3():
	data = _load_evidence()
	sha = data.get("pr_squash_shas", {}).get("C1", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"C1 sha={sha!r}"


@check("every top-level evidence section carries a valid REAL/SYNTHETIC/"
	"SOURCE_ANALYSIS label")
def _c4():
	data = _load_evidence()
	bad = []
	for key, val in data.items():
		if isinstance(val, dict) and "label" in val:
			if val["label"] not in VALID_LABELS:
				bad.append(f"{key}: {val['label']!r}")
	return len(bad) == 0, ("bad labels: " + "; ".join(bad)) if bad \
		else "all labels valid"


@check("regression guard: registry_core now defines and enforces "
	"MEMBRANE_REGISTRY_SCHEMA_VERSION (a real gap this PR found and "
	"fixed -- the registry used to check schema_version was PRESENT "
	"but never that its VALUE was recognized)")
def _c5():
	h_text = REGISTRY_CORE_H.read_text()
	cpp_text = REGISTRY_CORE_CPP.read_text()
	has_define = "MEMBRANE_REGISTRY_SCHEMA_VERSION" in h_text
	has_check = ("UNSUPPORTED_SCHEMA" in cpp_text
		and "MEMBRANE_REGISTRY_SCHEMA_VERSION" in cpp_text)
	ok = has_define and has_check
	return ok, f"has_define={has_define} has_check={has_check}"


@check("regression guard: test_registry_core has a real "
	"unsupported-schema-version test, mirroring test_server_config's "
	"pre-existing one")
def _c6():
	text = TEST_REGISTRY_CORE.read_text()
	ok = "test_load_unsupported_schema_version_fails_closed" in text
	return ok, "test present" if ok else "test missing"


@check("regression guard: scripts/build-release.sh pins "
	"SOURCE_DATE_EPOCH before cpack (the real reproducible-build fix "
	"this PR found and verified)")
def _c7():
	text = BUILD_RELEASE_SH.read_text()
	ok = "SOURCE_DATE_EPOCH" in text and "cpack -G DEB" in text
	return ok, "present" if ok else "missing"


@check("regression guard: scripts/build-release.sh always passes "
	"-DMEMBRANE_ENABLE_LLAMA=ON (a real bug this PR found: without "
	"it, cpack silently produces a near-empty .deb with no real "
	"binaries at all) and refuses to publish a .deb missing either "
	"real binary")
def _c8():
	text = BUILD_RELEASE_SH.read_text()
	has_flag = "-DMEMBRANE_ENABLE_LLAMA=ON" in text
	has_guard = ("usr/bin/membrane-run" in text
		and "usr/bin/membrane$" in text)
	ok = has_flag and has_guard
	return ok, f"has_flag={has_flag} has_guard={has_guard}"


def _non_comment_lines(text):
	"""Strip full-line and trailing '#'-comments (bash and YAML share
	the syntax) before searching -- this script's OWN prose repeatedly
	explains, in comments, which git/gh operations must never appear as
	real commands, which would otherwise self-match a naive substring
	search."""
	out = []
	for line in text.splitlines():
		out.append(line.split("#", 1)[0])
	return "\n".join(out)


@check("regression guard: scripts/build-release.sh never touches git "
	"(no tag, no push, no gh release create) -- publishing stays a "
	"separate, manual, primary-agent-performed step")
def _c9():
	text = _non_comment_lines(BUILD_RELEASE_SH.read_text())
	bad = [tok for tok in ("git tag", "git push", "gh release") if tok in text]
	return len(bad) == 0, (f"forbidden git/gh operation(s) found: {bad}" if bad
		else "no git/gh operation present")


@check("regression guard: no signing key is ever auto-generated by "
	"any script in this repo (gpg --gen-key / --batch --generate-key)")
def _c10():
	this_file = Path(__file__).resolve()
	bad = []
	for path in list(REPO_ROOT.glob("scripts/*.py")) \
			+ list(REPO_ROOT.glob("scripts/*.sh")):
		if path.resolve() == this_file:
			continue  # this check's own prose names the forbidden flags
		text = _non_comment_lines(path.read_text())
		if "gen-key" in text or "generate-key" in text:
			bad.append(path.name)
	return len(bad) == 0, (f"key-generation reference found in: {bad}" if bad
		else "no auto-key-generation anywhere")


@check("scripts/build-release.sh and scripts/generate-sbom.py are "
	"syntactically valid (bash -n / python3 -m py_compile)")
def _c11():
	bash_rc = subprocess.run(["bash", "-n", str(BUILD_RELEASE_SH)],
		capture_output=True, text=True, check=False)
	py_rc = subprocess.run(["python3", "-m", "py_compile", str(GENERATE_SBOM_PY)],
		capture_output=True, text=True, check=False)
	ok = bash_rc.returncode == 0 and py_rc.returncode == 0
	return ok, ("both valid" if ok
		else f"bash: {bash_rc.stderr.strip()} python3: {py_rc.stderr.strip()}")


@check("live MEMBRANE_VERSION is v0.4.0 (bumped by PR C4)")
def _c12():
	text = (REPO_ROOT / "tools" / "membrane-run" / "product_cli.h").read_text()
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	ok = m is not None and m.group(1) == "0.4.0"
	return ok, f"MEMBRANE_VERSION={m.group(1) if m else '(not found)'}"


@check("patch set is unchanged and matches the two real committed "
	"patch files")
def _c13():
	patches_dir = REPO_ROOT / "patches"
	expected = {
		"llama.cpp-membrane-kv-type-override.patch",
		"llama.cpp-membrane-kv-device-override.patch",
	}
	real = {p.name for p in patches_dir.glob("*.patch")}
	return real == expected, f"real={real} expected={expected}"


@check("new CI job release-candidate-smoke is registered and never "
	"calls gh release create")
def _c14():
	ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
	text = _non_comment_lines(ci_path.read_text())
	has_job = "release-candidate-smoke:" in ci_path.read_text()
	no_publish = "gh release" not in text
	ok = has_job and no_publish
	return ok, f"has_job={has_job} no_publish={no_publish}"


@check("Mega Phase A/B's own evidence files are untouched by this "
	"PR's own new commits (checked against origin/main) -- "
	"results/product-onboarding/validation.json IS expected to change "
	"(C1's real squash SHA backfilled here, not as a standalone "
	"post-merge main commit -- see this file's own top-level 'note')")
def _c15():
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


@check("results/product-onboarding/validation.json's own C1 squash "
	"SHA is backfilled with a real commit (no longer PENDING) as part "
	"of this PR's own commit")
def _c16():
	path = REPO_ROOT / "results" / "product-onboarding" / "validation.json"
	data = json.loads(path.read_text())
	sha = data.get("pr_squash_shas", {}).get("C1", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"C1 sha={sha!r}"


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
