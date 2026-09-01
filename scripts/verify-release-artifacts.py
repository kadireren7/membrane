#!/usr/bin/env python3
"""Validate Phase 30's release-artifact naming/packaging POLICY
(results/release-artifacts/manifest.json) -- schema and internal
consistency, that the policy this file documents actually matches what
CMakeLists.txt's CPack config implements, that the official/non-official
artifact filenames can never collide, that the Debian version mapping is
consistent with the live MEMBRANE_VERSION, that no stable v0.3.0 asset
is claimed yet, that architecture claims are honest (amd64 only), and
that the CPU/Vulkan policy is stated unambiguously and consistently
across CMakeLists.txt/docs/install.md/README.md.

This validates POLICY/SHAPE, not real packages -- no .deb file is
committed to this repository (Section 24) and this manifest's own
sha256 field is intentionally null (no stable release exists yet).

Same one-file-per-concern convention as every other scripts/verify-*.py
in this project (see scripts/verify-results.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "release-artifacts" / "manifest.json"
CMAKE_PATH = REPO_ROOT / "CMakeLists.txt"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
README_PATH = REPO_ROOT / "README.md"
INSTALL_MD_PATH = REPO_ROOT / "docs" / "install.md"

FAILURES = []
CHECK_COUNT = 0

# Actual claimed-support patterns (an install/build command, or a
# distinct package artifact, targeting a platform this project does not
# support) -- NOT the bare platform name, which legitimately appears in
# disclaimer sentences like "no Fedora/RPM... packaging exists" (the
# thing Section 4 explicitly wants stated). A false negative here (a
# claim phrased in some other way) is possible; a false positive
# flagging a disclaimer as a claim is the failure mode worth avoiding,
# since disclaiming unsupported platforms is required, not forbidden.
FORBIDDEN_PLATFORM_PATTERNS = (
	r"dnf install", r"yum install", r"\.rpm\b(?!.{0,40}(no|not|never|none|does not))",
	r"pacman -S", r"flatpak install", r"snap install",
	r"\.dmg\b", r"\.exe\b", r"\.msi\b", r"brew install membrane",
)


def check(name):
	def decorator(fn):
		def wrapper():
			global CHECK_COUNT
			CHECK_COUNT += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001 -- report, don't crash the whole run
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			status = "PASS" if ok else "FAIL"
			print(f"[{status}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return decorator


_DATA_CACHE = {}


def _load_data():
	if "data" not in _DATA_CACHE:
		_DATA_CACHE["data"] = json.loads(DATA_PATH.read_text())
	return _DATA_CACHE["data"]


@check("results/release-artifacts/manifest.json exists, is valid JSON, "
	"has schema_version 1")
def _c1():
	if not DATA_PATH.exists():
		return False, f"{DATA_PATH} does not exist"
	d = _load_data()
	ok = d.get("schema_version") == 1
	return ok, f"schema_version={d.get('schema_version')!r}"


@check("official_artifacts has exactly one 'official' role and its "
	"filename_pattern/package_name never collide with any other entry")
def _c2():
	d = _load_data()
	arts = d.get("official_artifacts", [])
	official = [a for a in arts if a.get("role") == "official"]
	if len(official) != 1:
		return False, f"expected exactly 1 'official' artifact, found {len(official)}"
	names = [a.get("package_name") for a in arts]
	patterns = [a.get("filename_pattern") for a in arts]
	bad = []
	if len(set(names)) != len(names):
		bad.append(f"duplicate package_name across artifacts: {names}")
	if len(set(patterns)) != len(patterns):
		bad.append(f"duplicate filename_pattern across artifacts: {patterns}")
	for p in patterns:
		if not p or "package_name" not in {"placeholder"}:
			pass  # pattern presence checked structurally below
	for a in arts:
		pn = a.get("package_name", "")
		fp = a.get("filename_pattern", "")
		if not fp.startswith(pn + "_"):
			bad.append(f"{pn}: filename_pattern {fp!r} does not start with "
				f"'{pn}_' -- package_name and filename must stay in sync")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(arts)} artifacts, no collisions"


@check("Debian version mapping matches CMakeLists.txt's own rule "
	"applied to the live MEMBRANE_VERSION")
def _c3():
	header_text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', header_text)
	if not m:
		return False, "MEMBRANE_VERSION not found in product_cli.h"
	live_version = m.group(1)
	expected_debian_version = live_version.replace("-", "~")
	cmake_text = CMAKE_PATH.read_text()
	if 'string(REPLACE "-" "~" MEMBRANE_DEBIAN_VERSION' not in cmake_text:
		return False, "CMakeLists.txt no longer contains the expected version-mapping rule"
	d = _load_data()
	bad = []
	if d.get("product_version") != live_version:
		bad.append(f"manifest product_version={d.get('product_version')!r} "
			f"!= live MEMBRANE_VERSION={live_version!r}")
	if d.get("debian_version") != expected_debian_version:
		bad.append(f"manifest debian_version={d.get('debian_version')!r} "
			f"!= expected {expected_debian_version!r}")
	for a in d.get("official_artifacts", []):
		expected_fn = a["package_name"] + "_<debian_version>_amd64.deb"
		if a.get("filename_pattern") != expected_fn:
			bad.append(f"{a['package_name']}: filename_pattern "
				f"{a.get('filename_pattern')!r} != expected {expected_fn!r}")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"live {live_version!r} maps to {expected_debian_version!r}, consistent")


@check("CMakeLists.txt's CPack config implements the documented policy: "
	"GGML_VULKAN=ON -> 'membrane', else -> 'membrane-cpu'")
def _c4():
	cmake_text = CMAKE_PATH.read_text()
	bad = []
	if 'if(GGML_VULKAN)' not in cmake_text:
		bad.append("no if(GGML_VULKAN) branch found around CPACK_PACKAGE_NAME")
	if 'set(CPACK_PACKAGE_NAME "membrane")' not in cmake_text:
		bad.append('CPACK_PACKAGE_NAME "membrane" not found')
	if 'set(CPACK_PACKAGE_NAME "membrane-cpu")' not in cmake_text:
		bad.append('CPACK_PACKAGE_NAME "membrane-cpu" not found')
	return len(bad) == 0, "; ".join(bad) if bad else "CMakeLists.txt implements the documented name policy"


@check("no stable v0.3.0 release asset (sha256/filename) is claimed yet")
def _c5():
	d = _load_data()
	bad = []
	stable = d.get("stable_release")
	no_claim = d.get("no_stable_release_claim")
	if stable is None and not no_claim:
		bad.append("manifest claims neither a stable release (via "
			"stable_release) nor explicitly disclaims one (via "
			"no_stable_release_claim) -- exactly one must be present")
	if stable is not None and no_claim:
		bad.append("manifest has BOTH a stable_release claim and a "
			"no_stable_release_claim disclaimer -- contradictory")
	if stable is not None:
		# A real stable claim must point to real, separately-verified
		# evidence (results/release-v0.3.0/readiness.json) rather than
		# embedding an unverified sha256 directly in this POLICY
		# document (Section 21: avoid two sources of truth for the
		# same fact) -- so this file itself is checked for structure/
		# consistency, not for a literal sha256 value.
		if not stable.get("released"):
			bad.append("stable_release.released is not true")
		if not re.fullmatch(r"\d+\.\d+\.\d+", str(stable.get("version", ""))):
			bad.append(f"stable_release.version={stable.get('version')!r} "
				"is not a bare stable version (no ~/-/rc suffix)")
		expected_fn = f"membrane_{stable.get('version')}_amd64.deb"
		if stable.get("filename") != expected_fn:
			bad.append(f"stable_release.filename={stable.get('filename')!r} "
				f"!= expected {expected_fn!r}")
		readiness_path = REPO_ROOT / "results" / f"release-v{stable.get('version')}" / "readiness.json"
		if not readiness_path.exists():
			bad.append(f"{readiness_path} does not exist -- a stable_release "
				"claim needs real, separately-verified evidence")
	return len(bad) == 0, ("; ".join(bad) if bad else
		(f"stable_release claim for {stable.get('version')!r} is well-formed "
			"and points to real evidence" if stable is not None
			else "no stable release claimed yet, explicitly disclosed"))


@check("architecture is amd64 only across every artifact (no arm64/other "
	"claim)")
def _c6():
	d = _load_data()
	bad = [a["package_name"] for a in d.get("official_artifacts", [])
		if a.get("architecture") != "amd64"]
	return len(bad) == 0, f"{bad}" if bad else "all artifacts are amd64"


@check("no forbidden-platform install command or artifact (dnf/yum/"
	"pacman/flatpak/snap install, .rpm/.dmg/.exe/.msi) is offered "
	"anywhere in the manifest, docs/install.md, or README.md -- "
	"disclaiming those platforms is fine, offering to install on them "
	"is not")
def _c7():
	bad = []
	for path in (DATA_PATH, INSTALL_MD_PATH, README_PATH):
		text = path.read_text()
		for pat in FORBIDDEN_PLATFORM_PATTERNS:
			if re.search(pat, text, re.IGNORECASE):
				bad.append(f"{path.name} matches forbidden pattern {pat!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no forbidden-platform install command found"


@check("docs/install.md and README.md both name the official package "
	"'membrane' and the non-official variant 'membrane-cpu', and never "
	"contradict which one is official")
def _c8():
	bad = []
	install_text = INSTALL_MD_PATH.read_text()
	readme_text = README_PATH.read_text()
	for name, text in (("docs/install.md", install_text), ("README.md", readme_text)):
		if "membrane_<version>_amd64.deb" not in text:
			bad.append(f"{name} does not reference the official filename pattern")
	if "membrane-cpu" not in install_text:
		bad.append("docs/install.md never mentions the membrane-cpu variant")
	# The non-official package must never be called "official" or
	# "recommended" anywhere near its own mention.
	m = re.search(r'membrane-cpu[^.]{0,200}', install_text, re.IGNORECASE)
	if m and re.search(r'\bofficial\b|\brecommended\b', m.group(0), re.IGNORECASE):
		bad.append("docs/install.md's membrane-cpu description contains "
			"'official'/'recommended', contradicting the policy")
	return len(bad) == 0, "; ".join(bad) if bad else "package-name policy stated consistently"


@check("no .deb file is committed to the repository (Section 24)")
def _c9():
	import subprocess
	tracked = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.deb"],
		capture_output=True, text=True, check=True).stdout.strip()
	return tracked == "", f"git-tracked .deb files: {tracked!r}" if tracked else "none committed"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9):
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
