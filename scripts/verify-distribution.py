#!/usr/bin/env python3
"""Validate Phase 29's distribution/.deb-packaging evidence.

Checks results/distribution/validation.json's schema and internal
consistency (every package entry has real sha256/contents/dependency
fields, install/remove results are present and clean, no forbidden
payload -- test/benchmark/model/build-tree content -- ever appears in a
package's declared contents), that the product-version -> Debian-
package-version mapping this file records actually matches what
CMakeLists.txt's own mapping rule would produce from the live
MEMBRANE_VERSION, that the package name/architecture claims are honest,
and that docs/README actually reference the real package name and a
correctly-mapped example version. This validates the COMMITTED evidence
file (captured locally against real `.deb` builds -- see
docs/install.md); no `.deb` file itself is committed (Section 24: git
tracks metadata/scripts, not binary packages), so this does not
re-verify a live file's sha256 -- it checks the evidence file's own
internal consistency and shape.

Same one-file-per-concern convention as every other scripts/verify-*.py
in this project (see scripts/verify-results.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "distribution" / "validation.json"
CMAKE_PATH = REPO_ROOT / "CMakeLists.txt"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
README_PATH = REPO_ROOT / "README.md"
INSTALL_MD_PATH = REPO_ROOT / "docs" / "install.md"

FAILURES = []
CHECK_COUNT = 0

REQUIRED_PACKAGE_FIELDS = (
	"id", "filename", "build_command", "product_version", "package_version",
	"architecture", "sha256", "file_size_bytes", "installed_size_kib",
	"runtime_dependencies", "runtime_dependencies_source", "contents",
	"no_forbidden_payload", "install_validation", "version_result",
	"doctor_result", "list_devices_result", "remove_validation",
)

FORBIDDEN_CONTENT_PATTERNS = (
	r"\.gguf$", r"/tests?/", r"/benchmarks?/", r"\.git/", r"^test_",
	r"/models/",
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


_NO_PATH_RE = re.compile(r'(?<![\w.\-:/])/(?:home|tmp|mnt|var|Users)/[A-Za-z0-9_.\-/]+')


@check("results/distribution/validation.json exists, is valid JSON, has schema_version 1")
def _c1():
	if not DATA_PATH.exists():
		return False, f"{DATA_PATH} does not exist"
	d = _load_data()
	ok = d.get("schema_version") == 1
	return ok, f"schema_version={d.get('schema_version')!r}"


@check("captured_at_commit looks like a real commit SHA (40 hex chars)")
def _c2():
	d = _load_data()
	sha = d.get("captured_at_commit", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"captured_at_commit={sha!r}"


@check("at least a cpu package entry exists (mandatory candidate, Section 25), "
	"each package has every required field")
def _c3():
	d = _load_data()
	pkgs = {p.get("id"): p for p in d.get("packages", [])}
	if "cpu" not in pkgs:
		return False, "no 'cpu' package entry found"
	bad = []
	for pid, p in pkgs.items():
		missing = [f for f in REQUIRED_PACKAGE_FIELDS if f not in p]
		if missing:
			bad.append(f"{pid}: missing {missing}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(pkgs)} package(s), all fields present"


@check("every package's sha256 is well-formed (64 hex chars) and unique per package")
def _c4():
	d = _load_data()
	shas = {}
	bad = []
	for p in d.get("packages", []):
		sha = p.get("sha256", "")
		if not re.fullmatch(r"[0-9a-f]{64}", sha):
			bad.append(f"{p.get('id')}: malformed sha256 {sha!r}")
			continue
		if sha in shas:
			bad.append(f"{p.get('id')} and {shas[sha]} share the same sha256 -- "
				"a copy-paste error, real packages differ")
		shas[sha] = p.get("id")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(shas)} distinct sha256 values"


@check("every package declares architecture amd64 only (no unbuilt/unvalidated "
	"arch claim)")
def _c5():
	d = _load_data()
	bad = [p["id"] for p in d.get("packages", []) if p.get("architecture") != "amd64"]
	return len(bad) == 0, f"{bad}" if bad else "all packages are amd64"


@check("no package's declared contents include forbidden payload "
	"(tests/benchmarks/models/git metadata/build-tree files)")
def _c6():
	d = _load_data()
	bad = []
	for p in d.get("packages", []):
		for entry in p.get("contents", []):
			for pat in FORBIDDEN_CONTENT_PATTERNS:
				if re.search(pat, entry, re.IGNORECASE):
					bad.append(f"{p['id']}: {entry!r} matches forbidden pattern {pat!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no forbidden payload in any package's contents"


@check("every package's contents include the membrane-run binary and the "
	"license, and nothing claims to be under a build directory")
def _c7():
	d = _load_data()
	bad = []
	for p in d.get("packages", []):
		contents = p.get("contents", [])
		if not any("bin/membrane-run" in c for c in contents):
			bad.append(f"{p['id']}: missing /usr/bin/membrane-run")
		if not any("LICENSE" in c for c in contents):
			bad.append(f"{p['id']}: missing a LICENSE file")
	return len(bad) == 0, "; ".join(bad) if bad else "membrane-run + LICENSE present in every package"


@check("every install_validation/remove_validation is REAL (not simulated) "
	"and exit_code 0")
def _c8():
	d = _load_data()
	bad = []
	for p in d.get("packages", []):
		for key in ("install_validation", "remove_validation"):
			v = p.get(key, {})
			if v.get("evidence_class") != "REAL":
				bad.append(f"{p['id']}/{key}: evidence_class={v.get('evidence_class')!r}, expected REAL")
			if v.get("exit_code") != 0:
				bad.append(f"{p['id']}/{key}: exit_code={v.get('exit_code')!r}, expected 0")
	return len(bad) == 0, "; ".join(bad) if bad else "every install/remove REAL and exit_code 0"


@check("every remove_validation reports 0 files remaining (Section 10: clean uninstall)")
def _c9():
	d = _load_data()
	bad = [p["id"] for p in d.get("packages", [])
		if p.get("remove_validation", {}).get("files_remaining") != 0]
	return len(bad) == 0, f"{bad}" if bad else "every package uninstalls cleanly (0 files remaining)"


@check("every version_result's stdout matches the package's own recorded "
	"product_version")
def _c10():
	d = _load_data()
	bad = []
	for p in d.get("packages", []):
		expected = p.get("product_version", "")
		stdout = p.get("version_result", {}).get("stdout", "")
		if expected not in stdout:
			bad.append(f"{p['id']}: product_version {expected!r} not found in "
				f"version_result.stdout {stdout!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "version output matches recorded product_version"


@check("vulkan package's runtime_dependencies include libvulkan1 and the cpu "
	"package's do not (accurate, auto-detected dependency declaration -- "
	"Section 6/7)")
def _c11():
	d = _load_data()
	pkgs = {p["id"]: p for p in d.get("packages", [])}
	bad = []
	cpu = pkgs.get("cpu")
	vulkan = pkgs.get("vulkan")
	if cpu and any("libvulkan" in dep for dep in cpu.get("runtime_dependencies", [])):
		bad.append("cpu package declares a libvulkan dependency -- should not")
	if vulkan and not any("libvulkan" in dep for dep in vulkan.get("runtime_dependencies", [])):
		bad.append("vulkan package does not declare a libvulkan dependency -- should")
	for p in d.get("packages", []):
		if any("glslc" in dep.lower() for dep in p.get("runtime_dependencies", [])):
			bad.append(f"{p['id']}: declares glslc (a build-only tool) as a runtime "
				"dependency -- Section 6: build deps must never leak into runtime deps")
	return len(bad) == 0, "; ".join(bad) if bad else "dependency declarations are accurate per variant"


@check("no absolute developer-machine filesystem path (home/tmp/mnt/var/Users) "
	"anywhere in the evidence file")
def _c12():
	text = DATA_PATH.read_text()
	hits = _NO_PATH_RE.findall(text)
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("the .deb file itself is NOT committed to the repository (Section 24)")
def _c13():
	deb_files = list(REPO_ROOT.rglob("*.deb"))
	# Exclude anything under a gitignored build-* directory or third_party --
	# this check is about what git actually TRACKS, so ask git directly
	# rather than trust .gitignore pattern-matching here.
	import subprocess
	tracked = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.deb"],
		capture_output=True, text=True, check=True).stdout.strip()
	return tracked == "", f"git-tracked .deb files: {tracked!r}" if tracked else "no .deb committed"


@check("the product-version -> Debian-package-version mapping recorded here "
	"matches what CMakeLists.txt's own mapping rule produces from the live "
	"MEMBRANE_VERSION (no drift between the two)")
def _c14():
	header_text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', header_text)
	if not m:
		return False, "MEMBRANE_VERSION not found in product_cli.h"
	live_version = m.group(1)
	# Same rule as CMakeLists.txt's string(REPLACE "-" "~" ...): replace
	# every hyphen with a tilde (Python's str.replace with no count limit
	# matches CMake's string(REPLACE) semantics exactly -- both replace
	# ALL occurrences, not just the first).
	expected_debian_version = live_version.replace("-", "~")
	cmake_text = CMAKE_PATH.read_text()
	if 'string(REPLACE "-" "~" MEMBRANE_DEBIAN_VERSION' not in cmake_text:
		return False, "CMakeLists.txt no longer contains the expected version-mapping rule"
	d = _load_data()
	bad = []
	if d.get("membrane_version") != live_version:
		bad.append(f"validation.json membrane_version={d.get('membrane_version')!r} "
			f"!= live MEMBRANE_VERSION={live_version!r}")
	if d.get("package_version") != expected_debian_version:
		bad.append(f"validation.json package_version={d.get('package_version')!r} "
			f"!= expected {expected_debian_version!r}")
	for p in d.get("packages", []):
		if p.get("package_version") != expected_debian_version:
			bad.append(f"{p['id']}: package_version={p.get('package_version')!r} "
				f"!= expected {expected_debian_version!r}")
		if f"membrane_{expected_debian_version}_amd64.deb" != p.get("filename"):
			bad.append(f"{p['id']}: filename={p.get('filename')!r} does not match "
				f"expected membrane_{expected_debian_version}_amd64.deb")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"live MEMBRANE_VERSION={live_version!r} maps to {expected_debian_version!r}, "
		"consistent everywhere")


@check("CMakeLists.txt's CPack config names the package 'membrane' and docs "
	"reference that same name (no stale package-name claim)")
def _c15():
	cmake_text = CMAKE_PATH.read_text()
	m = re.search(r'set\(CPACK_PACKAGE_NAME\s+"([^"]+)"\)', cmake_text)
	if not m:
		return False, "CPACK_PACKAGE_NAME not set in CMakeLists.txt"
	pkg_name = m.group(1)
	bad = []
	for path in (README_PATH, INSTALL_MD_PATH):
		text = path.read_text()
		if f"{pkg_name}_" not in text and f"sudo apt remove {pkg_name}" not in text:
			bad.append(f"{path.name} never references the '{pkg_name}' package")
	return len(bad) == 0, "; ".join(bad) if bad else f"package name {pkg_name!r} referenced consistently"


@check("docs/install.md documents apt remove/dpkg -r for uninstall, and "
	"never suggests a destructive/root-unsafe alternative")
def _c16():
	text = INSTALL_MD_PATH.read_text()
	bad = []
	if "apt remove" not in text and "dpkg -r" not in text:
		bad.append("no apt remove/dpkg -r uninstall instructions found")
	for bad_pattern in (r"curl[^\n]*\|\s*sh", r"rm -rf /(?!\S)", r"chmod 777"):
		if re.search(bad_pattern, text):
			bad.append(f"docs/install.md contains a dangerous pattern: {bad_pattern}")
	return len(bad) == 0, "; ".join(bad) if bad else "uninstall documented safely, no dangerous patterns"


@check("no CI/build script for packaging runs curl-pipe-sh, auto-sudo, or "
	"modifies shell profile files (Section 29 security audit)")
def _c17():
	ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
	bad = []
	if re.search(r"curl[^\n]*\|\s*(sh|bash)", ci_text):
		bad.append("ci.yml contains a curl-pipe-shell pattern")
	if re.search(r"\.bashrc|\.profile|\.zshrc", ci_text):
		bad.append("ci.yml appears to modify a shell profile file")
	install_sh = REPO_ROOT / "scripts" / "install.sh"
	if install_sh.exists():
		text = install_sh.read_text()
		if re.search(r"curl[^\n]*\|\s*(sh|bash)", text):
			bad.append("scripts/install.sh contains a curl-pipe-shell pattern")
		if re.search(r"\bsudo\b", text) and "read" not in text.lower():
			bad.append("scripts/install.sh appears to invoke sudo without "
				"an explicit user confirmation/opt-in path")
		if re.search(r"\.bashrc|\.profile|\.zshrc", text):
			bad.append("scripts/install.sh appears to modify a shell profile file")
	return len(bad) == 0, "; ".join(bad) if bad else "no unsafe install pattern found"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12,
			_c13, _c14, _c15, _c16, _c17):
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
