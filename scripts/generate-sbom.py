#!/usr/bin/env python3
"""Generate a CycloneDX-shaped software bill of materials for MEMBRANE.

Mega Phase C, PR C2. Hand-rolled rather than produced by a third-party
SBOM tool (no C++/vendored-header equivalent of a Python/JS dependency
resolver exists that would add real value here) -- but every field is
read directly from this repo's own real, current state (the vendored
header versions actually bundled, the llama.cpp submodule's actual
pinned commit, the actual local patch files applied, and -- when a
built .deb is available -- the actual `Depends:` line CPack computed
from a real `dpkg-shlibdeps` scan of the real binary) rather than a
hand-maintained, driftable list.

Usage:
  scripts/generate-sbom.py [--deb PATH] [--out PATH]

  --deb PATH   a real, already-built .deb (either backend variant) to
               read the real runtime Depends: line from. Without this,
               falls back to the last real, dpkg-observed dependency
               set for a CPU-only build (documented in the fallback
               constant below) with a note that it may be stale --
               always prefer --deb against a freshly built package
               when generating the artifact that ships with a release.
  --out PATH   write the SBOM here (default: stdout).

Exit code: 0 on success, 1 if a required repo file could not be read.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Last real `dpkg -I` observation of a CPU-only build's own computed
# Depends: line (Mega Phase C, PR C2 package-content audit). Used only
# when --deb is not given; always stale-labeled so it is never mistaken
# for a fresh, authoritative reading.
FALLBACK_CPU_DEPENDS = [
	"libc6 (>= 2.38)", "libgcc-s1 (>= 3.4)", "libgomp1 (>= 6)",
	"libstdc++6 (>= 13.1)",
]


def _read(path):
	full = REPO_ROOT / path
	if not full.exists():
		print(f"generate-sbom.py: missing expected file: {path}", file=sys.stderr)
		sys.exit(1)
	return full.read_text()


def _membrane_version():
	text = _read("tools/membrane-run/product_cli.h")
	m = re.search(r'#\s*define\s+MEMBRANE_VERSION\s+"([^"]+)"', text)
	if not m:
		print("generate-sbom.py: MEMBRANE_VERSION not found", file=sys.stderr)
		sys.exit(1)
	return m.group(1)


def _llama_cpp_commit():
	result = subprocess.run(
		["git", "submodule", "status", "third_party/llama.cpp"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=True)
	m = re.match(r"^[ +-]?([0-9a-f]{40})", result.stdout.strip())
	if not m:
		print("generate-sbom.py: could not parse llama.cpp submodule commit",
			file=sys.stderr)
		sys.exit(1)
	return m.group(1)


def _ggml_version():
	text = _read("third_party/llama.cpp/ggml/CMakeLists.txt")
	major = re.search(r"set\(GGML_VERSION_MAJOR (\d+)\)", text)
	minor = re.search(r"set\(GGML_VERSION_MINOR (\d+)\)", text)
	patch = re.search(r"set\(GGML_VERSION_PATCH (\d+)\)", text)
	if not (major and minor and patch):
		print("generate-sbom.py: could not parse ggml version", file=sys.stderr)
		sys.exit(1)
	return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def _nlohmann_json_version():
	text = _read("third_party/llama.cpp/vendor/nlohmann/json.hpp")
	major = re.search(r"#define NLOHMANN_JSON_VERSION_MAJOR (\d+)", text)
	minor = re.search(r"#define NLOHMANN_JSON_VERSION_MINOR (\d+)", text)
	patch = re.search(r"#define NLOHMANN_JSON_VERSION_PATCH (\d+)", text)
	if not (major and minor and patch):
		print("generate-sbom.py: could not parse nlohmann::json version",
			file=sys.stderr)
		sys.exit(1)
	return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def _cpp_httplib_version():
	text = _read("third_party/llama.cpp/vendor/cpp-httplib/httplib.h")
	m = re.search(r'#define CPPHTTPLIB_VERSION "([^"]+)"', text)
	if not m:
		print("generate-sbom.py: could not parse cpp-httplib version",
			file=sys.stderr)
		sys.exit(1)
	return m.group(1)


def _patch_files():
	patches_dir = REPO_ROOT / "patches"
	return sorted(p.name for p in patches_dir.glob("*.patch"))


def _runtime_depends(deb_path):
	if deb_path is None:
		return FALLBACK_CPU_DEPENDS, "fallback (no --deb given, may be stale)"
	result = subprocess.run(["dpkg-deb", "-I", deb_path],
		capture_output=True, text=True, check=True)
	m = re.search(r"^\s*Depends:\s*(.+)$", result.stdout, re.MULTILINE)
	if not m:
		print(f"generate-sbom.py: no Depends: line found in {deb_path}",
			file=sys.stderr)
		sys.exit(1)
	deps = [d.strip() for d in m.group(1).split(",")]
	return deps, f"real dpkg-deb -I read of {Path(deb_path).name}"


def build_sbom(deb_path):
	membrane_version = _membrane_version()
	llama_commit = _llama_cpp_commit()
	depends, depends_source = _runtime_depends(deb_path)
	components = [
		{
			"type": "library",
			"name": "llama.cpp",
			"version": llama_commit,
			"description": "vendored as a git submodule at third_party/"
				"llama.cpp; carries " + str(len(_patch_files()))
				+ " local MEMBRANE patch(es): " + ", ".join(_patch_files()),
			"scope": "required",
		},
		{
			"type": "library",
			"name": "ggml",
			"version": _ggml_version(),
			"description": "bundled inside the llama.cpp submodule, "
				"unmodified by MEMBRANE's own patches",
			"scope": "required",
		},
		{
			"type": "library",
			"name": "nlohmann-json",
			"version": _nlohmann_json_version(),
			"purl": f"pkg:github/nlohmann/json@v{_nlohmann_json_version()}",
			"description": "vendored header-only, under third_party/"
				"llama.cpp/vendor/nlohmann/",
			"scope": "required",
		},
		{
			"type": "library",
			"name": "cpp-httplib",
			"version": _cpp_httplib_version(),
			"purl": f"pkg:github/yhirose/cpp-httplib@v{_cpp_httplib_version()}",
			"description": "vendored header-only, under third_party/"
				"llama.cpp/vendor/cpp-httplib/",
			"scope": "required",
		},
	]
	for dep in depends:
		components.append({
			"type": "library",
			"name": dep.split(" ")[0],
			"version": "",
			"description": f"system runtime dependency ({depends_source}); "
				"version constraint (if any) carried in the .deb's own "
				"Depends: field, not duplicated here",
			"scope": "required",
		})
	return {
		"bomFormat": "CycloneDX",
		"specVersion": "1.5",
		"version": 1,
		"metadata": {
			"component": {
				"type": "application",
				"name": "membrane",
				"version": membrane_version,
			},
			"tools": [{"name": "scripts/generate-sbom.py", "vendor": "membrane"}],
		},
		"components": components,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--deb", default=None,
		help="a real, already-built .deb to read Depends: from")
	parser.add_argument("--out", default=None,
		help="write the SBOM here (default: stdout)")
	args = parser.parse_args()

	sbom = build_sbom(args.deb)
	text = json.dumps(sbom, indent=2) + "\n"
	if args.out:
		Path(args.out).write_text(text)
		print(f"generate-sbom.py: wrote {args.out}", file=sys.stderr)
	else:
		sys.stdout.write(text)
	return 0


if __name__ == "__main__":
	sys.exit(main())
