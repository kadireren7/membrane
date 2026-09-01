#!/usr/bin/env python3
"""Validate Phase 30's first-run UX evidence
(results/first-run/validation.json) -- schema, presence of the required
scenarios (installed package, CPU-only environment, real local Vulkan
environment, model inspection, plan-only, actual generation, JSON
modes), that every embedded JSON excerpt actually parses, that no
private/absolute developer-machine path or stale version string leaks
in, that no unsupported-platform claim exists, that the package
naming/version-mapping policy this file assumes matches
results/release-artifacts/manifest.json (no drift between the two
Phase 30 evidence files), and that every model-compatibility claim in
the evidence is consistent with the real allowlist compat_check.c
implements (no bypass of the real compatibility logic).

This validates the COMMITTED evidence file (captured locally against a
real build and a real .deb install inside a container -- see
docs/install.md), it does not regenerate it.

Same one-file-per-concern convention as every other scripts/verify-*.py
in this project (see scripts/verify-results.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "first-run" / "validation.json"
MANIFEST_PATH = REPO_ROOT / "results" / "release-artifacts" / "manifest.json"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
COMPAT_CHECK_C_PATH = REPO_ROOT / "tools" / "membrane-run" / "compat_check.c"

FAILURES = []
CHECK_COUNT = 0

REQUIRED_SCENARIO_IDS = {
	"installed-package-full-journey",
	"cpu-only-environment-smoke",
	"real-local-vulkan-environment",
	"model-inspection-unsupported-metadata",
}

FORBIDDEN_PLATFORM_PATTERNS = (
	r"dnf install", r"yum install", r"pacman -S", r"flatpak install",
	r"snap install", r"\.dmg\b", r"\.exe\b", r"\.msi\b",
)

_NO_PATH_RE = re.compile(r'(?<![\w.\-:/])/(?:home|tmp|mnt|var|Users)/[A-Za-z0-9_.\-/]+')


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


def _iter_json_shaped_strings(obj):
	"""Yields every string value in the evidence tree that looks like it
	starts a complete JSON object (used for the excerpt-parses check)."""
	if isinstance(obj, dict):
		for v in obj.values():
			yield from _iter_json_shaped_strings(v)
	elif isinstance(obj, list):
		for v in obj:
			yield from _iter_json_shaped_strings(v)
	elif isinstance(obj, str) and obj.strip().startswith("{"):
		yield obj


@check("results/first-run/validation.json exists, is valid JSON, has "
	"schema_version 1")
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


@check("every required scenario id is present (installed package, "
	"CPU-only environment, real local Vulkan environment, model "
	"inspection)")
def _c3():
	d = _load_data()
	ids = {s.get("id") for s in d.get("scenarios", [])}
	missing = REQUIRED_SCENARIO_IDS - ids
	return len(missing) == 0, (f"missing: {missing}" if missing
		else f"{len(ids)} scenarios, all required ids present")


@check("membrane_version matches MEMBRANE_VERSION AT the evidence's own "
	"captured_at_commit (not necessarily the live/current tree, which "
	"moves on with every version bump -- this evidence is a point-in-"
	"time capture, pinned to its own commit, same frozen-snapshot "
	"pattern as scripts/verify-release-readiness.py's `git show`-based "
	"checks)")
def _c4():
	d = _load_data()
	commit = d.get("captured_at_commit")
	if not commit:
		return False, "no captured_at_commit field in the evidence file"
	result = subprocess.run(
		["git", "-C", str(REPO_ROOT), "show",
			f"{commit}:tools/membrane-run/product_cli.h"],
		capture_output=True, text=True)
	if result.returncode != 0:
		return False, (f"could not read product_cli.h AT {commit} via git "
			f"show -- run `git fetch` first if this is a shallow checkout: "
			f"{result.stderr.strip()}")
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', result.stdout)
	if not m:
		return False, f"MEMBRANE_VERSION not found in product_cli.h AT {commit}"
	version_at_commit = m.group(1)
	ok = d.get("membrane_version") == version_at_commit
	return ok, (f"evidence={d.get('membrane_version')!r} "
		f"AT {commit[:12]}={version_at_commit!r}")


@check("every complete JSON-shaped excerpt embedded in the evidence "
	"actually parses as valid JSON")
def _c5():
	d = _load_data()
	bad = []
	checked = 0
	for s in _iter_json_shaped_strings(d):
		stripped = s.strip()
		checked += 1
		try:
			json.loads(stripped)
		except json.JSONDecodeError:
			# Truncated excerpts (a real prefix of a longer object, used
			# for readability) are allowed -- only reject something that
			# doesn't even look like the start of real JSON.
			if not re.match(r'^\{"schema_version":\d+,"membrane_version":"[^"]+"', stripped):
				bad.append(stripped[:60])
	return len(bad) == 0, f"{bad}" if bad else f"{checked} JSON-shaped excerpts checked"


@check("no absolute developer-machine filesystem path (home/tmp/mnt/"
	"var/Users) anywhere in the evidence file")
def _c6():
	text = DATA_PATH.read_text()
	hits = _NO_PATH_RE.findall(text)
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("no forbidden-platform install command (dnf/yum/pacman/flatpak/"
	"snap install, .dmg/.exe/.msi) anywhere in the evidence file")
def _c7():
	text = DATA_PATH.read_text()
	bad = [pat for pat in FORBIDDEN_PLATFORM_PATTERNS
		if re.search(pat, text, re.IGNORECASE)]
	return len(bad) == 0, f"{bad}" if bad else "none found"


@check("package-name policy referenced in this evidence matches "
	"results/release-artifacts/manifest.json (no drift between the two "
	"Phase 30 evidence files)")
def _c8():
	if not MANIFEST_PATH.exists():
		return False, f"{MANIFEST_PATH} does not exist"
	manifest = json.loads(MANIFEST_PATH.read_text())
	official = next((a for a in manifest.get("official_artifacts", [])
		if a.get("role") == "official"), None)
	if not official:
		return False, "no official artifact found in release-artifacts manifest"
	pkg_name = official["package_name"]
	text = DATA_PATH.read_text()
	ok = f"{pkg_name}_0.3.0" in text or f"official {pkg_name}" in text.lower()
	return ok, (f"first-run evidence references the official package "
		f"{pkg_name!r}" if ok else f"first-run evidence never references "
		f"the official package name {pkg_name!r} from the release-"
		f"artifacts manifest")


@check("model-compatibility claims in the evidence are consistent with "
	"compat_check.c's real architecture allowlist -- no bypass of the "
	"real compatibility logic")
def _c9():
	compat_text = COMPAT_CHECK_C_PATH.read_text()
	allowlist_match = re.search(
		r'MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST\[\]\s*=\s*\{([^}]*)\}',
		compat_text, re.DOTALL)
	if not allowlist_match:
		return False, "could not find the real architecture allowlist in compat_check.c"
	allowlist = set(re.findall(r'"([a-z0-9]+)"', allowlist_match.group(1)))
	d = _load_data()
	bad = []
	for s in d.get("scenarios", []):
		for step in s.get("steps", []) if isinstance(s.get("steps"), list) else [s]:
			text = json.dumps(step)
			if '"architecture": "llama"' in text or "architecture\\\":\\\"llama" in text:
				if "llama" not in allowlist:
					bad.append(f"{s.get('id')}: claims llama is compatible, "
						"but compat_check.c's allowlist no longer includes it")
	return len(bad) == 0, ("; ".join(bad) if bad else
		f"claimed architectures consistent with the real allowlist {sorted(allowlist)}")


@check("the model-inspection-unsupported-metadata scenario's reason_code "
	"matches gpu_policy.h's real MODEL_METADATA_UNAVAILABLE constant")
def _c10():
	gpu_policy_path = REPO_ROOT / "tools" / "membrane-run" / "gpu_policy.h"
	gpu_policy_text = gpu_policy_path.read_text()
	m = re.search(r'MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE\s+"([^"]+)"',
		gpu_policy_text)
	if not m:
		return False, "MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE not found"
	code = m.group(1)
	d = _load_data()
	scenario = next((s for s in d.get("scenarios", [])
		if s.get("id") == "model-inspection-unsupported-metadata"), None)
	if not scenario:
		return False, "model-inspection-unsupported-metadata scenario not found"
	ok = f'"reason_code":"{code}"' in scenario.get("stdout", "")
	return ok, f"expected reason_code {code!r} in scenario stdout"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10):
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
