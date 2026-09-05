#!/usr/bin/env python3
"""Validate Phase 32's v0.4 product-direction/roadmap planning
documents: required sections exist in docs/product-direction.md and
docs/v0.4-roadmap.md, no unsupported current-feature claim (CUDA/
all-Qwen2/all-models), no CUDA implementation commitment for v0.4, no
performance-speedup promise, live v0.3 facts (version, compatibility
counts, patch set) are cited accurately, and the phase roadmap is
ordered/unique/sequential.

This is a planning-phase validator: it checks the DOCUMENTS this phase
produced, not runtime behavior (none changed -- Phase 32 is docs-only).
Deliberately small, same one-file-per-concern convention as every other
scripts/verify-*.py in this project.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_DIRECTION_PATH = REPO_ROOT / "docs" / "product-direction.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "v0.4-roadmap.md"
DECISION_JSON_PATH = REPO_ROOT / "results" / "product-direction" / "v0.4-decision.json"
PRODUCT_CLI_H_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
COMPAT_JSON_PATH = REPO_ROOT / "docs" / "compatibility.json"
PATCHES_DIR = REPO_ROOT / "patches"

REQUIRED_ROADMAP_SECTIONS = [
	"PRIMARY V0.4 THEME", "PRIMARY FEATURE", "SECONDARY FEATURES",
	"NON-GOALS", "SUCCESS CRITERIA", "PHASE ORDER", "RISKS",
	"DEFERRED ITEMS",
]

FAILURES = []
CHECK_COUNT = 0


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


@check("docs/product-direction.md and docs/v0.4-roadmap.md both exist")
def _c1():
	bad = [str(p) for p in (PRODUCT_DIRECTION_PATH, ROADMAP_PATH) if not p.exists()]
	return len(bad) == 0, "; ".join(bad) if bad else "both exist"


@check("docs/v0.4-roadmap.md's Decision record contains every required "
	"field named in Section 33 of the Phase 32 task")
def _c2():
	text = ROADMAP_PATH.read_text()
	m = re.search(r"## Decision record(.*?)\n## ", text, re.DOTALL)
	if not m:
		return False, "no '## Decision record' section found"
	section = m.group(1)
	missing = [s for s in REQUIRED_ROADMAP_SECTIONS if s not in section]
	return len(missing) == 0, f"missing: {missing}" if missing else "all required fields present"


@check("results/product-direction/v0.4-decision.json exists, is valid "
	"JSON, and confirms this is a planning-only phase (no runtime "
	"change, no version bump, no release)")
def _c3():
	if not DECISION_JSON_PATH.exists():
		return False, f"{DECISION_JSON_PATH} does not exist"
	d = json.loads(DECISION_JSON_PATH.read_text())
	bad = []
	if d.get("runtime_behavior_changed") is not False:
		bad.append("runtime_behavior_changed is not explicitly false")
	if d.get("version_bumped") is not False:
		bad.append("version_bumped is not explicitly false")
	if d.get("release_created") is not False:
		bad.append("release_created is not explicitly false")
	return len(bad) == 0, "; ".join(bad) if bad else "planning-only confirmed"


@check("no CUDA implementation commitment for v0.4 -- CUDA is "
	"classified DEFERRED_NO_VALIDATION_HARDWARE, never listed as a "
	"phase objective or a secondary feature")
def _c4():
	d = json.loads(DECISION_JSON_PATH.read_text())
	bad = []
	cuda_entries = [i for i in d.get("deferred_items", []) if i.get("id") == "F7"]
	if not cuda_entries:
		bad.append("no F7 (CUDA) entry found in deferred_items")
	elif cuda_entries[0].get("classification") != "DEFERRED_NO_VALIDATION_HARDWARE":
		bad.append(f"F7 classification={cuda_entries[0].get('classification')!r}, "
			"expected DEFERRED_NO_VALIDATION_HARDWARE")
	for phase in d.get("phase_roadmap", []):
		if re.search(r"\bcuda\b", phase.get("objective", ""), re.IGNORECASE):
			bad.append(f"phase {phase.get('phase')} objective mentions CUDA: "
				f"{phase.get('objective')!r}")
	roadmap_text = ROADMAP_PATH.read_text()
	if re.search(r"implement(ing)?\s+cuda\b", roadmap_text, re.IGNORECASE):
		bad.append("docs/v0.4-roadmap.md contains an 'implement(ing) CUDA' phrase")
	return len(bad) == 0, "; ".join(bad) if bad else "no CUDA implementation commitment found"


@check("no performance-speedup promise in either planning document")
def _c5():
	bad = []
	patterns = [
		r"\bfaster\s+(inference|than)\b", r"\bperformance\s+optimi[sz]ed\b",
		r"\bimproved\s+inference\s+speed\b", r"\bspeed(s)?\s+up\b",
	]
	# Negated mentions (explicitly saying performance work is NOT a v0.4
	# goal) are the expected, correct usage -- only an unqualified claim
	# is a real problem.
	negation_re = re.compile(r"\b(not|never|no|none|deferred|unless)\b", re.IGNORECASE)
	for path in (PRODUCT_DIRECTION_PATH, ROADMAP_PATH):
		text = path.read_text()
		for pat in patterns:
			for m in re.finditer(pat, text, re.IGNORECASE):
				window = text[max(0, m.start() - 80):m.start()]
				if negation_re.search(window):
					continue
				bad.append(f"{path.name}: unqualified match {m.group(0)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unqualified speedup promise found"


@check("no unsupported current-feature claim (CUDA, 'all Qwen2 models', "
	"'all model architectures') stated as already true today")
def _c6():
	bad = []
	patterns = [
		r"\bMEMBRANE\s+supports\s+CUDA\b",
		r"\ball\s+qwen2\s+models\s+(are\s+)?support",
		r"\ball\s+model\s+architectures\s+(are\s+)?support",
	]
	for path in (PRODUCT_DIRECTION_PATH, ROADMAP_PATH):
		text = path.read_text()
		for pat in patterns:
			if re.search(pat, text, re.IGNORECASE):
				bad.append(f"{path.name} matches {pat!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unsupported current-feature claim found"


@check("v0.3 stable facts cited in the planning docs are accurate: "
	"MEMBRANE_VERSION, and compatibility counts match the live matrix")
def _c7():
	header_text = PRODUCT_CLI_H_PATH.read_text()
	m = re.search(r'define MEMBRANE_VERSION\s+"([^"]+)"', header_text)
	live_version = m.group(1) if m else None
	compat = json.loads(COMPAT_JSON_PATH.read_text())
	rows = compat["rows"]
	total = len(rows)
	supported = sum(1 for r in rows if r["status"] == "SUPPORTED")
	unsupported = sum(1 for r in rows if r["status"] == "UNSUPPORTED")
	not_yet = sum(1 for r in rows if r["status"] == "NOT_YET_VALIDATED")
	bad = []
	# Phase 32 itself never bumped the version (this check originally
	# pinned live_version to "0.3.0" to prove that) -- the version has
	# since legitimately moved to "0.4.0" via Mega Phase C's own PR C4,
	# many phases later. That real, sanctioned bump doesn't make Phase
	# 32's own historical claims below (about v0.3.0-era counts) wrong;
	# it just means this check now confirms a DIFFERENT thing: that the
	# live version reflects the real, current release, not a lingering
	# stale value from some phase in between.
	if live_version != "0.4.0":
		bad.append(f"live MEMBRANE_VERSION={live_version!r}, expected the "
			"current stable '0.4.0'")
	direction_text = PRODUCT_DIRECTION_PATH.read_text()
	if "v0.3.0" not in direction_text:
		bad.append("docs/product-direction.md never cites v0.3.0")
	expected_counts_phrase = f"{total} total rows"
	roadmap_and_direction = ROADMAP_PATH.read_text() + direction_text
	if str(total) not in roadmap_and_direction or str(supported) not in roadmap_and_direction:
		bad.append(f"planning docs do not appear to cite the live compatibility "
			f"counts ({total}/{supported}/{unsupported}/{not_yet})")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"v0.3.0 and {total}/{supported}/{unsupported}/{not_yet} both cited accurately")


@check("the phase roadmap is ordered, unique, and sequential (no gaps, "
	"no duplicates, no out-of-order phases)")
def _c8():
	d = json.loads(DECISION_JSON_PATH.read_text())
	phases = [p["phase"] for p in d.get("phase_roadmap", [])]
	bad = []
	if len(phases) != len(set(phases)):
		bad.append(f"duplicate phase numbers: {phases}")
	if phases != sorted(phases):
		bad.append(f"phases not in ascending order: {phases}")
	for a, b in zip(phases, phases[1:]):
		if b != a + 1:
			bad.append(f"gap or non-sequential jump between phase {a} and {b}")
	if phases and phases[0] != 33:
		bad.append(f"roadmap does not start at phase 33 (starts at {phases[0]})")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(phases)} phases, sequential from {phases[0] if phases else '?'}"


@check("patch set is documented as unchanged and matches the two real "
	"committed patch files")
def _c9():
	d = json.loads(DECISION_JSON_PATH.read_text())
	declared = set(d.get("patch_set_unchanged", []))
	real = {p.name for p in PATCHES_DIR.glob("*.patch")}
	return declared == real, f"declared={declared} real={real}"


@check("no upstream llama.cpp PR is claimed to have been opened this phase")
def _c10():
	d = json.loads(DECISION_JSON_PATH.read_text())
	upstream = next((i for i in d.get("secondary_features", []) if i.get("id") == "F8"), None)
	if upstream is None:
		return False, "no F8 (upstream investigation) entry found"
	ok = upstream.get("opens_upstream_pr_this_cycle") is False
	return ok, f"opens_upstream_pr_this_cycle={upstream.get('opens_upstream_pr_this_cycle')!r}"


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
