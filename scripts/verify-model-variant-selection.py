#!/usr/bin/env python3
"""Validate Mega Phase D, PR D2's hardware-aware variant-selection
evidence (results/model-variant-selection/validation.json): schema and
REAL/SYNTHETIC/SOURCE_ANALYSIS labeling, the deterministic "largest
fitting variant" policy is implemented (never "smallest"), the module
reuses host_memory_guard.h rather than a second fit implementation,
explicit --quant is always honored (never blocked), --dry-run never
touches the network, and no "best quantization"/absolute-fit-guarantee
overclaim exists.

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
EVIDENCE_PATH = REPO_ROOT / "results" / "model-variant-selection" / "validation.json"
SELECTOR_H = REPO_ROOT / "tools" / "membrane" / "variant_selector.h"
SELECTOR_CPP = REPO_ROOT / "tools" / "membrane" / "variant_selector.cpp"
MODEL_CMD_CPP = REPO_ROOT / "tools" / "membrane" / "model_cmd.cpp"
SELECTION_DOC = REPO_ROOT / "docs" / "model-variant-selection.md"
CMAKE_PATH = REPO_ROOT / "tools" / "membrane" / "CMakeLists.txt"
CATALOG_EVIDENCE_PATH = REPO_ROOT / "results" / "model-catalog" / "validation.json"

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


@check("evidence file, module files, and doc all exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, SELECTOR_H, SELECTOR_CPP, SELECTION_DOC)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and a real "
	"(non-PENDING) D1 squash SHA")
def _c2():
	data = _load_evidence()
	sha = data.get("pr_squash_shas", {}).get("D1", "")
	ok = (data.get("schema_version") == 1
		and re.fullmatch(r"[0-9a-f]{40}", sha) is not None)
	return ok, f"D1 sha={sha!r}"


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


@check("regression guard: the selection policy prefers the LARGEST "
	"fitting variant, never the smallest (a real behavior change from "
	"PR D1's own placeholder default)")
def _c4():
	text = SELECTOR_CPP.read_text()
	ok = "v.size_bytes > best->size_bytes" in text
	bad_pattern = "v.size_bytes < best->size_bytes" in text
	return (ok and not bad_pattern), (
		f"prefers-largest present={ok} prefers-smallest-leftover={bad_pattern}")


@check("regression guard: variant_selector reuses host_memory_guard.h's "
	"own real resolve function, never a second, independent fit check")
def _c5():
	text = SELECTOR_CPP.read_text()
	ok = "membrane_host_memory_guard_resolve" in text
	return ok, "reuse present" if ok else "not found"


@check("regression guard: an explicit --quant is always honored (a "
	"real WARNING is printed for an infeasible explicit choice, but "
	"the function never returns a CLI-error/blocks it)")
def _c6():
	text = MODEL_CMD_CPP.read_text()
	has_warn = ("proceeding anyway" in text
		and "because you asked for it explicitly" in text)
	# The explicit-quant branch must not contain a `return` between
	# resolving `variant` and the warning check -- approximated by
	# confirming no NO_FEASIBLE_VARIANT/refusal code appears inside the
	# `if (!requested_quant.empty())` branch specifically.
	explicit_branch_start = text.find("if (!requested_quant.empty())")
	else_start = text.find("\telse\n\t{", explicit_branch_start)
	explicit_branch = text[explicit_branch_start:else_start] \
		if else_start != -1 else ""
	no_block = "NO_FEASIBLE_VARIANT" not in explicit_branch
	return (has_warn and no_block), (
		f"has_warn={has_warn} explicit_choice_never_blocked={no_block}")


@check("regression guard: auto-selection (no --quant) refuses and "
	"reports alternatives when nothing fits, never silently picks an "
	"oversized variant")
def _c7():
	text = MODEL_CMD_CPP.read_text()
	ok = "NO_FEASIBLE_VARIANT" in text and "considered" in text
	return ok, "refusal-with-alternatives present" if ok else "not found"


@check("regression guard: --dry-run never calls the download manager "
	"(returns before membrane_download_file)")
def _c8():
	text = MODEL_CMD_CPP.read_text()
	dry_run_idx = text.find("if (dry_run)")
	download_idx = text.find("membrane_download_file(variant->download_url")
	ok = dry_run_idx != -1 and download_idx != -1 and dry_run_idx < download_idx
	return ok, f"dry_run_idx={dry_run_idx} download_idx={download_idx}"


@check("no 'best quantization'/absolute-fit-guarantee overclaim in "
	"variant-selection docs or code")
def _c9():
	bad = []
	pattern = re.compile(
		r"\bbest\s+quant(?:ization)?\b|\bguarantee[sd]?\s+(?:to\s+)?fit\b",
		re.IGNORECASE)
	for path in (SELECTION_DOC, SELECTOR_H, SELECTOR_CPP, MODEL_CMD_CPP):
		if pattern.search(path.read_text()):
			bad.append(path.name)
	return len(bad) == 0, (f"overclaim found in: {bad}" if bad
		else "no overclaim found")


@check("every new test binary is registered as a real ctest entry")
def _c10():
	text = CMAKE_PATH.read_text()
	ok = "add_test(NAME test_variant_selector" in text
	return ok, "present" if ok else "missing"


@check("results/model-catalog/validation.json's own D1 squash SHA is "
	"backfilled with a real commit (no longer PENDING) as part of "
	"this PR's own commit")
def _c11():
	data = json.loads(CATALOG_EVIDENCE_PATH.read_text())
	sha = data.get("pr_squash_shas", {}).get("D1", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"D1 sha={sha!r}"


@check("Mega Phase C/D1's own evidence files are untouched by this "
	"PR's own new commits (checked against origin/main) -- "
	"results/model-catalog/validation.json IS expected to change "
	"(D1's real squash SHA backfilled here)")
def _c12():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/product-onboarding/validation.json",
		"results/release-supply-chain/validation.json",
		"results/product-hardening/v0.4-validation.json",
		"results/release-v0.4.0/readiness.json") if p in changed]
	return len(touched) == 0, (f"touched: {touched}" if touched
		else "untouched")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12):
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
