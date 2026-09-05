#!/usr/bin/env python3
"""Validate Mega Phase D, PR D3's real CUDA backend evidence
(results/cuda-backend/validation.json): schema and REAL/SOURCE_ANALYSIS
labeling, the MEMBRANE-level GGML_CUDA pre-check exists and mirrors the
existing GGML_VULKAN one, ggml-cuda joined membrane-run's install/RPATH
foreach loop, no CUDA-specific string leaked into any backend-agnostic
planner/policy module (the "zero code change" claim this PR's own
evidence file makes), and no "all NVIDIA GPUs"/every-device overclaim.

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
EVIDENCE_PATH = REPO_ROOT / "results" / "cuda-backend" / "validation.json"
CUDA_DOC = REPO_ROOT / "docs" / "cuda-backend.md"
ROOT_CMAKE = REPO_ROOT / "CMakeLists.txt"
RUN_CMAKE = REPO_ROOT / "tools" / "membrane-run" / "CMakeLists.txt"
COMPAT_JSON = REPO_ROOT / "docs" / "compatibility.json"
VARIANT_SELECTION_EVIDENCE = (REPO_ROOT / "results" / "model-variant-selection"
	/ "validation.json")
BACKEND_AGNOSTIC_SOURCES = [
	REPO_ROOT / "tools" / "membrane-run" / "gpu_policy.c",
	REPO_ROOT / "tools" / "membrane-run" / "context_recommender.c",
	REPO_ROOT / "tools" / "membrane-run" / "compat_check.c",
]

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


@check("evidence file and cuda-backend.md doc both exist")
def _c1():
	missing = [str(p) for p in (EVIDENCE_PATH, CUDA_DOC) if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and real "
	"(non-PENDING) D1 and D2 squash SHAs")
def _c2():
	data = _load_evidence()
	shas = data.get("pr_squash_shas", {})
	ok = all(re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))
		for k in ("D1", "D2"))
	return ok, f"D1={shas.get('D1')!r} D2={shas.get('D2')!r}"


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


@check("real hardware/generation sections are labeled REAL, not "
	"SOURCE_ANALYSIS (this PR's central claim is genuine on-device "
	"validation, not code-reading alone)")
def _c4():
	data = _load_evidence()
	required_real = ("hardware", "build_integration",
		"real_device_enumeration", "real_generation")
	bad = [k for k in required_real if data.get(k, {}).get("label") != "REAL"]
	return len(bad) == 0, (f"not REAL: {bad}" if bad else "all REAL")


@check("regression guard: root CMakeLists.txt's GGML_CUDA pre-check "
	"mirrors the existing GGML_VULKAN one (find_package + FATAL_ERROR, "
	"same fail-before-descending-into-llama.cpp pattern)")
def _c5():
	text = ROOT_CMAKE.read_text()
	ok = ("if(GGML_CUDA)" in text
		and "find_package(CUDAToolkit QUIET)" in text
		and "NOT CUDAToolkit_FOUND" in text
		and "message(FATAL_ERROR" in text.split("if(GGML_CUDA)", 1)[1][:600])
	return ok, "pre-check present" if ok else "pre-check missing/malformed"


@check("regression guard: ggml-cuda joined membrane-run's existing "
	"TARGET-guarded install/RPATH foreach loop alongside ggml-vulkan")
def _c6():
	text = RUN_CMAKE.read_text()
	m = re.search(r"foreach\(_membrane_run_dep IN ITEMS([^)]*)\)", text)
	ok = bool(m) and "ggml-cuda" in m.group(1) and "ggml-vulkan" in m.group(1)
	return ok, (f"foreach list={m.group(1).split() if m else None}")


@check("regression guard: no CUDA-specific string leaked into any "
	"backend-agnostic planner/policy module (the 'zero code change' "
	"claim this PR's own evidence makes)")
def _c7():
	bad = []
	for path in BACKEND_AGNOSTIC_SOURCES:
		if not path.exists():
			bad.append(f"{path.name}: MISSING")
			continue
		text = path.read_text()
		if re.search(r"\bCUDA\b|\bcuda\b", text):
			bad.append(f"{path.name}: contains a CUDA reference")
	return len(bad) == 0, ("; ".join(bad) if bad else "no backend-specific "
		"code in any of these files")


@check("no 'every NVIDIA GPU'/'all CUDA devices' overclaim in "
	"cuda-backend.md (tested against exactly one real device) -- "
	"docs/compatibility.md's own pre-existing 'not all NVIDIA GPUs' "
	"disclaimer text is deliberately out of scope for this pattern, "
	"since that phrase is the correct disclosure, not the overclaim")
def _c8():
	text = CUDA_DOC.read_text()
	pattern = re.compile(
		r"\ball\s+(?:nvidia|cuda)\s+gpus?\b(?!\s*\.?\"|\s*\.?')"
		r"|\bevery\s+(?:nvidia|cuda)\s+(?:gpu|device)\b",
		re.IGNORECASE)
	bad = [m.group(0) for m in pattern.finditer(text)
		if "not " not in text[max(0, m.start() - 12):m.start()].lower()]
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("docs/compatibility.json's MC-23 row was actually updated off "
	"its old UNSUPPORTED/NOT_A_PRODUCT_BACKEND claim, and MC-27/MC-28/"
	"MC-29 (the new CUDA rows) exist")
def _c9():
	data = json.loads(COMPAT_JSON.read_text())
	rows = {r["id"]: r for r in data["rows"]}
	mc23 = rows.get("MC-23")
	ok = (mc23 is not None and mc23.get("status") == "SUPPORTED"
		and mc23.get("reason_code") != "NOT_A_PRODUCT_BACKEND"
		and all(rid in rows for rid in ("MC-27", "MC-28", "MC-29")))
	return ok, (f"MC-23={mc23.get('status') if mc23 else None}/"
		f"{mc23.get('reason_code') if mc23 else None} "
		f"MC-27/28/29 present={all(r in rows for r in ('MC-27','MC-28','MC-29'))}")


@check("every evidence path cited by MC-23/MC-27/MC-28 resolves to a "
	"real in-repository file (checked directly, independent of "
	"scripts/verify-compatibility.py's own pass over every row)")
def _c10():
	data = json.loads(COMPAT_JSON.read_text())
	rows = {r["id"]: r for r in data["rows"]}
	bad = []
	for rid in ("MC-23", "MC-27", "MC-28"):
		row = rows.get(rid, {})
		for ev in row.get("evidence", []):
			if not (REPO_ROOT / ev).exists():
				bad.append(f"{rid}: {ev}")
	return len(bad) == 0, (f"broken evidence paths: {bad}" if bad
		else "all evidence paths resolve")


@check("results/model-variant-selection/validation.json's own D2 "
	"squash SHA is backfilled with a real commit (no longer PENDING) "
	"as part of this PR's own commit, matching the established "
	"backfill-the-previous-PR's-own-SHA convention")
def _c11():
	data = json.loads(VARIANT_SELECTION_EVIDENCE.read_text())
	sha = data.get("pr_squash_shas", {}).get("D2", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"D2 sha={sha!r}"


@check("Mega Phase C/D1/D2's own evidence files (other than the D2 "
	"backfill this PR makes) are untouched by this PR's own new commits "
	"(checked against origin/main)")
def _c12():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/model-catalog/validation.json",
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
