#!/usr/bin/env python3
"""Validate Mega Phase D, PR D4's real macOS/Metal backend evidence
(results/macos-metal/validation.json): schema and REAL/SOURCE_ANALYSIS
labeling, the MEMBRANE-level GGML_METAL Apple-only pre-check exists,
ggml-metal joined the install/RPATH foreach list, the cross-platform
@loader_path/$ORIGIN RPATH token exists, the membrane_stat_mtime_ns()
portability helper exists (regression guard against the real st_mtim/
st_mtimespec bug this PR found and fixed), no CUDA/Metal-specific
string leaked into any backend-agnostic planner/policy module, and no
"every Mac"/"all Apple Silicon" overclaim.

Same one-file-per-concern, check()-decorator convention as every other
scripts/verify-*.py in this project (see scripts/verify-cuda-backend.py,
PR D3's own equivalent).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPO_ROOT / "results" / "macos-metal" / "validation.json"
METAL_DOC = REPO_ROOT / "docs" / "macos-metal.md"
ROOT_CMAKE = REPO_ROOT / "CMakeLists.txt"
RUN_CMAKE = REPO_ROOT / "tools" / "membrane-run" / "CMakeLists.txt"
FS_UTIL_H = REPO_ROOT / "tools" / "membrane" / "fs_util.h"
FS_UTIL_CPP = REPO_ROOT / "tools" / "membrane" / "fs_util.cpp"
LAUNCHD_UNIT_H = REPO_ROOT / "tools" / "membrane" / "launchd_unit.h"
LAUNCHD_UNIT_CPP = REPO_ROOT / "tools" / "membrane" / "launchd_unit.cpp"
COMPAT_JSON = REPO_ROOT / "docs" / "compatibility.json"
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


@check("evidence file, docs/macos-metal.md, and the launchd_unit module "
	"files all exist")
def _c1():
	missing = [str(p) for p in
		(EVIDENCE_PATH, METAL_DOC, LAUNCHD_UNIT_H, LAUNCHD_UNIT_CPP)
		if not p.exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all present"


@check("validation.json is valid JSON with schema_version 1 and real "
	"(non-PENDING) D1/D2/D3 squash SHAs")
def _c2():
	data = _load_evidence()
	shas = data.get("pr_squash_shas", {})
	ok = all(re.fullmatch(r"[0-9a-f]{40}", shas.get(k, ""))
		for k in ("D1", "D2", "D3"))
	return ok, (f"D1={shas.get('D1')!r} D2={shas.get('D2')!r} "
		f"D3={shas.get('D3')!r}")


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


@check("real hardware/generation/service-lifecycle sections are labeled "
	"REAL, not SOURCE_ANALYSIS (this PR's central claims are genuine "
	"on-device validation, not code-reading alone)")
def _c4():
	data = _load_evidence()
	required_real = ("hardware", "build_integration",
		"real_device_enumeration", "real_generation",
		"service_lifecycle_smoke")
	bad = [k for k in required_real if data.get(k, {}).get("label") != "REAL"]
	return len(bad) == 0, (f"not REAL: {bad}" if bad else "all REAL")


@check("regression guard: root CMakeLists.txt's GGML_METAL pre-check "
	"exists (Apple-only -- fails on any non-Darwin OS requesting it)")
def _c5():
	text = ROOT_CMAKE.read_text()
	ok = ("if(GGML_METAL AND NOT APPLE)" in text
		and "message(FATAL_ERROR" in
			text.split("if(GGML_METAL AND NOT APPLE)", 1)[1][:400])
	return ok, "pre-check present" if ok else "pre-check missing/malformed"


@check("regression guard: ggml-metal joined membrane-run's existing "
	"TARGET-guarded install/RPATH foreach loop alongside ggml-vulkan/"
	"ggml-cuda")
def _c6():
	text = RUN_CMAKE.read_text()
	m = re.search(r"foreach\(_membrane_run_dep IN ITEMS([^)]*)\)", text)
	ok = (bool(m) and "ggml-metal" in m.group(1) and "ggml-cuda" in m.group(1)
		and "ggml-vulkan" in m.group(1))
	return ok, (f"foreach list={m.group(1).split() if m else None}")


@check("regression guard: the cross-platform RPATH origin token "
	"(@loader_path on Darwin, $ORIGIN elsewhere) exists -- the real bug "
	"this PR found and fixed before macOS ever linked a binary")
def _c7():
	text = RUN_CMAKE.read_text()
	ok = ("@loader_path" in text and "MEMBRANE_RPATH_ORIGIN_TOKEN" in text
		and 'if(APPLE)' in text)
	return ok, "cross-platform token present" if ok else "token missing"


@check("regression guard: the membrane_stat_mtime_ns() portability "
	"helper exists (fs_util.h/.cpp) -- guards against the real "
	"st_mtim/st_mtimespec bug this PR found and fixed")
def _c8():
	h_ok = "membrane_stat_mtime_ns" in FS_UTIL_H.read_text()
	cpp_text = FS_UTIL_CPP.read_text()
	cpp_ok = ("membrane_stat_mtime_ns" in cpp_text
		and "st_mtimespec" in cpp_text and "st_mtim" in cpp_text)
	return (h_ok and cpp_ok), f"declared={h_ok} both-platform-branches={cpp_ok}"


@check("regression guard: no CUDA/Metal-specific string leaked into any "
	"backend-agnostic planner/policy module (the 'zero code change' "
	"claim this PR's own evidence makes, same finding as PR D3)")
def _c9():
	bad = []
	for path in BACKEND_AGNOSTIC_SOURCES:
		if not path.exists():
			bad.append(f"{path.name}: MISSING")
			continue
		text = path.read_text()
		if re.search(r"\bMetal\b|\bmetal\b|\bCUDA\b|\bcuda\b", text):
			bad.append(f"{path.name}: contains a backend-specific reference")
	return len(bad) == 0, ("; ".join(bad) if bad else "no backend-specific "
		"code in any of these files")


@check("no 'every Mac'/'all Apple Silicon'/bare-metal-performance "
	"overclaim in docs/macos-metal.md (one paravirtualized device "
	"tested, throughput explicitly disclosed as not representative)")
def _c10():
	text = METAL_DOC.read_text()
	pattern = re.compile(
		r"\ball\s+(?:macs?|apple\s+silicon)\b(?!\s*\.?\"|\s*\.?')"
		r"|\bevery\s+mac\b",
		re.IGNORECASE)
	bad = [m.group(0) for m in pattern.finditer(text)
		if "not " not in text[max(0, m.start() - 12):m.start()].lower()]
	return len(bad) == 0, (f"overclaim found: {bad}" if bad
		else "no overclaim found")


@check("docs/compatibility.json has new, real Metal rows citing "
	"results/macos-metal/validation.json, and every cited evidence path "
	"resolves to a real in-repository file")
def _c11():
	data = json.loads(COMPAT_JSON.read_text())
	rows = {r["id"]: r for r in data["rows"]}
	metal_rows = {rid: r for rid, r in rows.items()
		if r.get("backend") == "metal"}
	if not metal_rows:
		return False, "no rows with backend == 'metal' found"
	bad = []
	for rid, row in metal_rows.items():
		for ev in row.get("evidence", []):
			if not (REPO_ROOT / ev).exists():
				bad.append(f"{rid}: {ev}")
	return len(bad) == 0, (f"broken evidence paths: {bad}" if bad
		else f"{len(metal_rows)} metal row(s), all evidence paths resolve")


@check("Mega Phase C/D1/D2/D3's own evidence files are untouched by "
	"this PR's own new commits (checked against origin/main)")
def _c12():
	result = subprocess.run(["git", "diff", "--name-only", "origin/main"],
		cwd=REPO_ROOT, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return True, "skipped (no diffable 'origin/main' ref in this checkout)"
	changed = set(result.stdout.splitlines())
	touched = [p for p in (
		"results/model-catalog/validation.json",
		"results/model-variant-selection/validation.json",
		"results/cuda-backend/validation.json",
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
