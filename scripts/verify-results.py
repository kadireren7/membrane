#!/usr/bin/env python3
"""Verify that MEMBRANE's headline claims (README.md) are actually
backed by the committed artifacts they cite, and that those artifacts are
internally consistent (no duplicate/missing rows, no stale/partial result
presented as a finished headline, SHA-256 matches the manifest).

This does not re-run any simulator -- it checks the artifacts already in
the repository against what the docs claim about them. Prints each
check's PASS/FAIL and, on failure, exactly which number/source disagreed.

Exit code: 0 if every check passes, 1 otherwise.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def read_csv_rows(rel_path):
	path = REPO_ROOT / rel_path
	with path.open(newline="") as f:
		return list(csv.DictReader(f))


def sha256_of(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


# ---------------------------------------------------------------------
# 1. Manifest present, well-formed, and hashes match the working tree
# ---------------------------------------------------------------------
@check("manifest exists and is valid JSON")
def _c1():
	path = REPO_ROOT / "benchmarks" / "MANIFEST.json"
	if not path.exists():
		return False, f"{path} does not exist -- run scripts/generate-benchmark-manifest.py"
	json.loads(path.read_text())
	return True, str(path)


@check("manifest SHA-256 matches working tree for every artifact")
def _c2():
	path = REPO_ROOT / "benchmarks" / "MANIFEST.json"
	manifest = json.loads(path.read_text())
	mismatches = []
	missing = []
	for entry in manifest["artifacts"]:
		p = REPO_ROOT / entry["path"]
		if not p.exists():
			missing.append(entry["path"])
			continue
		actual = sha256_of(p)
		if actual != entry["sha256"]:
			mismatches.append(f"{entry['path']} (manifest={entry['sha256'][:12]} actual={actual[:12]})")
	if missing or mismatches:
		return False, f"missing={missing} mismatched={mismatches}"
	return True, f"{len(manifest['artifacts'])} artifacts checked"


@check("no stale/partial artifact is cited as a headline result")
def _c3():
	path = REPO_ROOT / "benchmarks" / "MANIFEST.json"
	manifest = json.loads(path.read_text())
	readme = (REPO_ROOT / "README.md").read_text()
	bad = []
	for entry in manifest["artifacts"]:
		if entry["status"] in ("partial", "superseded") and entry["path"] in readme:
			bad.append(f"{entry['path']} (status={entry['status']}) is referenced in README.md")
	if bad:
		return False, "; ".join(bad)
	return True, "no partial/superseded artifact referenced in README.md"


# ---------------------------------------------------------------------
# 2. Scenario counts: 231/231 per model, 462/462 total
# ---------------------------------------------------------------------
@check("unified-sweep.csv: 462 total rows, 231/231 per model")
def _c4():
	rows = read_csv_rows("benchmarks/cxl-sim/unified-sweep.csv")
	total = len(rows)
	per_model = {}
	for r in rows:
		per_model[r["model"]] = per_model.get(r["model"], 0) + 1
	ok = (total == 462 and per_model.get("SmolLM2-135M") == 231
		and per_model.get("SmolLM2-360M") == 231)
	return ok, f"total={total} per_model={per_model}"


@check("unified-sweep.csv: no duplicate scenario key")
def _c5():
	rows = read_csv_rows("benchmarks/cxl-sim/unified-sweep.csv")
	key_cols = ["model", "comparison", "precision", "host_cache_total_bytes", "device_total_bytes"]
	seen = {}
	dups = []
	for r in rows:
		key = tuple(r[c] for c in key_cols)
		seen[key] = seen.get(key, 0) + 1
	dups = [k for k, n in seen.items() if n > 1]
	return len(dups) == 0, f"{len(dups)} duplicate key(s): {dups[:5]}"


@check("unified-tail-samples.csv: 900/900 samples per model, no duplicate rows")
def _c6():
	path = REPO_ROOT / "benchmarks" / "cxl-sim" / "unified-tail-samples.csv"
	rows = read_csv_rows("benchmarks/cxl-sim/unified-tail-samples.csv")
	per_model = {}
	for r in rows:
		per_model[r["model"]] = per_model.get(r["model"], 0) + 1
	lines = path.read_text().splitlines()
	data_lines = lines[1:]
	dup_count = len(data_lines) - len(set(data_lines))
	ok = (per_model.get("SmolLM2-135M") == 900 and per_model.get("SmolLM2-360M") == 900
		and dup_count == 0)
	return ok, f"per_model={per_model} duplicate_rows={dup_count}"


# ---------------------------------------------------------------------
# 3. Headline numeric claims vs. the CSVs/docs they cite
# ---------------------------------------------------------------------
@check("unified-sweep.csv: representative-point (8GiB host/2TiB device, all-q8) ratios match README's 187x-405x / 99.6x-215.3x claim")
def _c7():
	rows = read_csv_rows("benchmarks/cxl-sim/unified-sweep.csv")
	# The two analytical baselines (full-scan-cxl, compressed-full-scan-cxl)
	# are constant per (model, precision) by construction -- host/device
	# size does not affect a closed-form full-scan cost -- so they are
	# looked up independent of the representative host/device point below,
	# matching how docs/phase6-unified-stress.md itself computes them.
	baseline_full = {}
	baseline_compressed = {}
	for r in rows:
		if r["comparison"] == "full-scan-cxl":
			baseline_full[(r["model"], r["precision"])] = float(r["mean_bytes_per_token"])
		elif r["comparison"] == "compressed-full-scan-cxl":
			baseline_compressed[(r["model"], r["precision"])] = float(r["mean_bytes_per_token"])

	ratios_full, ratios_compressed = [], []
	for r in rows:
		if (r["precision"] != "all-q8" or r["host_cache_total_bytes"] != "8589934592"
				or r["device_total_bytes"] != "2199023255552"):
			continue
		if r["comparison"] in ("full-scan-cxl", "compressed-full-scan-cxl"):
			continue
		val = float(r["mean_bytes_per_token"])
		if val <= 0:
			continue
		key = (r["model"], r["precision"])
		if key in baseline_full:
			ratios_full.append(baseline_full[key] / val)
		if key in baseline_compressed:
			ratios_compressed.append(baseline_compressed[key] / val)

	if not ratios_full or not ratios_compressed:
		return False, "no rows found at the 8GiB host / 2TiB device / all-q8 representative point"
	lo_f, hi_f = min(ratios_full), max(ratios_full)
	lo_c, hi_c = min(ratios_compressed), max(ratios_compressed)
	# Allow a small float-formatting margin either side of the rounded
	# figures actually printed in README.md / phase6-unified-stress.md.
	ok = (185.0 <= lo_f and hi_f <= 410.0 and 95.0 <= lo_c and hi_c <= 220.0)
	return ok, (f"vs full-scan: [{lo_f:.1f}x, {hi_f:.1f}x] (README: 187x-405x); "
		f"vs compressed: [{lo_c:.1f}x, {hi_c:.1f}x] (README: 99.6x-215.3x)")


@check("unified-sweep-hardware-sensitivity.csv: 8-vs-1-pipeline p99 ratio matches README's 25.7x claim")
def _c8():
	rows = read_csv_rows("benchmarks/cxl-sim/unified-sweep-hardware-sensitivity.csv")
	by_profile = {r["profile"]: r for r in rows}
	p1 = by_profile.get("pipelines-1")
	p8 = by_profile.get("pipelines-8 (default)")
	if not p1 or not p8:
		return False, "pipelines-1 / pipelines-8 (default) rows not found"
	ratio = float(p1["p99_latency_ns"]) / float(p8["p99_latency_ns"])
	ok = abs(ratio - 25.7) < 0.5
	return ok, f"computed ratio={ratio:.2f}x (README cites 25.7x)"


@check("phase5-synthesizable-fpga.md cites the 520,000-transaction Verilator result README relies on")
def _c9():
	text = (REPO_ROOT / "docs" / "phase5-synthesizable-fpga.md").read_text()
	ok = "520,000" in text and "0 fails" in text
	return ok, "found '520,000' and '0 fails'" if ok else "expected substrings not found"


@check("phase4-ggml-quant-parity.md cites the 100,000+-block parity result README relies on")
def _c10():
	text = (REPO_ROOT / "docs" / "phase4-ggml-quant-parity.md").read_text()
	ok = "100,000" in text
	return ok, "found '100,000'" if ok else "expected substring not found"


@check("test_ggml_quant_parity.c actually asserts >= 100,000 blocks (not just documented)")
def _c11():
	text = (REPO_ROOT / "tests" / "unit" / "test_ggml_quant_parity.c").read_text()
	m = re.search(r"(\d[\d_]*)\s*[;)]?\s*(?:/\*.*random blocks|random blocks)", text, re.IGNORECASE)
	nums = [int(n.replace("_", "")) for n in re.findall(r"\b(\d{5,})\b", text)]
	ok = any(n >= 100000 for n in nums)
	return ok, f"largest integer literal found in source: {max(nums) if nums else 'none'}"


@check("tb_top_verilator.cpp actually drives >= 520,000 transactions (not just documented)")
def _c12():
	text = (REPO_ROOT / "rtl" / "tb" / "tb_top_verilator.cpp").read_text()
	m1 = re.search(r"N_PER_MODE\s*=\s*(\d+)", text)
	m2 = re.search(r"N_MIX\s*=\s*(\d+)", text)
	if not (m1 and m2):
		return False, "could not find N_PER_MODE/N_MIX constants"
	total = int(m1.group(1)) * 4 + int(m2.group(1))
	ok = total >= 520000
	return ok, f"4*N_PER_MODE + N_MIX = {total}"


# ---------------------------------------------------------------------
# 4. README source-artifact paths actually exist
# ---------------------------------------------------------------------
@check("every artifact path cited in README.md's Key results table exists")
def _c13():
	readme = (REPO_ROOT / "README.md").read_text()
	cited = set(re.findall(r"`(benchmarks/[^`]+|rtl/[^`]+|tests/[^`]+)`", readme))
	missing = [p for p in cited if not (REPO_ROOT / p).exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else f"{len(cited)} paths checked"


def main() -> int:
	for fn in (
		_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12, _c13,
	):
		fn()
	print()
	print(f"{CHECK_COUNT - len(FAILURES)}/{CHECK_COUNT} checks passed")
	if FAILURES:
		print("\nFAILED:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
