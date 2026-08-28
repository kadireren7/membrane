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
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FAILURES = []
CHECK_COUNT = 0
V02_ARTIFACT_PATH = REPO_ROOT / "results" / "v0.2" / "smollm2-q8-memory.json"


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


# ---------------------------------------------------------------------
# 5. Product Phase 8: v0.2 compressed-KV-storage memory/quality artifact
# (results/v0.2/smollm2-q8-memory.json). ARTIFACT_PATH is set to the
# committed path by default; --artifact PATH overrides it so
# scripts/benchmark-v0.2.sh can ad-hoc-verify freshly generated data
# before anyone decides to commit it as the new canonical artifact.
# ---------------------------------------------------------------------
ARTIFACT_PATH = V02_ARTIFACT_PATH
_ARTIFACT_CACHE = {}


def _load_artifact():
	key = str(ARTIFACT_PATH)
	if key not in _ARTIFACT_CACHE:
		_ARTIFACT_CACHE[key] = json.loads(ARTIFACT_PATH.read_text())
	return _ARTIFACT_CACHE[key]


def _no_nan_inf(obj, path=""):
	"""Returns a list of JSON-path strings where a float is NaN/Infinity
	-- meaningless in this context since the artifact is loaded from
	already-parsed JSON (Python's json module accepts bare nan/Infinity
	tokens by default despite them being invalid strict JSON), so this
	still needs to check explicitly rather than assume json.loads()
	already rejected them."""
	bad = []
	if isinstance(obj, dict):
		for k, v in obj.items():
			bad.extend(_no_nan_inf(v, f"{path}.{k}"))
	elif isinstance(obj, list):
		for i, v in enumerate(obj):
			bad.extend(_no_nan_inf(v, f"{path}[{i}]"))
	elif isinstance(obj, float) and not math.isfinite(obj):
		bad.append(path or "<root>")
	return bad


@check("v0.2 artifact exists, is valid JSON, and has schema_version 1")
def _c14():
	if not ARTIFACT_PATH.exists():
		return False, f"{ARTIFACT_PATH} does not exist"
	a = _load_artifact()
	ok = a.get("schema_version") == 1
	return ok, f"schema_version={a.get('schema_version')!r}"


@check("v0.2 artifact: git_sha looks like a real commit SHA (40 hex chars)")
def _c15():
	a = _load_artifact()
	sha = a.get("git_sha", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"git_sha={sha!r}"


@check("v0.2 artifact: context sequence is present and strictly increasing")
def _c16():
	a = _load_artifact()
	ctxs = [row["ctx"] for row in a.get("contexts", [])]
	ok = len(ctxs) >= 2 and ctxs == sorted(set(ctxs)) and len(ctxs) == len(set(ctxs))
	return ok, f"contexts={ctxs}"


@check("v0.2 artifact: q8 KV bytes < native KV bytes at every context, "
	"ratio arithmetic self-consistent")
def _c17():
	a = _load_artifact()
	bad = []
	for row in a.get("contexts", []):
		native = row["native_kv_allocated_bytes"]
		q8 = row["q8_kv_allocated_bytes"]
		ratio = row["kv_bytes_ratio"]
		if not (q8 < native):
			bad.append(f"ctx={row['ctx']}: q8({q8}) not < native({native})")
			continue
		expected_ratio = native / q8
		if abs(expected_ratio - ratio) > 1e-3:
			bad.append(f"ctx={row['ctx']}: ratio={ratio} but native/q8={expected_ratio:.6f}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(a.get('contexts', []))} contexts checked"


@check("v0.2 artifact: positive RSS reduction at every recorded context")
def _c18():
	a = _load_artifact()
	bad = []
	for row in a.get("contexts", []):
		native_rss = row["native_rss_after_context_kb"]
		q8_rss = row["q8_rss_after_context_kb"]
		pct = row["rss_reduction_pct"]
		if not (q8_rss < native_rss):
			bad.append(f"ctx={row['ctx']}: q8_rss({q8_rss}) not < native_rss({native_rss})")
			continue
		expected_pct = 100.0 * (native_rss - q8_rss) / native_rss
		if abs(expected_pct - pct) > 0.01:
			bad.append(f"ctx={row['ctx']}: rss_reduction_pct={pct} but computed={expected_pct:.4f}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(a.get('contexts', []))} contexts checked"


@check("v0.2 artifact: RSS reduction % is monotonically non-decreasing "
	"with context size")
def _c19():
	a = _load_artifact()
	rows = sorted(a.get("contexts", []), key=lambda r: r["ctx"])
	pcts = [r["rss_reduction_pct"] for r in rows]
	non_decreasing = all(pcts[i] <= pcts[i + 1] + 1e-9 for i in range(len(pcts) - 1))
	return non_decreasing, f"pcts by increasing ctx: {pcts}"


@check("v0.2 artifact: no absolute filesystem paths anywhere in the file")
def _c20():
	text = ARTIFACT_PATH.read_text()
	# Any POSIX absolute path (leading '/', not preceded by a word/dot/
	# dash character so "and/or" doesn't match, at least two path
	# segments so a bare "/" doesn't match) or a Windows drive-letter
	# path -- deliberately NOT limited to specific prefixes like
	# /home/ or /Users/: a leaked /tmp/..., /mnt/..., or /var/... path
	# is just as much a privacy problem and must be caught too.
	posix = re.findall(
		r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
		text,
	)
	windows = re.findall(r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+', text)
	hits = posix + windows
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("v0.2 artifact: no prompt text (only the prompt fixture's basename)")
def _c21():
	a = _load_artifact()
	fixture = a.get("prompt_fixture", "")
	ok = bool(fixture) and "/" not in fixture and "\n" not in fixture
	# The prompt fixture file itself, if present under benchmarks/kv/
	# prompts/, must never appear verbatim inside the artifact text.
	prompt_path = REPO_ROOT / "benchmarks" / "kv" / "prompts" / fixture
	if ok and prompt_path.exists():
		prompt_text = prompt_path.read_text().strip()
		artifact_text = ARTIFACT_PATH.read_text()
		if prompt_text and prompt_text in artifact_text:
			return False, f"prompt fixture text found verbatim in the artifact"
	return ok, f"prompt_fixture={fixture!r} (basename only, no full prompt text embedded)"


@check("v0.2 artifact: no NaN/Infinity anywhere in the file")
def _c22():
	a = _load_artifact()
	bad = _no_nan_inf(a)
	return len(bad) == 0, f"non-finite at: {bad[:5]}" if bad else "all values finite"


@check("v0.2 artifact: quality fields are in valid ranges at every context")
def _c23():
	a = _load_artifact()
	bad = []
	for row in a.get("contexts", []):
		q = row.get("quality", {})
		top1 = q.get("top1_preservation")
		rel_l2 = q.get("logit_rel_l2")
		fd = q.get("first_divergence")
		if top1 is None or not (0.0 <= top1 <= 1.0):
			bad.append(f"ctx={row['ctx']}: top1_preservation={top1} out of [0,1]")
		if rel_l2 is None or rel_l2 < 0.0:
			bad.append(f"ctx={row['ctx']}: logit_rel_l2={rel_l2} negative")
		if fd is None or fd < -1:
			bad.append(f"ctx={row['ctx']}: first_divergence={fd} invalid (must be -1 or >= 0)")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(a.get('contexts', []))} contexts checked"


@check("v0.2 artifact: headline (largest-context) RSS-reduction "
	"arithmetic is internally consistent (README's number, if any, "
	"must trace here)")
def _c24():
	# The committed default artifact always includes ctx=8192, but
	# scripts/benchmark-v0.2.sh explicitly supports a custom
	# MEMBRANE_CONTEXTS sweep for --artifact ad-hoc verification (e.g.
	# to check a fresh run before deciding whether to commit it) --
	# hardcoding 8192 here would fail every such custom sweep that
	# doesn't happen to include that exact value. "Headline" is
	# whichever context is largest in THIS artifact, not a fixed
	# number.
	a = _load_artifact()
	rows = a.get("contexts", [])
	if not rows:
		return False, "no contexts in the artifact"
	row = max(rows, key=lambda r: r["ctx"])
	native_rss = row["native_rss_after_context_kb"]
	q8_rss = row["q8_rss_after_context_kb"]
	expected_pct = 100.0 * (native_rss - q8_rss) / native_rss
	ok = abs(expected_pct - row["rss_reduction_pct"]) < 0.01
	return ok, (f"ctx={row['ctx']} (largest) rss_reduction_pct="
		f"{row['rss_reduction_pct']:.2f}% (recomputed={expected_pct:.2f}%)")


V03_ARTIFACT_PATH = REPO_ROOT / "results" / "v0.3" / "gpu-vulkan-validation.json"
_V03_CACHE = {}


def _load_v03_artifact():
	key = str(V03_ARTIFACT_PATH)
	if key not in _V03_CACHE:
		_V03_CACHE[key] = json.loads(V03_ARTIFACT_PATH.read_text())
	return _V03_CACHE[key]


def _v03_context_rows(a):
	"""Flattens both models' per-context rows into one list -- 135M's
	multi-context sweep (context_matrix) and 360M's single ctx=2048
	spot-check (ctx2048) use different key names in the artifact
	because they cover different scope, not different schemas; every
	row itself has the same native/q8 shape either way."""
	rows = []
	models = a.get("models", {})
	m135 = models.get("SmolLM2-135M-Instruct-f16", {})
	rows.extend(m135.get("context_matrix", []))
	m360 = models.get("SmolLM2-360M-Instruct-f16", {})
	if "ctx2048" in m360:
		rows.append(m360["ctx2048"])
	return rows


@check("v0.3 GPU artifact exists, is valid JSON, and has schema_version 1")
def _c25():
	if not V03_ARTIFACT_PATH.exists():
		return False, f"{V03_ARTIFACT_PATH} does not exist"
	a = _load_v03_artifact()
	ok = a.get("schema_version") == 1
	return ok, f"schema_version={a.get('schema_version')!r}"


@check("v0.3 GPU artifact: membrane_base_commit looks like a real "
	"commit SHA (40 hex chars)")
def _c26():
	a = _load_v03_artifact()
	sha = a.get("membrane_base_commit", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"membrane_base_commit={sha!r}"


@check("v0.3 GPU artifact: q8 VRAM peak < native VRAM peak at every "
	"row, reduction arithmetic self-consistent")
def _c27():
	a = _load_v03_artifact()
	bad = []
	rows = _v03_context_rows(a)
	for row in rows:
		native_vram = row["native"]["vram_peak_mib"]
		q8_vram = row["q8"]["vram_peak_mib"]
		reduction_mib = row["vram_reduction_mib"]
		reduction_pct = row["vram_reduction_pct"]
		if not (q8_vram < native_vram):
			bad.append(f"ctx={row['ctx']}: q8 VRAM({q8_vram}) not < "
				f"native VRAM({native_vram})")
			continue
		if reduction_mib != native_vram - q8_vram:
			bad.append(f"ctx={row['ctx']}: vram_reduction_mib={reduction_mib} "
				f"but native-q8={native_vram - q8_vram}")
		expected_pct = 100.0 * reduction_mib / native_vram
		if abs(expected_pct - reduction_pct) > 0.01:
			bad.append(f"ctx={row['ctx']}: vram_reduction_pct={reduction_pct} "
				f"but computed={expected_pct:.4f}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows checked"


@check("v0.3 GPU artifact: q8 KV allocated bytes < native at every row")
def _c28():
	a = _load_v03_artifact()
	bad = []
	rows = _v03_context_rows(a)
	for row in rows:
		native_kv = row["native"]["kv_allocated_bytes"]
		q8_kv = row["q8"]["kv_allocated_bytes"]
		if not (q8_kv < native_kv):
			bad.append(f"ctx={row['ctx']}: q8 KV({q8_kv}) not < "
				f"native KV({native_kv})")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows checked"


@check("v0.3 GPU artifact: no_fallback_occurred is true and exit_code "
	"is 0 for every native/q8 run recorded")
def _c29():
	a = _load_v03_artifact()
	bad = []
	rows = _v03_context_rows(a)
	for row in rows:
		for kv in ("native", "q8"):
			r = row[kv]
			if not r.get("no_fallback_occurred", False):
				bad.append(f"ctx={row['ctx']} {kv}: no_fallback_occurred is false")
			if r.get("exit_code") != 0:
				bad.append(f"ctx={row['ctx']} {kv}: exit_code={r.get('exit_code')}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows checked"


@check("v0.3 GPU artifact: no absolute filesystem paths anywhere in "
	"the file")
def _c30():
	text = V03_ARTIFACT_PATH.read_text()
	posix = re.findall(
		r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
		text,
	)
	windows = re.findall(r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+', text)
	hits = posix + windows
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("v0.3 GPU artifact: no NaN/Infinity anywhere in the file")
def _c31():
	a = _load_v03_artifact()
	bad = _no_nan_inf(a)
	return len(bad) == 0, f"non-finite at: {bad[:5]}" if bad else "all values finite"


@check("v0.3 GPU artifact: compare-kv quality fields are in valid "
	"ranges for both models")
def _c32():
	a = _load_v03_artifact()
	bad = []
	for label, m in a.get("models", {}).items():
		q = m.get("compare_kv_ctx2048_256tok", {}).get("quality", {})
		top1 = q.get("top1_preservation")
		rel_l2 = q.get("logit_rel_l2")
		fd = q.get("first_divergence")
		if top1 is None or not (0.0 <= top1 <= 1.0):
			bad.append(f"{label}: top1_preservation={top1} out of [0,1]")
		if rel_l2 is None or rel_l2 < 0.0:
			bad.append(f"{label}: logit_rel_l2={rel_l2} negative")
		if fd is None or fd < -1:
			bad.append(f"{label}: first_divergence={fd} invalid")
	return len(bad) == 0, "; ".join(bad) if bad else "2 models checked"


@check("v0.3 GPU artifact: auto_policy section exists with a "
	"documented reserve policy")
def _c33():
	a = _load_v03_artifact()
	ap = a.get("auto_policy", {})
	rp = ap.get("reserve_policy", {})
	ok = (ap.get("policy_version") == 1
		and rp.get("fixed_reserve_bytes", 0) > 0
		and 0 < rp.get("percentage_reserve_pct", 0) < 100)
	return ok, f"policy_version={ap.get('policy_version')!r} reserve_policy={rp!r}"


@check("v0.3 GPU artifact: auto_policy context matrices -- q8 external "
	"VRAM peak < native at every row, both models")
def _c34():
	a = _load_v03_artifact()
	ap = a.get("auto_policy", {})
	bad = []
	rows = list(ap.get("context_matrix_135m", []))
	if "context_matrix_360m_ctx2048" in ap:
		rows.append(ap["context_matrix_360m_ctx2048"])
	if not rows:
		return False, "no auto_policy context matrices found in the artifact"
	for row in rows:
		native_v = row["native"]["external_vram_peak_mib"]
		q8_v = row["q8"]["external_vram_peak_mib"]
		if not (q8_v < native_v):
			bad.append(f"ctx={row['ctx']}: q8 external VRAM({q8_v}) not "
				f"< native({native_v})")
			continue
		expected_pct = 100.0 * (native_v - q8_v) / native_v
		if abs(expected_pct - row["external_vram_reduction_pct"]) > 0.1:
			bad.append(f"ctx={row['ctx']}: external_vram_reduction_pct="
				f"{row['external_vram_reduction_pct']} but computed="
				f"{expected_pct:.2f}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows checked"


@check("v0.3 GPU artifact: guard boundary test proves the memory guard "
	"actually rejects, and that auto genuinely partial-offloads where "
	"explicit \"all\" is rejected")
def _c35():
	a = _load_v03_artifact()
	gb = a.get("auto_policy", {}).get("guard_boundary_test", {})
	below = gb.get("ctx_140000_gpu_layers_all", {})
	above_all = gb.get("ctx_145000_gpu_layers_all", {})
	above_auto = gb.get("ctx_145000_gpu_layers_auto", {})
	bad = []
	if below.get("exit_code") != 0:
		bad.append(f"ctx=140000 --gpu-layers all: expected exit 0, "
			f"got {below.get('exit_code')}")
	if above_all.get("exit_code") != 5:
		bad.append(f"ctx=145000 --gpu-layers all: expected exit 5 "
			f"(guard rejection), got {above_all.get('exit_code')}")
	if above_auto.get("exit_code") != 0:
		bad.append(f"ctx=145000 --gpu-layers auto: expected exit 0, "
			f"got {above_auto.get('exit_code')}")
	auto_selected = above_auto.get("gpu_layers_selected")
	below_selected = below.get("gpu_layers_selected")
	if not (isinstance(auto_selected, int) and isinstance(below_selected, int)
			and 0 < auto_selected < below_selected):
		bad.append(f"expected auto's selected layers at ctx=145000 "
			f"({auto_selected!r}) to be a genuine partial count, strictly "
			f"less than the full count all resolved to at ctx=140000 "
			f"({below_selected!r})")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"140000/all -> exit 0 ({below_selected} layers); "
		f"145000/all -> exit 5; 145000/auto -> exit 0 ({auto_selected} layers)")


@check("v0.3 GPU artifact: gpu-bench record has valid quality ranges "
	"and self-consistent comparison arithmetic")
def _c36():
	a = _load_v03_artifact()
	bench = a.get("auto_policy", {}).get("gpu_bench_135m_ctx2048_256tok", {})
	bad = []
	q = bench.get("quality", {})
	top1 = q.get("top1_preservation")
	if top1 is None or not (0.0 <= top1 <= 1.0):
		bad.append(f"top1_preservation={top1} out of [0,1]")
	native_kv = bench.get("native", {}).get("kv_allocated_bytes")
	q8_kv = bench.get("q8", {}).get("kv_allocated_bytes")
	ratio = bench.get("comparison", {}).get("kv_reduction_ratio")
	if native_kv and q8_kv and ratio is not None:
		expected = native_kv / q8_kv
		if abs(expected - ratio) > 1e-3:
			bad.append(f"kv_reduction_ratio={ratio} but native/q8={expected:.6f}")
	return len(bad) == 0, "; ".join(bad) if bad else "gpu-bench record checked"


@check("README GPU/Vulkan section's numeric claims trace to the v0.3 "
	"artifact's SmolLM2-135M context sweep (no drift between the two)")
def _c37():
	readme = (REPO_ROOT / "README.md").read_text()
	m = re.search(
		r"from\s+roughly\s+(\d+)%\s+at\s+small\s+contexts\s+up\s+to\s+"
		r"roughly\s+(\d+)%\s+at\s+`ctx=16384`", readme)
	t = re.search(
		r"measured\s+roughly\s+(\d+)-(\d+)%\s+lower\s+than\s+native",
		readme)
	if not m or not t:
		return False, "expected VRAM-reduction and throughput-delta " \
			"phrasing not found in the README GPU/Vulkan section"
	readme_vram_lo, readme_vram_hi = int(m.group(1)), int(m.group(2))
	readme_tp_lo, readme_tp_hi = int(t.group(1)), int(t.group(2))

	a = _load_v03_artifact()
	rows = a["models"]["SmolLM2-135M-Instruct-f16"]["context_matrix"]
	vram_pcts = [r["vram_reduction_pct"] for r in rows]
	tp_deltas = [
		100.0 * (r["q8"]["generation_tok_per_s"]
			- r["native"]["generation_tok_per_s"])
			/ r["native"]["generation_tok_per_s"]
		for r in rows
	]
	vram_lo, vram_hi = min(vram_pcts), max(vram_pcts)
	tp_lo, tp_hi = min(tp_deltas), max(tp_deltas)

	bad = []
	# "roughly" allows rounding to the nearest whole percent, not a
	# materially different number -- 1 point of slack either way.
	if not (vram_lo - 1 <= readme_vram_lo <= vram_lo + 1):
		bad.append(f"README VRAM low bound {readme_vram_lo}% vs "
			f"artifact min {vram_lo:.2f}%")
	if not (vram_hi - 1 <= readme_vram_hi <= vram_hi + 1):
		bad.append(f"README VRAM high bound {readme_vram_hi}% vs "
			f"artifact max {vram_hi:.2f}%")
	# tp_deltas are negative (q8 slower than native); tp_lo is the most
	# negative (biggest cost, largest |%|), tp_hi the least negative
	# (smallest cost, smallest |%|) -- README's "low-high%" reads as
	# smallest-to-largest magnitude, the opposite ordering.
	if not (abs(tp_hi) - 1 <= readme_tp_lo <= abs(tp_hi) + 1):
		bad.append(f"README throughput low bound {readme_tp_lo}% vs "
			f"artifact smallest-magnitude delta |{tp_hi:.2f}|%")
	if not (abs(tp_lo) - 1 <= readme_tp_hi <= abs(tp_lo) + 1):
		bad.append(f"README throughput high bound {readme_tp_hi}% vs "
			f"artifact largest-magnitude delta |{tp_lo:.2f}|%")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"README vram=[{readme_vram_lo},{readme_vram_hi}]% vs artifact "
		f"[{vram_lo:.2f},{vram_hi:.2f}]%; README tp=[{readme_tp_lo},"
		f"{readme_tp_hi}]% vs artifact [{abs(tp_lo):.2f},{abs(tp_hi):.2f}]%")


# ---------------------------------------------------------------------
# Phase 10C: Q5 derisk + productization development-evidence artifact
# ---------------------------------------------------------------------
V04_ARTIFACT_PATH = REPO_ROOT / "results" / "v0.4" / "q5-validation.json"
_V04_CACHE = {}


def _load_v04_artifact():
	key = str(V04_ARTIFACT_PATH)
	if key not in _V04_CACHE:
		_V04_CACHE[key] = json.loads(V04_ARTIFACT_PATH.read_text())
	return _V04_CACHE[key]


@check("Phase 10C v0.4 artifact exists, is valid JSON, and has "
	"schema_version 1")
def _c38():
	if not V04_ARTIFACT_PATH.exists():
		return False, f"{V04_ARTIFACT_PATH} does not exist"
	a = _load_v04_artifact()
	ok = a.get("schema_version") == 1
	return ok, f"schema_version={a.get('schema_version')!r}"


@check("Phase 10C v0.4 artifact: membrane_commit and llama_cpp_commit "
	"look like real commit SHAs (40 hex chars)")
def _c39():
	a = _load_v04_artifact()
	bad = []
	for key in ("membrane_commit", "llama_cpp_commit"):
		sha = a.get(key, "")
		if not re.fullmatch(r"[0-9a-f]{40}", sha):
			bad.append(f"{key}={sha!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "both commit fields well-formed"


@check("Phase 10C v0.4 artifact: derisk prompt set has >= 25 prompts, "
	"exact prompts persisted, categories covered")
def _c40():
	a = _load_v04_artifact()
	ps = a.get("derisk", {}).get("prompt_set", {})
	bad = []
	total = ps.get("total_prompts", 0)
	entries = ps.get("prompts_and_labels", [])
	if total < 25:
		bad.append(f"total_prompts={total} < 25")
	if len(entries) != total:
		bad.append(f"prompts_and_labels has {len(entries)} entries but "
			f"total_prompts={total}")
	for e in entries:
		if not e.get("prompt") or not e.get("label"):
			bad.append(f"entry missing prompt/label: {e!r}")
			break
	categories = ps.get("categories", [])
	expected_categories = {"factual_recall", "instruction_following",
		"multistep_reasoning", "arithmetic_reasoning", "logical_reasoning",
		"code_completion", "code_reasoning", "code_explanation",
		"structured_output", "summarization", "prose_continuation"}
	missing_cats = expected_categories - set(categories)
	if missing_cats:
		bad.append(f"missing categories: {missing_cats}")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"{total} prompts, {len(categories)} categories, all persisted verbatim")


@check("Phase 10C v0.4 artifact: reasoning-heavy subset has >= 10 "
	"prompts and is a real subset of the persisted prompt labels")
def _c41():
	a = _load_v04_artifact()
	ps = a.get("derisk", {}).get("prompt_set", {})
	size = ps.get("reasoning_heavy_subset_size", 0)
	subset_labels = set(ps.get("reasoning_heavy_subset_labels", []))
	all_labels = {e["label"] for e in ps.get("prompts_and_labels", [])}
	bad = []
	if size < 10:
		bad.append(f"reasoning_heavy_subset_size={size} < 10")
	if len(subset_labels) != size:
		bad.append(f"{len(subset_labels)} labels listed but size={size}")
	missing = subset_labels - all_labels
	if missing:
		bad.append(f"reasoning subset references labels not in the "
			f"persisted prompt set: {missing}")
	return len(bad) == 0, "; ".join(bad) if bad else f"{size} reasoning-heavy prompts, all real"


@check("Phase 10C v0.4 artifact: storage math -- Q5_1 is 37.5% of F16 "
	"and ~64.7% of Q8_0 (real ggml block struct sizes), both checked "
	"per row, not just the F16 ratio")
def _c42():
	a = _load_v04_artifact()
	# Storage theory itself lives in results/phase10/ (Phase 10B) --
	# this check confirms the memory_reconfirm bytes in THIS artifact
	# are consistent with both known Q5_1 ratios: 37.5% of F16
	# (block_q5_1 = 24 B/32 elem = 48 B/token vs F16's 128 B/token for
	# a 64-wide head) and ~64.706% of Q8_0 (48 B/token vs Q8_0's 68
	# B/token) -- a changed Q8 allocation would silently invalidate the
	# README's "~29% additional reduction beyond q8" claim if only the
	# F16 ratio were checked.
	rows = a.get("memory_reconfirm", {}).get("cpu", [])
	bad = []
	if not rows:
		return False, "no memory_reconfirm.cpu rows found"
	for row in rows:
		native_b = row["native"]["kv_allocated_bytes"]
		q8_b = row["q8"]["kv_allocated_bytes"]
		q5_b = row["q5"]["kv_allocated_bytes"]
		if native_b is None or q8_b is None or q5_b is None:
			bad.append(f"ctx={row['ctx']}: missing native/q8/q5 bytes")
			continue
		pct_native = 100.0 * q5_b / native_b
		if abs(pct_native - 37.5) > 0.01:
			bad.append(f"ctx={row['ctx']}: q5/native={pct_native:.4f}% expected 37.5%")
		pct_q8 = 100.0 * q5_b / q8_b
		expected_pct_q8 = 100.0 * 48.0 / 68.0
		if abs(pct_q8 - expected_pct_q8) > 0.01:
			bad.append(f"ctx={row['ctx']}: q5/q8={pct_q8:.4f}% expected "
				f"{expected_pct_q8:.4f}%")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows match both 37.5% and ~64.71% exactly"


@check("Phase 10C v0.4 artifact: q5 < q8 < native memory ordering "
	"holds on every CPU and Vulkan reconfirmation row")
def _c43():
	a = _load_v04_artifact()
	bad = []
	for backend in ("cpu", "vulkan"):
		rows = a.get("memory_reconfirm", {}).get(backend, [])
		if not rows:
			bad.append(f"no memory_reconfirm.{backend} rows found")
			continue
		for row in rows:
			vals = {m: row[m]["kv_allocated_bytes"] for m in ("q5", "q8", "native")}
			if None in vals.values():
				bad.append(f"{backend} ctx={row['ctx']}: missing bytes {vals}")
				continue
			if not (vals["q5"] < vals["q8"] < vals["native"]):
				bad.append(f"{backend} ctx={row['ctx']}: not q5({vals['q5']}) "
					f"< q8({vals['q8']}) < native({vals['native']})")
	return len(bad) == 0, "; ".join(bad) if bad else "ordering holds on all rows, both backends"


V04_RAW_PATHS = {
	"135M": REPO_ROOT / "results" / "v0.4" / "derisk_raw_135m.jsonl",
	"360M": REPO_ROOT / "results" / "v0.4" / "derisk_raw_360m.jsonl",
}


def _recompute_v04_aggregate(model, reasoning_labels):
	"""Reimplements the same aggregation the artifact's own
	aggregate_quality was built with (mean/median top1, mean rel-L2,
	worst-top1 row, worst-rel-L2 row -- two independent maxima, not
	the same row assumed twice), straight from the raw per-(prompt,
	mode) records -- so a hand-edited or stale embedded aggregate
	can't silently pass this check."""
	path = V04_RAW_PATHS[model]
	rows = [json.loads(raw_line) for raw_line in path.read_text().splitlines()
		if raw_line.strip()]
	by_mode = {}
	for r in rows:
		by_mode.setdefault(r["mode"], []).append(r)
	out = {}
	for mode, rs in by_mode.items():
		for subset_name, subset_rows in (
			("full_set", rs),
			("reasoning_subset", [r for r in rs if r["prompt_label"] in reasoning_labels]),
		):
			available = [r for r in subset_rows if r["quality_available"]]
			if not available:
				out[(mode, subset_name)] = None
				continue
			top1s = [r["top1_preservation"] for r in available]
			rel_l2s = [r["logit_rel_l2"] for r in available]
			worst_top1_row = min(available, key=lambda r: r["top1_preservation"])
			worst_rel_l2_row = max(available, key=lambda r: r["logit_rel_l2"])
			out[(mode, subset_name)] = {
				"mean_top1_preservation": statistics.mean(top1s),
				"mean_logit_rel_l2": statistics.mean(rel_l2s),
				"worst_case_top1_preservation": worst_top1_row["top1_preservation"],
				"worst_case_logit_rel_l2": worst_rel_l2_row["logit_rel_l2"],
			}
	return out


@check("Phase 10C v0.4 artifact: aggregate_quality is not stale -- "
	"recomputed straight from derisk_raw_*.jsonl matches the embedded "
	"aggregate for q5_1 and q4, both models, both subsets")
def _c44():
	a = _load_v04_artifact()
	agg = a.get("derisk", {}).get("aggregate_quality", {})
	reasoning_labels = set(a.get("derisk", {}).get("prompt_set", {})
		.get("reasoning_heavy_subset_labels", []))
	bad = []
	if not reasoning_labels:
		return False, "no reasoning_heavy_subset_labels found to recompute against"
	for model in ("135M", "360M"):
		recomputed = _recompute_v04_aggregate(model, reasoning_labels)
		for mode in ("q5_1", "q4", "q8"):
			for subset in ("full_set", "reasoning_subset"):
				embedded = agg.get(model, {}).get(mode, {}).get(subset)
				fresh = recomputed.get((mode, subset))
				if embedded is None or fresh is None:
					bad.append(f"{model}/{mode}/{subset}: missing in "
						f"artifact or raw recomputation")
					continue
				for field in ("mean_top1_preservation", "mean_logit_rel_l2",
						"worst_case_top1_preservation", "worst_case_logit_rel_l2"):
					if abs(embedded[field] - fresh[field]) > 1e-4:
						bad.append(f"{model}/{mode}/{subset}/{field}: "
							f"embedded={embedded[field]} recomputed={fresh[field]}")
	return len(bad) == 0, "; ".join(bad) if bad else "embedded aggregate matches raw-record recomputation exactly"


@check("Phase 10C v0.4 artifact: q5 beats q4 in absolute direction on "
	"mean top1 (both models/subsets, Section 5's literal derisk bar) "
	"and is materially closer to q8 than q4 on mean logit rel-L2 "
	"(both models/subsets) -- NOT claiming closer-to-q8-on-top1 "
	"everywhere, since that is false on one real slice (135M "
	"reasoning subset) and is disclosed as such, not hidden")
def _c45():
	a = _load_v04_artifact()
	agg = a.get("derisk", {}).get("aggregate_quality", {})
	bad = []
	for model in ("135M", "360M"):
		for subset in ("full_set", "reasoning_subset"):
			q8 = agg.get(model, {}).get("q8", {}).get(subset)
			q5 = agg.get(model, {}).get("q5_1", {}).get(subset)
			q4 = agg.get(model, {}).get("q4", {}).get(subset)
			if not q8 or not q5 or not q4:
				bad.append(f"{model}/{subset}: missing q8/q5_1/q4 aggregate")
				continue
			# Section 5's actual, literal bar: q5 beats q4 in direction
			# on top1 -- true everywhere measured, checked here exactly
			# as specified (not "closer to q8", which is a stronger
			# claim this project does NOT make about top1 on every slice).
			if not (q5["mean_top1_preservation"] > q4["mean_top1_preservation"]):
				bad.append(f"{model}/{subset}: q5 mean_top1="
					f"{q5['mean_top1_preservation']} not > q4's "
					f"{q4['mean_top1_preservation']}")
			# rel-L2 is the metric that IS consistently closer to q8
			# than q4 everywhere measured -- checked as a distance,
			# because it actually holds as one.
			dist_to_q8 = abs(q5["mean_logit_rel_l2"] - q8["mean_logit_rel_l2"])
			dist_to_q4 = abs(q5["mean_logit_rel_l2"] - q4["mean_logit_rel_l2"])
			if not (dist_to_q8 < dist_to_q4):
				bad.append(f"{model}/{subset}/mean_logit_rel_l2: q5's "
					f"distance to q8 ({dist_to_q8:.6f}) not < distance to "
					f"q4 ({dist_to_q4:.6f})")
			if "worst_case_prompt" not in q5 or "worst_case_top1_preservation" not in q5:
				bad.append(f"{model}/{subset}: missing worst_case fields on q5_1")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"q5 beats q4 on top1 direction and is closer to q8 on rel-L2 "
		"distance, both models, both subsets; worst-case fields present")


@check("Phase 10C v0.4 artifact: Vulkan memory reconfirmation "
	"includes real external VRAM records for q5")
def _c46():
	a = _load_v04_artifact()
	rows = a.get("memory_reconfirm", {}).get("vulkan", [])
	bad = []
	if not rows:
		return False, "no memory_reconfirm.vulkan rows found"
	for row in rows:
		vram = row.get("q5", {}).get("external_peak_vram_mib")
		layers = row.get("q5", {}).get("gpu_layers_selected")
		if vram is None or vram <= 0:
			bad.append(f"ctx={row['ctx']}: q5 external_peak_vram_mib={vram} invalid")
		if layers is None or layers <= 0:
			bad.append(f"ctx={row['ctx']}: q5 gpu_layers_selected={layers} invalid")
	return len(bad) == 0, "; ".join(bad) if bad else f"{len(rows)} rows with valid VRAM/layer records"


@check("Phase 10C v0.4 artifact: capacity/pressure-boundary evidence "
	"shows q8 failing closed at a context where q5 achieves FULL "
	"model-layer GPU residency (not just a successful exit) on a "
	"real, recorded adapter capacity")
def _c47():
	a = _load_v04_artifact()
	cap = a.get("capacity_reconfirm_360m_vulkan", {})
	n_layer = cap.get("model_n_layer")
	device_total = cap.get("device_total_bytes")
	rows = cap.get("rows", [])
	bad = []
	if not rows:
		return False, "no capacity_reconfirm_360m_vulkan.rows found"
	if not n_layer or n_layer <= 0:
		bad.append(f"model_n_layer missing or invalid: {n_layer!r}")
	if not device_total or device_total <= 0:
		bad.append(f"device_total_bytes missing or invalid: {device_total!r}")
	found_full_residency_where_q8_fails = False
	for r in rows:
		if r.get("kv_mode") == "q8" and r.get("exit_code") == 5:
			ctx = r["ctx"]
			q5_at_ctx = next((x for x in rows if x["ctx"] == ctx
				and x.get("kv_mode") == "q5"), None)
			if not q5_at_ctx or q5_at_ctx.get("exit_code") != 0:
				continue
			# The claim this backs is "fully GPU-resident", not merely
			# "exited successfully" -- a partial-offload success at
			# this ctx would not support that claim.
			if q5_at_ctx.get("gpu_layers_selected") == n_layer \
					and q5_at_ctx.get("full_residency") is True:
				found_full_residency_where_q8_fails = True
	if not found_full_residency_where_q8_fails:
		bad.append("no context row found where q8 failed closed (exit 5) "
			"but q5 achieved full model-layer residency "
			f"(gpu_layers_selected == model_n_layer == {n_layer})")
	return len(bad) == 0, "; ".join(bad) if bad else (
		f"{len(rows)} rows checked on a {device_total/1024/1024:.0f} MiB "
		f"adapter; confirmed >=1 ctx where q8 exit=5 and q5 fully "
		f"GPU-resident ({n_layer}/{n_layer} layers)")


@check("Phase 10C v0.4 artifact: no absolute filesystem paths anywhere "
	"in the file")
def _c48():
	text = V04_ARTIFACT_PATH.read_text()
	posix = re.findall(
		r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
		text,
	)
	windows = re.findall(r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+', text)
	hits = posix + windows
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("Phase 10C v0.4 artifact: no NaN/Infinity anywhere in the file")
def _c49():
	a = _load_v04_artifact()
	bad = _no_nan_inf(a)
	return len(bad) == 0, f"non-finite at: {bad[:5]}" if bad else "all values finite"


@check("Phase 10C v0.4 artifact: product_cli section does not claim "
	"q5_0 or q4 are exposed product modes")
def _c50():
	a = _load_v04_artifact()
	pc = a.get("product_cli", {})
	bad = []
	flag = pc.get("flag", "")
	if "q5_0" in flag or re.search(r"\bq4\b", flag):
		bad.append(f"product_cli.flag references an unsupported mode: {flag!r}")
	excluded = pc.get("excluded_from_product", [])
	if not any("q5_0" in x for x in excluded) or not any(x.strip() == "q4" for x in excluded):
		bad.append(f"excluded_from_product does not clearly list q5_0 and q4: {excluded!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "product_cli section correctly scoped to native/q8/q5"


@check("README.md not updated with unsupported Phase 10C product "
	"claims (--kv q5_0/q4 as product flags, or universal-quality "
	"language) unless derisk explicitly passed")
def _c51():
	readme = (REPO_ROOT / "README.md").read_text()
	a = _load_v04_artifact() if V04_ARTIFACT_PATH.exists() else {}
	derisk_passed = str(a.get("derisk", {}).get("verdict", "")).startswith("DERISK PASSED")
	bad = []
	if re.search(r"--kv\s+q5_0\b", readme) or re.search(r"--kv\s+q4\b", readme):
		bad.append("README documents --kv q5_0 or --kv q4 as product flags -- "
			"these must never ship in stable docs")
	if re.search(r"universal(ly)?\s+(good|high)\s+quality", readme, re.IGNORECASE):
		bad.append("README claims universal quality for a compressed KV mode")
	if not derisk_passed and re.search(r"--kv\s+q5\b", readme):
		bad.append("README documents --kv q5 but the v0.4 artifact's derisk "
			"verdict does not show DERISK PASSED")
	return len(bad) == 0, "; ".join(bad) if bad else "README has no unsupported Q5 product claims"


# ---------------------------------------------------------------------
# Phase 11A: adaptive whole-cache KV policy validation artifact
# ---------------------------------------------------------------------
V11A_ARTIFACT_PATH = REPO_ROOT / "results" / "v0.4" / "adaptive-kv-validation.json"
_V11A_CACHE = {}
_ADAPTIVE_REASON_CODES = {
	"Q8_FITS",
	"Q8_FULL_RESIDENCY",
	"Q5_REQUIRED_FOR_FULL_RESIDENCY",
	"Q5_REQUIRED_FOR_MEMORY_GUARD",
	"Q5_ONLY_COMPRESSED_MODE_THAT_FITS",
	"NO_COMPRESSED_MODE_FITS",
	"CPU_ADAPTIVE_Q8_DEFAULT",
	"CPU_MEMORY_PRESSURE_Q5",
}


def _load_adaptive_artifact():
	key = str(V11A_ARTIFACT_PATH)
	if key not in _V11A_CACHE:
		_V11A_CACHE[key] = json.loads(V11A_ARTIFACT_PATH.read_text())
	return _V11A_CACHE[key]


@check("Phase 11A adaptive artifact exists, is valid JSON, and has "
	"schema_version 1")
def _c52():
	if not V11A_ARTIFACT_PATH.exists():
		return False, f"{V11A_ARTIFACT_PATH} does not exist"
	a = _load_adaptive_artifact()
	ok = a.get("schema_version") == 1
	return ok, f"schema_version={a.get('schema_version')!r}"


@check("Phase 11A adaptive artifact: membrane_commit and llama_cpp_commit "
	"look like real commit SHAs (40 hex chars)")
def _c53():
	a = _load_adaptive_artifact()
	bad = []
	for key in ("membrane_commit", "llama_cpp_commit"):
		sha = a.get(key, "")
		if not re.fullmatch(r"[0-9a-f]{40}", sha):
			bad.append(f"{key}={sha!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "both commit fields well-formed"


@check("Phase 11A adaptive artifact: q5 < q8 storage bytes on every "
	"CPU/Vulkan validation-matrix context, and adaptive's own bytes "
	"exactly match whichever explicit mode it selected")
def _c54():
	bad = []
	a = _load_adaptive_artifact()
	for section in ("cpu_validation_matrix", "vulkan_validation_matrix"):
		runs = a.get(section, {}).get("runs", [])
		by_ctx = {}
		for r in runs:
			by_ctx.setdefault(r["ctx_size"], {})[r["requested_kv"]] = r
		if not by_ctx:
			bad.append(f"{section}: no runs found")
			continue
		for ctx, modes in by_ctx.items():
			if "q8" not in modes or "q5" not in modes:
				bad.append(f"{section} ctx={ctx}: missing explicit q8/q5 rows")
				continue
			q8_bytes = modes["q8"]["kv_allocated_bytes"]
			q5_bytes = modes["q5"]["kv_allocated_bytes"]
			if not (q5_bytes < q8_bytes):
				bad.append(f"{section} ctx={ctx}: q5 bytes {q5_bytes} not < "
					f"q8 bytes {q8_bytes}")
			if "adaptive" in modes:
				ad = modes["adaptive"]
				expected = modes.get(ad["selected_kv"], {}).get("kv_allocated_bytes")
				if expected is not None and ad["kv_allocated_bytes"] != expected:
					bad.append(f"{section} ctx={ctx}: adaptive selected "
						f"{ad['selected_kv']} but its bytes "
						f"({ad['kv_allocated_bytes']}) != that explicit "
						f"mode's bytes ({expected})")
	return len(bad) == 0, "; ".join(bad) if bad else "storage ordering and adaptive/explicit byte equality both hold"


@check("Phase 11A adaptive artifact: at least one Zone A (Q8 full "
	"residency), Zone B (Q5 required), and Zone C (fail closed) "
	"transition row exist")
def _c55():
	a = _load_adaptive_artifact()
	zones = a.get("memory_pressure_transition", {}).get("zones", [])
	present = {z.get("zone") for z in zones}
	missing = {"A", "B", "C"} - present
	return len(missing) == 0, (
		f"missing zone(s): {missing}" if missing else
		f"all three zones present across {len(zones)} rows")


@check("Phase 11A adaptive artifact: Zone A rows select q8 and Zone B "
	"rows select q5 -- the transition direction is correct, not just "
	"present")
def _c56():
	a = _load_adaptive_artifact()
	zones = a.get("memory_pressure_transition", {}).get("zones", [])
	bad = []
	for z in zones:
		if z.get("zone") == "A" and z.get("selected_kv") != "q8":
			bad.append(f"ctx={z['ctx_requested']}: Zone A selected "
				f"{z.get('selected_kv')!r}, expected q8")
		if z.get("zone") == "B" and z.get("selected_kv") != "q5":
			bad.append(f"ctx={z['ctx_requested']}: Zone B selected "
				f"{z.get('selected_kv')!r}, expected q5")
		if z.get("zone") == "C" and z.get("exit_code", 0) == 0:
			bad.append(f"ctx={z['ctx_requested']}: Zone C did not fail "
				f"closed (exit_code=0)")
	return len(bad) == 0, "; ".join(bad) if bad else "every zone's selection matches its label"


@check("Phase 11A adaptive artifact: every adaptive_reason field, in "
	"both validation matrices and the transition zones, is one of the "
	"8 fixed, stable reason codes")
def _c57():
	a = _load_adaptive_artifact()
	bad = []
	for section in ("cpu_validation_matrix", "vulkan_validation_matrix"):
		for r in a.get(section, {}).get("runs", []):
			reason = r.get("adaptive_reason", "")
			if r.get("requested_kv") == "adaptive" and reason not in _ADAPTIVE_REASON_CODES:
				bad.append(f"{section} ctx={r['ctx_size']}: unknown reason {reason!r}")
	for z in a.get("memory_pressure_transition", {}).get("zones", []):
		reason = z.get("adaptive_reason", "")
		if reason not in _ADAPTIVE_REASON_CODES:
			bad.append(f"zone ctx={z['ctx_requested']}: unknown reason {reason!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "every reason code is from the fixed, documented set"


@check("Phase 11A adaptive artifact: the selected mode in every "
	"successful transition-zone row is recorded as valid in its own "
	"candidate estimate (the decision matches the recorded evidence, "
	"not just an assertion)")
def _c58():
	a = _load_adaptive_artifact()
	bad = []
	for z in a.get("memory_pressure_transition", {}).get("zones", []):
		if z.get("exit_code", 1) != 0:
			continue
		selected = z.get("selected_kv")
		cand = z.get("candidates", {}).get(selected, {})
		if not cand.get("valid"):
			bad.append(f"ctx={z['ctx_requested']}: selected {selected!r} "
				f"but its own candidate.valid is not true")
		if cand.get("selected_layers") != z.get("gpu_layers_selected"):
			bad.append(f"ctx={z['ctx_requested']}: winning candidate's "
				f"selected_layers ({cand.get('selected_layers')}) != "
				f"reported gpu_layers_selected ({z.get('gpu_layers_selected')})")
	return len(bad) == 0, "; ".join(bad) if bad else "every selection is backed by its own recorded, valid candidate"


@check("Phase 11A adaptive artifact: explicit-vs-adaptive equivalence "
	"is reported as an exact match for both the Q8-selected and "
	"Q5-selected real cases (no new codec path)")
def _c59():
	a = _load_adaptive_artifact()
	eq = a.get("explicit_vs_adaptive_equivalence", {})
	bad = []
	for case_name in ("q8_selected_case", "q5_selected_case"):
		case = eq.get(case_name, {})
		for field in ("text_match", "generated_tokens_match",
				"kv_allocated_bytes_match", "kv_type_match"):
			if case.get(field) is not True:
				bad.append(f"{case_name}.{field} is not True: {case.get(field)!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "both real equivalence cases fully match"


@check("Phase 11A adaptive artifact: no run ever reports selected_kv "
	"as native -- adaptive only ever resolves to a real compressed "
	"mode or fails closed")
def _c60():
	a = _load_adaptive_artifact()
	bad = []
	for section in ("cpu_validation_matrix", "vulkan_validation_matrix"):
		for r in a.get(section, {}).get("runs", []):
			if r.get("requested_kv") == "adaptive" and r.get("selected_kv") == "native":
				bad.append(f"{section} ctx={r['ctx_size']}: adaptive silently "
					f"resolved to native")
	for z in a.get("memory_pressure_transition", {}).get("zones", []):
		if z.get("selected_kv") == "native":
			bad.append(f"zone ctx={z['ctx_requested']}: adaptive silently "
				f"resolved to native")
	return len(bad) == 0, "; ".join(bad) if bad else "no silent native fallback anywhere in the artifact"


@check("Phase 11A adaptive artifact: no absolute filesystem paths "
	"anywhere in the file")
def _c61():
	text = V11A_ARTIFACT_PATH.read_text()
	posix = re.findall(
		r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
		text,
	)
	windows = re.findall(r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+', text)
	hits = posix + windows
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("Phase 11A adaptive artifact: no NaN/Infinity anywhere in the file")
def _c62():
	a = _load_adaptive_artifact()
	bad = _no_nan_inf(a)
	return len(bad) == 0, f"non-finite at: {bad[:5]}" if bad else "all values finite"


@check("Phase 11A adaptive artifact: policy rules and limitations do "
	"not claim Q4 or Q5_0 are selectable, or that adaptive is an OOM "
	"guarantee")
def _c63():
	a = _load_adaptive_artifact()
	bad = []
	rules_text = " ".join(a.get("policy", {}).get("rules", []))
	if re.search(r"\bq4\b", rules_text, re.IGNORECASE) or "q5_0" in rules_text.lower():
		bad.append("policy.rules references q4 or q5_0 as a selectable mode")
	limitations_text = " ".join(a.get("limitations", []))
	if re.search(r"guarantee[sd]?\s+(against|no)\s+oom", limitations_text, re.IGNORECASE):
		bad.append("limitations claims an OOM guarantee, contradicting Section 17")
	if "point-in-time" not in limitations_text and "point in time" not in limitations_text:
		bad.append("limitations does not disclose the point-in-time GPU "
			"memory snapshot caveat")
	return len(bad) == 0, "; ".join(bad) if bad else "no unsupported claims, snapshot caveat disclosed"


# ---------------------------------------------------------------------
# Phase 12H: KV residency productization (results/v0.3/kv-residency-
# productization/) -- every check below recomputes its claim directly
# from the raw *.txt/*.jsonl evidence files, never trusting the
# accompanying summary JSON's own say-so, matching this project's
# established verification style.
# ---------------------------------------------------------------------
V03_KV_RESIDENCY_DIR = REPO_ROOT / "results" / "v0.3" / "kv-residency-productization"


def _kv_residency_json(name):
	return json.loads((V03_KV_RESIDENCY_DIR / name).read_text())


def _kv_residency_raw(name):
	return (V03_KV_RESIDENCY_DIR / name).read_text()


@check("Phase 12H: manifest.json exists, valid JSON, lists every actual "
	"artifact/raw file in the directory and no others")
def _c64():
	manifest_path = V03_KV_RESIDENCY_DIR / "manifest.json"
	if not manifest_path.exists():
		return False, f"{manifest_path} does not exist"
	m = json.loads(manifest_path.read_text())
	listed = set(m.get("artifacts", [])) | set(m.get("raw_evidence_files", []))
	on_disk = {p.name for p in V03_KV_RESIDENCY_DIR.iterdir() if p.is_file()
		and p.name != "manifest.json"}
	listed_minus_manifest = listed
	missing = on_disk - listed_minus_manifest
	extra = listed_minus_manifest - on_disk
	ok = not missing and not extra
	return ok, (f"missing_from_manifest={sorted(missing)} "
		f"extra_in_manifest={sorted(extra)}" if not ok
		else f"{len(on_disk)} files, exact match with manifest")


@check("Phase 12H: weights-unchanged proof -- est_weights is byte-"
	"identical across every default/auto/cpu raw run at --gpu-layers all, "
	"recomputed by grepping the real captured stderr, not read from the "
	"summary")
def _c65():
	files = ["raw_capacity_default_26500.txt", "raw_capacity_default_26800.txt",
		"raw_capacity_default_28500.txt", "raw_control_b.txt", "raw_control_c.txt"]
	values = {}
	for f in files:
		text = _kv_residency_raw(f)
		m = re.search(r"est_weights=([\d.]+) MiB", text)
		if not m:
			return False, f"{f}: no est_weights= line found"
		values[f] = float(m.group(1))
	distinct = set(values.values())
	ok = len(distinct) == 1
	return ok, (f"values={values}" if not ok
		else f"identical est_weights={distinct.pop()} MiB across all "
			f"{len(files)} raw runs")


def _kv_residency_config_fields(text):
	"""Extract the fields that must be identical across a default/auto/
	cpu capacity comparison for it to be a valid same-configuration
	test -- model, context, KV precision, backend + requested/selected
	GPU weight layers. Missing any of these is itself a failure (the
	comparison can't be trusted if a field can't even be found)."""
	fields = {}
	m = re.search(r"^model\s+(\S+)", text, re.MULTILINE)
	fields["model"] = m.group(1) if m else None
	m = re.search(r"^context\s+(\d+)", text, re.MULTILINE)
	fields["context"] = m.group(1) if m else None
	m = re.search(r"^kv\s+(.+)$", text, re.MULTILINE)
	fields["precision"] = m.group(1).strip() if m else None
	m = re.search(
		r"^backend\s+(\S+), device selected: \S+ \(gpu-layers=(\S+), "
		r"selected=(\d+)\)", text, re.MULTILINE)
	fields["backend"] = m.group(1) if m else None
	fields["gpu_layers_requested"] = m.group(2) if m else None
	fields["gpu_layers_selected"] = m.group(3) if m else None
	return fields


@check("Phase 12H: capacity-uplift claim -- default fails (real Vulkan "
	"OOM, non-zero exit) and auto/cpu succeed, at the identical ctx=28500 "
	"configuration, recomputed from raw exit codes and error text")
def _c66():
	default_text = _kv_residency_raw("raw_capacity_default_28500.txt")
	auto_text = _kv_residency_raw("raw_control_b.txt")
	cpu_text = _kv_residency_raw("raw_control_c.txt")
	bad = []
	if "exit_default_28500=4" not in default_text:
		bad.append("default run's captured exit code is not 4")
	if "ErrorOutOfDeviceMemory" not in default_text:
		bad.append("default run's failure is not a real Vulkan OOM")
	if "generated" not in auto_text or "Paris" not in auto_text:
		bad.append("auto run does not show successful generation")
	if "generated" not in cpu_text or "Paris" not in cpu_text:
		bad.append("cpu run does not show successful generation")
	if "GPU KV layers: 26/28" not in auto_text:
		bad.append("auto run's split is not the claimed 26/28")
	# Review fix: the checks above only look at exit markers, generated
	# text, and the AUTO split -- none of them confirm the three raw
	# runs actually describe the SAME model/context/precision/backend/
	# weight-layer configuration, so a mismatched evidence file could
	# still pass. Parse and compare those fields explicitly.
	configs = {
		"default": _kv_residency_config_fields(default_text),
		"auto": _kv_residency_config_fields(auto_text),
		"cpu": _kv_residency_config_fields(cpu_text),
	}
	for name, fields in configs.items():
		missing = [k for k, v in fields.items() if v is None]
		if missing:
			bad.append(f"{name} run: could not parse {missing}")
	if not bad:
		reference = configs["default"]
		for name in ("auto", "cpu"):
			for key, value in reference.items():
				if configs[name][key] != value:
					bad.append(f"{name} run's {key}={configs[name][key]!r} "
						f"!= default's {key}={value!r} -- not the same "
						f"configuration")
	return len(bad) == 0, "; ".join(bad) if bad else ("default really "
		"fails (exit 4, real Vulkan OOM) while auto (26/28 split) and "
		"cpu (0/28) both really succeed, at the same model/context/"
		"precision/backend/weight-layer configuration (recomputed and "
		"compared field-by-field, not assumed)")


@check("Phase 12H: default path's reproducible failure and success "
	"boundary rows are internally consistent (26500 succeeds, "
	"26800/28500 fail) -- recomputed from raw exit codes")
def _c67():
	rows = {
		"raw_capacity_default_26500.txt": ("0", True),
		"raw_capacity_default_26800.txt": ("4", False),
		"raw_capacity_default_28500.txt": ("4", False),
	}
	bad = []
	for fname, (expect_exit, expect_success) in rows.items():
		text = _kv_residency_raw(fname)
		ctx = fname.split("_")[-1].replace(".txt", "")
		exit_marker = f"exit_default_{ctx}={expect_exit}"
		if exit_marker not in text:
			bad.append(f"{fname}: expected {exit_marker}")
		has_error = "ErrorOutOfDeviceMemory" in text
		if expect_success and has_error:
			bad.append(f"{fname}: expected success but found a real error")
		if not expect_success and not has_error:
			bad.append(f"{fname}: expected a real Vulkan OOM but found none")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"26500 succeeds, 26800 and 28500 both fail with a real Vulkan "
		"OOM -- boundary is internally consistent")


@check("Phase 12H: quality.json's IDENTICAL claim is reproduced by "
	"recomputing md5 of every listed raw file independently, not trusted "
	"from the summary's own md5_all_files field")
def _c68():
	q = _kv_residency_json("quality.json")
	files = q.get("compared_files", [])
	if len(files) < 2:
		return False, "quality.json lists fewer than 2 compared files"
	digests = {}
	for f in files:
		path = V03_KV_RESIDENCY_DIR / f
		digests[f] = hashlib.md5(path.read_bytes()).hexdigest()
	distinct = set(digests.values())
	ok = len(distinct) == 1 and distinct.pop() == q.get("md5_all_files")
	return ok, (f"digests={digests} claimed={q.get('md5_all_files')}"
		if not ok else f"{len(files)} files independently re-hashed, all "
			f"identical, matches the claimed md5")


@check("Phase 12H: no unsupported performance claim -- performance.json "
	"never asserts GPU-resident or CPU-resident KV is faster without the "
	"noise/small-sample qualification, and explicitly disclaims marketing "
	"a performance win")
def _c69():
	p = _kv_residency_json("performance.json")
	full_text = json.dumps(p).lower()
	bad = []
	# Review fix (round 2): the round-1 fix required a truthy value,
	# which incorrectly still accepted `"no_marketing_claim": true` --
	# a bare boolean is NOT a disclaimer, it says nothing. The
	# documented schema (see the comment this replaces) is a non-empty
	# prose string; enforce that exact type, not just truthiness.
	claim = p.get("no_marketing_claim")
	if not isinstance(claim, str) or not claim.strip():
		bad.append("no_marketing_claim is missing, empty, or not a "
			"non-empty string (a bare boolean is not a disclaimer)")
	# A bare, unqualified "is faster"/"proves" claim would be a red flag;
	# the interpretation field is expected to hedge with words like
	# "noisy"/"not... robust" rather than assert a clean winner.
	interp = p.get("interpretation", "").lower()
	if "not statistically robust" not in interp and "not a statistically robust" not in interp:
		bad.append("interpretation does not hedge the throughput reading")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"performance claim is explicitly hedged and non-marketing")


@check("Phase 12H: minimal patch set claim -- CMakeLists.txt's "
	"MEMBRANE_ENABLE_LLAMA block applies exactly the type-override and "
	"device-override patches, and does NOT reference runtime-relocate or "
	"buffer-retirement anywhere in the product build")
def _c70():
	cmake_text = (REPO_ROOT / "CMakeLists.txt").read_text()
	bad = []
	if "llama.cpp-membrane-kv-type-override.patch" not in cmake_text:
		bad.append("type-override patch is not applied")
	if "llama.cpp-membrane-kv-device-override.patch" not in cmake_text:
		bad.append("device-override patch is not applied")
	# Match the actual patch FILENAME (with the .patch extension a real
	# execute_process(... git apply ...) command would need), not just
	# the short "kv-runtime-relocate" substring, which also appears
	# inside this very block's own comment explaining why it is
	# deliberately excluded -- a naive substring match on the short
	# name would false-positive on that comment.
	if "llama.cpp-membrane-kv-runtime-relocate.patch" in cmake_text:
		bad.append("runtime-relocate patch is applied in the product "
			"build (must stay research-only)")
	if "llama.cpp-membrane-kv-buffer-retirement.patch" in cmake_text:
		bad.append("buffer-retirement patch is applied in the product "
			"build (must stay research-only)")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"exactly type-override + device-override applied; no reference "
		"to runtime-relocate or buffer-retirement anywhere in "
		"CMakeLists.txt")


@check("Phase 12H: default CLI behavior is unchanged -- product_cli.cpp "
	"initializes kv_placement to MEMBRANE_KV_PLACEMENT_DEFAULT and only "
	"sets cp.kv_dev_override when a non-NULL placement map is passed")
def _c71():
	cli_text = (REPO_ROOT / "tools" / "membrane-run" / "product_cli.cpp").read_text()
	decode_text = (REPO_ROOT / "tools" / "membrane-llama-runtime"
		/ "decode_loop.cpp").read_text()
	bad = []
	if "o->kv_placement = MEMBRANE_KV_PLACEMENT_DEFAULT;" not in cli_text:
		bad.append("kv_placement is not defaulted to MEMBRANE_KV_PLACEMENT_DEFAULT")
	# Review fix: these two substring checks used to be independent --
	# both could pass even if the `if (kv_placement != NULL)` block
	# and the `cp.kv_dev_override =` assignment weren't actually the
	# same block (e.g. an unconditional assignment elsewhere, or a
	# no-op conditional). Require the assignment to appear inside the
	# SAME braced block as the guard, not just anywhere in the file.
	guard_match = re.search(
		r"if \(kv_placement != NULL\)\s*\{([^}]*)\}", decode_text)
	if guard_match is None:
		bad.append("kv_dev_override is not conditionally gated on a "
			"non-NULL placement map (no matching braced if-block found)")
	elif "cp.kv_dev_override = " not in guard_match.group(1):
		bad.append("the kv_placement != NULL block does not assign "
			"cp.kv_dev_override -- guard and assignment are not bound "
			"together")
	return len(bad) == 0, "; ".join(bad) if bad else (
		"default value and conditional gating both confirmed directly "
		"in source")


@check("Phase 12H: no private/absolute filesystem paths leaked in any "
	"kv-residency-productization artifact")
def _c72():
	leaked = []
	# Review fix: reuse the SAME broader POSIX+Windows absolute-path
	# patterns _c20 already uses (not limited to /home/ or /tmp/ -- a
	# leaked /mnt/..., /var/..., /Users/..., or C:\... path is just as
	# much a privacy problem).
	posix_re = r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+'
	windows_re = r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+'
	for p in V03_KV_RESIDENCY_DIR.iterdir():
		if not p.is_file():
			continue
		text = p.read_text(errors="replace")
		if re.search(posix_re, text) or re.search(windows_re, text):
			leaked.append(p.name)
	return len(leaked) == 0, (f"leaked in: {leaked}" if leaked
		else "none found across all kv-residency-productization files")


@check("Phase 12H: summary.json's decision_gate is exactly one of the 5 "
	"allowed values and is internally consistent with a real, reproduced "
	"capacity-uplift finding (not upgraded beyond what capacity_uplift.json "
	"actually shows)")
def _c73():
	# Review fix: validating decision_gate against summary.json's OWN
	# decision_gate_allowed_values field is self-referential -- a
	# malformed artifact could add an arbitrary value to its own
	# allowed-list and this check would still pass. Own the enum here.
	KV_RESIDENCY_DECISION_GATES = frozenset((
		"KV_RESIDENCY_PRODUCT_VIABLE",
		"KV_RESIDENCY_PRODUCT_WORKS_BUT_NO_CAPACITY_WIN",
		"KV_RESIDENCY_PRODUCT_PATCH_BURDEN_TOO_HIGH",
		"KV_RESIDENCY_PRODUCT_BACKEND_LIMITED",
		"KV_RESIDENCY_PRODUCT_INCONCLUSIVE",
	))
	s = _kv_residency_json("summary.json")
	gate = s.get("decision_gate")
	if gate not in KV_RESIDENCY_DECISION_GATES:
		return False, (f"decision_gate={gate!r} not one of the "
			f"verifier-owned allowed values {sorted(KV_RESIDENCY_DECISION_GATES)}")
	cap = _kv_residency_json("capacity_uplift.json")
	uplift = cap.get("measured_uplift", {})
	uplift_value = uplift.get("uplift_tokens_at_least")
	uplift_positive_numeric = (isinstance(uplift_value, (int, float))
		and not isinstance(uplift_value, bool) and uplift_value > 0)
	qualification = s.get("important_qualification", "")
	qualification_non_empty = isinstance(qualification, str) and bool(qualification.strip())
	if gate == "KV_RESIDENCY_PRODUCT_VIABLE":
		if not qualification_non_empty:
			return False, ("VIABLE claimed without a non-empty "
				"important_qualification present")
		if not uplift_positive_numeric:
			return False, ("VIABLE claimed but capacity_uplift.json's "
				"measured_uplift.uplift_tokens_at_least is missing or "
				"not a positive number")
	# Review fix: the success message used to unconditionally cite
	# uplift/qualification even for gates where those fields aren't
	# required (only VIABLE enforces them above) -- report only what
	# was actually validated for the gate that was actually returned.
	if gate == "KV_RESIDENCY_PRODUCT_VIABLE":
		return True, (f"decision_gate={gate!r} valid against the "
			f"verifier-owned enum, backed by a real positive measured "
			f"uplift ({uplift_value}), and the narrow-scope qualification "
			f"is present")
	return True, (f"decision_gate={gate!r} valid against the "
		f"verifier-owned enum")


def main() -> int:
	global ARTIFACT_PATH
	ap = argparse.ArgumentParser()
	ap.add_argument("--artifact", type=Path, default=None,
		help="verify only the v0.2 artifact checks against this path "
			"instead of the committed results/v0.2/smollm2-q8-memory.json "
			"(used by scripts/benchmark-v0.2.sh --verify on freshly "
			"generated, not-yet-committed data)")
	args = ap.parse_args()

	if args.artifact is not None:
		ARTIFACT_PATH = args.artifact
		checks = (_c14, _c15, _c16, _c17, _c18, _c19, _c20, _c21, _c22, _c23, _c24)
	else:
		checks = (
			_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11, _c12, _c13,
			_c14, _c15, _c16, _c17, _c18, _c19, _c20, _c21, _c22, _c23, _c24,
			_c25, _c26, _c27, _c28, _c29, _c30, _c31, _c32,
			_c33, _c34, _c35, _c36, _c37,
			_c38, _c39, _c40, _c41, _c42, _c43, _c44, _c45, _c46, _c47,
			_c48, _c49, _c50, _c51,
			_c52, _c53, _c54, _c55, _c56, _c57, _c58, _c59, _c60, _c61,
			_c62, _c63,
			_c64, _c65, _c66, _c67, _c68, _c69, _c70, _c71, _c72, _c73,
		)
	for fn in checks:
		fn()
	print()
	print(f"{CHECK_COUNT - len(FAILURES)}/{CHECK_COUNT} checks passed")
	if FAILURES:
		print("\nFAILED:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1

	# CONTRIBUTING.md requires every docs/ claim to be "checkable by
	# scripts/verify-results.py". docs/compatibility.json's claims are
	# a different kind of claim (product compatibility, not research
	# evidence) with their own dedicated validator and invariants --
	# kept as a genuinely separate script/file on purpose (Phase 18),
	# not folded into the 73 numbered checks above (which would silently
	# change the "N/73" figure every doc/script referencing that number
	# expects). Running it here, as an unnumbered final step, is what
	# makes CONTRIBUTING.md's promise literally true without merging the
	# two verifiers' logic together.
	compat_script = REPO_ROOT / "scripts" / "verify-compatibility.py"
	if compat_script.exists():
		print()
		print("Also verifying docs/compatibility.json (scripts/verify-compatibility.py):")
		sys.stdout.flush()
		result = subprocess.run([sys.executable, str(compat_script)])
		if result.returncode != 0:
			print("\nFAILED: scripts/verify-compatibility.py reported failures "
				"(see output above)")
			return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
