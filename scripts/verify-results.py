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
		r'(?<![\w.\-])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
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
		r'(?<![\w.\-])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+',
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
	return 0


if __name__ == "__main__":
	sys.exit(main())
