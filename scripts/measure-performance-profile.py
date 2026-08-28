#!/usr/bin/env python3
"""Reproduces the committed results/performance-profiling/measurements.json
artifact: runs membrane-run at a small, fixed, controlled set of points
(real local GGUF models, no downloads), 3 repeats each, and records the
"timings"/"throughput" fields its own --json output already reports
(Phase 24's own additive instrumentation -- see docs/performance-profiling.md).

This is a measurement/observation script -- it does NOT optimize anything
(Phase 24's own explicit "profiling, not optimization" scope).

Usage:
  scripts/measure-performance-profile.py [OUTPUT_JSON]

MODEL_PATHs are hardcoded to the fixed local fixtures this repo already
uses elsewhere (models/SmolLM2-*.gguf, models/qwen2.5-1.5b-*.gguf) --
never downloaded here. OUTPUT_JSON defaults to a scratch path; it is
NEVER the committed results/performance-profiling/measurements.json unless
you pass that path explicitly.

Environment:
  MEMBRANE_RUN_BIN   path to a membrane-run binary built with
                     -DGGML_VULKAN=ON (so both CPU and Vulkan points
                     below can run from one binary)
                     (default: ./build-vulkan/tools/membrane-run/membrane-run)
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPEATS = 3
DEFAULT_CTX = 2048
DEFAULT_GEN_TOKENS = 32

# Fixed, small, controlled point set (Section 7 of the Phase 24 task: 8-12
# points total, not a sweep). Every point uses ctx=2048/gen-tokens=32
# unless noted.
POINTS = [
	{
		"label": "smollm2-135m_vulkan_native",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "native",
		"kv_placement": None, "gpu_layers": "all",
	},
	{
		"label": "smollm2-135m_vulkan_q8",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "q8",
		"kv_placement": None, "gpu_layers": "all",
	},
	{
		"label": "smollm2-135m_vulkan_q5",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "q5",
		"kv_placement": None, "gpu_layers": "all",
	},
	{
		"label": "smollm2-135m_vulkan_adaptive",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "adaptive",
		"kv_placement": None, "gpu_layers": "all",
	},
	{
		"label": "smollm2-135m_cpu_native",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "cpu", "kv": "native",
		"kv_placement": None, "gpu_layers": "0",
	},
	{
		"label": "smollm2-135m_cpu_q8",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "cpu", "kv": "q8",
		"kv_placement": None, "gpu_layers": "0",
	},
	{
		"label": "smollm2-360m_vulkan_native",
		"model": "models/SmolLM2-360M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "native",
		"kv_placement": None, "gpu_layers": "all",
	},
	{
		"label": "qwen2.5-1.5b_vulkan_native_gpu_placement",
		"model": "models/qwen2.5-1.5b-instruct-fp16.gguf",
		"model_arch": "qwen2", "backend": "vulkan", "kv": "native",
		"kv_placement": "gpu", "gpu_layers": "all",
	},
	{
		"label": "qwen2.5-1.5b_vulkan_native_cpu_placement",
		"model": "models/qwen2.5-1.5b-instruct-fp16.gguf",
		"model_arch": "qwen2", "backend": "vulkan", "kv": "native",
		"kv_placement": "cpu", "gpu_layers": "all",
	},
	{
		# Real substitute/supplementary point (Section 29: never force a
		# point past real, reproducible host resource pressure) -- see
		# docs/performance-profiling.md's own note on the two GPU-weight
		# points above, which failed closed via AUTO_FALLBACK_EXHAUSTED
		# on this specific run (real, reproducible transient VRAM
		# pressure on this shared host, not a MEMBRANE defect). This
		# gpu_layers=0 point needs no GPU memory at all, so it still
		# gives real large-model CPU timing data even when the GPU
		# points above cannot complete.
		"label": "qwen2.5-1.5b_cpu_native",
		"model": "models/qwen2.5-1.5b-instruct-fp16.gguf",
		"model_arch": "qwen2", "backend": "cpu", "kv": "native",
		"kv_placement": None, "gpu_layers": "0",
		"gen_tokens": 16,
	},
	{
		# Section 14: a representative PARTIAL offload point -- "0" and
		# "all" are already covered by the CPU/Vulkan-native points
		# above (reused, not duplicated); this is the one new "half"
		# data point. SmolLM2-135M has 30 real layers (see
		# docs/joint-planner.md's own SmolLM2-135M citation).
		"label": "smollm2-135m_vulkan_partial_half_15",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": "native",
		"kv_placement": None, "gpu_layers": "15",
	},
	{
		# Section 15: isolates planner/runtime overhead -- compare
		# directly against smollm2-135m_vulkan_q8 above (--auto resolves
		# to the identical physical configuration on this host: 30/30
		# layers, q8, GPU-resident KV -- verified in
		# docs/performance-profiling.md, not assumed).
		"label": "smollm2-135m_vulkan_auto",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_arch": "llama", "backend": "vulkan", "kv": None,
		"kv_placement": None, "gpu_layers": None, "auto": True,
	},
]


def run_once(bin_path, point):
	args = ["--model", point["model"], "--prompt",
		"The capital of France is", "--ctx", str(point.get("ctx", DEFAULT_CTX)),
		"--gen-tokens", str(point.get("gen_tokens", DEFAULT_GEN_TOKENS)),
		"--json", "--quiet"]
	if point.get("auto"):
		args += ["--auto"]
	else:
		args += ["--gpu-layers", point["gpu_layers"], "--kv", point["kv"]]
		if point["kv_placement"]:
			args += ["--kv-placement", point["kv_placement"]]
	proc = subprocess.run([bin_path] + args, capture_output=True, text=True,
		cwd=str(REPO_ROOT))
	try:
		parsed = json.loads(proc.stdout)
	except json.JSONDecodeError:
		parsed = None
	logical_cmd = ["membrane-run"] + args
	return (proc.returncode, parsed, logical_cmd)


def median_min_max(values):
	values = [v for v in values if v is not None]
	if not values:
		return {"median": None, "min": None, "max": None, "raw": []}
	return {
		"median": round(statistics.median(values), 3),
		"min": round(min(values), 3),
		"max": round(max(values), 3),
		"raw": [round(v, 3) for v in values],
	}


TIMING_FIELDS = ("total_ms", "planner_ms", "model_load_ms", "tokenization_ms",
	"context_create_ms", "prefill_ms", "decode_ms", "first_token_ms")
THROUGHPUT_FIELDS = ("prefill_tokens_per_second", "decode_tokens_per_second")


def build_measurement(point, bin_path, membrane_commit):
	raw_runs = []
	logical_cmd = None
	device = None
	resolved = {}
	ok_count = 0
	last_failure_reason = None
	# CodeRabbit review (PR #34): captured defensively in case a failed
	# run's own JSON ever carries a "fallback" object -- verified during
	# this fix that main.cpp's print_error_json() (the schema an
	# AUTO_FALLBACK_EXHAUSTED failure actually uses) does NOT include one
	# today, so this stays empty for every currently-observed failure
	# mode; documented as a real, disclosed JSON-schema gap in
	# docs/performance-profiling.md rather than silently assumed
	# resolved by adding this field.
	failed_fallback_attempted = []
	for rep in range(REPEATS):
		returncode, parsed, logical_cmd = run_once(bin_path, point)
		if returncode == 0 and parsed is not None and parsed.get("ok"):
			ok_count += 1
			raw_runs.append(parsed)
			device = parsed.get("gpu", {}).get("device_selected") or device
			resolved = {
				"gpu_layers": parsed.get("gpu", {}).get("gpu_layers_selected"),
				"kv_precision": parsed.get("resolved", {}).get("kv"),
				"kv_placement": parsed.get("resolved", {}).get("kv_placement"),
				"fallback_attempted": parsed.get("fallback", {}).get("attempted"),
			}
		else:
			if parsed is not None and isinstance(parsed.get("error"), dict):
				last_failure_reason = parsed["error"].get("reason_code")
			if parsed is not None and isinstance(parsed.get("fallback"), dict):
				failed_fallback_attempted.append(
					parsed["fallback"].get("attempted"))
			print(f"    rep {rep + 1}/{REPEATS}: FAILED (exit {returncode}"
				f"{f', {last_failure_reason}' if last_failure_reason else ''})",
				file=sys.stderr)

	if failed_fallback_attempted:
		resolved["fallback_attempted_on_failed_runs"] = failed_fallback_attempted

	timings_agg = {}
	for field in TIMING_FIELDS:
		timings_agg[field] = median_min_max(
			[r["timings"].get(field) for r in raw_runs])
	throughput_agg = {}
	for field in THROUGHPUT_FIELDS:
		throughput_agg[field] = median_min_max(
			[r["timings"]["throughput"].get(field) for r in raw_runs])
	# CodeRabbit review (PR #34): real token counts per successful run,
	# so the validator can recompute throughput from
	# tokens/(stage_ms/1000) and cross-check it against the reported
	# value -- a hand-edited artifact can no longer swap in an
	# arbitrary-but-internally-consistent throughput number undetected.
	token_counts = {
		"prompt_tokens": [r.get("execution", {}).get("prompt_tokens")
			for r in raw_runs],
		"generated_tokens": [r.get("execution", {}).get("generated_tokens")
			for r in raw_runs],
	}

	return {
		"schema_version": 1,
		"label": point["label"],
		"commit": membrane_commit,
		"backend": point["backend"],
		"device": device,
		"model": Path(point["model"]).name,
		"model_arch": point["model_arch"],
		"context": point.get("ctx", DEFAULT_CTX),
		"gen_tokens_requested": point.get("gen_tokens", DEFAULT_GEN_TOKENS),
		"request": {
			"gpu_layers": point.get("gpu_layers"),
			"kv_precision": point.get("kv"),
			"kv_placement": point.get("kv_placement") or "default",
			"auto": bool(point.get("auto")),
		},
		"resolved": resolved,
		"repeats": REPEATS,
		"ok_count": ok_count,
		"token_counts": token_counts,
		"note": None if ok_count > 0 else (
			f"0/{REPEATS} repeats succeeded -- last observed failure "
			f"reason_code={last_failure_reason!r}. See "
			f"docs/performance-profiling.md for this specific point's "
			f"own real, disclosed root-cause discussion (never silently "
			f"omitted, never a fabricated measurement)."),
		"timings_ms": timings_agg,
		"throughput": throughput_agg,
		"provenance": {
			"command": " ".join(logical_cmd) if logical_cmd else None,
			"measurement_method": "membrane-run --json's own \"timings\" "
				"object (Phase 24 instrumentation, CLOCK_MONOTONIC via "
				"decode_loop.cpp's seconds_since()), median/min/max over "
				f"{REPEATS} repeats",
		},
	}


def main():
	out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
		"/tmp/membrane-performance-profile.json")
	bin_path = os.environ.get(
		"MEMBRANE_RUN_BIN", "./build-vulkan/tools/membrane-run/membrane-run")
	if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
		print(f"error: membrane-run binary not found/executable at "
			f"{bin_path}", file=sys.stderr)
		print("build it first: cmake --build build-vulkan "
			"--target membrane-run", file=sys.stderr)
		return 1

	# CodeRabbit review (PR #34): a missing fixture used to be silently
	# skipped, letting an incomplete matrix pass as if it were the full
	# committed 12-point set (the validator only checked "non-empty").
	# Preflight ALL fixtures before running anything, and fail closed --
	# no partial artifact is ever written.
	missing = [point["model"] for point in POINTS
		if not (REPO_ROOT / point["model"]).is_file()]
	if missing:
		print("error: required model fixture(s) missing, refusing to "
			"write a partial artifact:", file=sys.stderr)
		for rel in sorted(set(missing)):
			print(f"  - {REPO_ROOT / rel}", file=sys.stderr)
		return 1

	commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
		capture_output=True, text=True, check=True).stdout.strip()
	run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

	measurements = []
	for point in POINTS:
		print(f"=== {point['label']} ({REPEATS} repeats) ===", file=sys.stderr)
		m = build_measurement(point, bin_path, commit)
		measurements.append(m)
		med_total = m["timings_ms"]["total_ms"]["median"]
		med_decode = m["throughput"]["decode_tokens_per_second"]["median"]
		print(f"    ok={m['ok_count']}/{REPEATS} total_ms(median)="
			f"{med_total} decode_tok_s(median)={med_decode}", file=sys.stderr)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps({
		"schema_version": 1,
		"note": "Phase 24: inference performance profiling at a small, "
			"fixed set of controlled points, 3 repeats each. Profiling "
			"only -- no optimization was performed in this phase. See "
			"docs/performance-profiling.md.",
		"membrane_commit": commit,
		"generated_at": run_timestamp,
		"host": {"note": "generated on the developer machine that ran "
			"scripts/measure-performance-profile.py -- see each "
			"measurement's own device/backend field for the exact "
			"tested hardware; never generalize a device-scoped number."},
		"measurements": measurements,
	}, indent=2) + "\n")
	print(f"wrote {len(measurements)} measurements to {out_path}",
		file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
