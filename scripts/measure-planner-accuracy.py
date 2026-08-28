#!/usr/bin/env python3
"""Reproduces the committed results/planner-accuracy/measurements.json
artifact: runs membrane-run at a small, fixed set of controlled points
(real local GGUF models, no downloads) and compares its PRE-LOAD memory
estimate (gpu_policy.estimated_model_bytes/estimated_kv_bytes in its own
--json output) against what actually happened (gpu_memory_observed's
post-run device re-query, and the memory.rss_after_*_kb checkpoints
membrane-run already reports).

This is a measurement/observation script -- it computes error metrics,
it does NOT tune or change the planner (see docs/planner-accuracy.md).

Usage:
  scripts/measure-planner-accuracy.py [OUTPUT_JSON]

MODEL_PATHs are hardcoded to the fixed local fixtures this repo already
uses elsewhere (models/SmolLM2-*.gguf, models/qwen2.5-1.5b-*.gguf) --
never downloaded here. OUTPUT_JSON defaults to a scratch path; it is
NEVER the committed results/planner-accuracy/measurements.json unless
you pass that path explicitly.

Environment:
  MEMBRANE_RUN_BIN   path to a membrane-run binary built with
                     -DGGML_VULKAN=ON (so both CPU and Vulkan points
                     below can run from one binary)
                     (default: ./build-vulkan/tools/membrane-run/membrane-run)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed, small, controlled point set (Section 19.7: 4-8 points, not a
# sweep). Every point uses the SAME ctx/gen_tokens unless noted, so
# differences in observed error trace back to precision/placement/model,
# not to varying the context.
POINTS = [
	{
		"label": "smollm2-135m_vulkan_native_default",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "vulkan", "kv": "native", "kv_placement": None,
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-135m_vulkan_q8_default",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "vulkan", "kv": "q8", "kv_placement": None,
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-135m_vulkan_q5_default",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "vulkan", "kv": "q5", "kv_placement": None,
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-135m_vulkan_adaptive_default",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "vulkan", "kv": "adaptive", "kv_placement": None,
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-135m_cpu_native",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "cpu", "kv": "native", "kv_placement": None,
		"gpu_layers": "0", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-135m_cpu_q8",
		"model": "models/SmolLM2-135M-Instruct-f16.gguf",
		"model_family": "SmolLM2-135M-Instruct", "model_arch": "llama",
		"backend": "cpu", "kv": "q8", "kv_placement": None,
		"gpu_layers": "0", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "smollm2-360m_vulkan_native_default",
		"model": "models/SmolLM2-360M-Instruct-f16.gguf",
		"model_family": "SmolLM2-360M-Instruct", "model_arch": "llama",
		"backend": "vulkan", "kv": "native", "kv_placement": None,
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
	{
		"label": "qwen2.5-1.5b_vulkan_native_cpu_placement",
		"model": "models/qwen2.5-1.5b-instruct-fp16.gguf",
		"model_family": "Qwen2.5-1.5B-Instruct", "model_arch": "qwen2",
		"backend": "vulkan", "kv": "native", "kv_placement": "cpu",
		"gpu_layers": "all", "ctx": 2048, "gen_tokens": 8,
	},
]


def run_point(bin_path, point):
	cmd = [bin_path, "--model", str(REPO_ROOT / point["model"]),
		"--prompt", "The capital of France is", "--ctx", str(point["ctx"]),
		"--gen-tokens", str(point["gen_tokens"]), "--json",
		"--gpu-layers", point["gpu_layers"], "--kv", point["kv"]]
	if point["kv_placement"]:
		cmd += ["--kv-placement", point["kv_placement"]]
	t0 = time.time()
	proc = subprocess.run(cmd, capture_output=True, text=True)
	wall_seconds = time.time() - t0
	try:
		parsed = json.loads(proc.stdout)
	except json.JSONDecodeError:
		parsed = None
	return (proc.returncode, parsed, proc.stderr, wall_seconds, cmd)


def classify_outcome(returncode, parsed):
	if returncode == 0 and parsed is not None and parsed.get("ok"):
		return "FIT"
	if returncode == 3:
		return "MODEL_LOAD_FAILED"
	if returncode == 4:
		return "CONTEXT_CREATION_OR_GENERATION_FAILED"
	if returncode == 5:
		return "BACKEND_ALLOCATION_OR_COMPAT_REJECTED"
	return "UNKNOWN_FAILURE"


def build_measurement(point, returncode, parsed, stderr, wall_seconds, cmd,
		membrane_commit):
	m = {
		"schema_version": 1,
		"label": point["label"],
		"timestamp": None,  # filled by caller with a single shared run timestamp
		"commit": membrane_commit,
		"model": Path(point["model"]).name,
		"model_arch": point["model_arch"],
		"backend": point["backend"],
		"device": None,
		"context": point["ctx"],
		"gpu_layers_requested": point["gpu_layers"],
		"gpu_layers_resolved": None,
		"kv_precision_requested": point["kv"],
		"kv_precision_resolved": None,
		"kv_placement_requested": point["kv_placement"] or "default",
		"kv_placement_resolved": None,
		"planner": {
			"available_gpu_bytes": None, "reserved_gpu_bytes": None,
			"estimated_weight_gpu_bytes": None, "estimated_kv_gpu_bytes": None,
			"estimated_total_gpu_bytes": None,
			"estimated_total_gpu_bytes_basis": None,
			"estimated_host_kv_bytes": None,
		},
		"observed": {
			"gpu_free_before_bytes": None, "gpu_free_after_bytes": None,
			"gpu_observed_delta_bytes": None,
			"host_rss_after_model_load_kb": None,
			"host_rss_after_context_kb": None,
			"host_rss_after_context_delta_kb": None,
			"host_rss_peak_kb": None,
		},
		"outcome": {
			"classification": classify_outcome(returncode, parsed),
			"exit_code": returncode, "wall_seconds": round(wall_seconds, 3),
		},
		"errors": {
			"gpu_estimate_error_bytes": None,
			"gpu_estimate_error_percent": None,
			"host_kv_estimate_error_kb": None,
			"host_kv_estimate_error_percent": None,
		},
		"provenance": {
			"command": " ".join(cmd),
			"measurement_method": "gpu: driver-reported free heap bytes via "
				"ggml_backend_dev_get_props(), read once before model load "
				"and once after generation (membrane-run's own "
				"gpu_policy/gpu_memory_observed JSON fields); host: /proc/"
				"self/status VmRSS via membrane-run's own memory.* JSON "
				"checkpoints. Neither is a true continuous peak sample.",
		},
	}
	if parsed is None:
		return m
	gp = parsed.get("gpu_policy") or {}
	obs = parsed.get("gpu_memory_observed") or {}
	mem = parsed.get("memory") or {}
	resolved = parsed.get("resolved") or {}
	m["device"] = resolved.get("device") or None
	m["gpu_layers_resolved"] = resolved.get("gpu_layers")
	m["kv_precision_resolved"] = parsed.get("selected_kv")
	m["kv_placement_resolved"] = resolved.get("kv_placement")

	placement = m["kv_placement_resolved"]
	weight_bytes = gp.get("estimated_model_bytes")
	kv_bytes = gp.get("estimated_kv_bytes")
	m["planner"]["available_gpu_bytes"] = gp.get("device_free_bytes")
	m["planner"]["reserved_gpu_bytes"] = gp.get("safety_reserve_bytes")
	m["planner"]["estimated_weight_gpu_bytes"] = weight_bytes
	m["planner"]["estimated_kv_gpu_bytes"] = kv_bytes
	m["planner"]["estimated_host_kv_bytes"] = (
		parsed.get("storage") or {}).get("kv_allocated_bytes")
	if weight_bytes is not None:
		if placement == "cpu":
			# KV was explicitly kept off the GPU -- the real-world GPU
			# footprint this plan implies is weights only. gpu_policy's
			# OWN budget check still reserves room for KV regardless
			# (a known, previously-documented gap -- see
			# docs/planner-accuracy.md), but that is a capacity-check
			# behavior, not what actually landed on the device.
			m["planner"]["estimated_total_gpu_bytes"] = weight_bytes
			m["planner"]["estimated_total_gpu_bytes_basis"] = (
				"weights_only (kv_placement=cpu)")
		elif kv_bytes is not None:
			m["planner"]["estimated_total_gpu_bytes"] = weight_bytes + kv_bytes
			m["planner"]["estimated_total_gpu_bytes_basis"] = "weights_plus_kv"

	if obs.get("available"):
		before = obs.get("device_free_bytes_before")
		after = obs.get("device_free_bytes_after")
		m["observed"]["gpu_free_before_bytes"] = before
		m["observed"]["gpu_free_after_bytes"] = after
		if before is not None and after is not None:
			m["observed"]["gpu_observed_delta_bytes"] = before - after

	m["observed"]["host_rss_after_model_load_kb"] = mem.get(
		"rss_after_model_load_kb")
	m["observed"]["host_rss_after_context_kb"] = mem.get("rss_after_context_kb")
	m["observed"]["host_rss_peak_kb"] = mem.get("peak_rss_kb")
	if (mem.get("rss_after_context_kb") is not None
			and mem.get("rss_after_model_load_kb") is not None):
		m["observed"]["host_rss_after_context_delta_kb"] = (
			mem["rss_after_context_kb"] - mem["rss_after_model_load_kb"])

	est_total = m["planner"]["estimated_total_gpu_bytes"]
	obs_delta = m["observed"]["gpu_observed_delta_bytes"]
	if est_total is not None and obs_delta is not None:
		err = obs_delta - est_total
		m["errors"]["gpu_estimate_error_bytes"] = err
		if obs_delta != 0:
			m["errors"]["gpu_estimate_error_percent"] = round(
				100.0 * err / obs_delta, 3)

	# Host-side KV estimate accuracy is only a clean, isolated comparison
	# when KV actually lives in host RAM for this run (CPU backend, or
	# GPU backend with kv_placement=cpu) -- otherwise the RSS delta after
	# context creation reflects llama.cpp/ggml host-side bookkeeping
	# buffers, not the KV cache itself, and comparing it to the KV-bytes
	# estimate would be misleading.
	kv_est_bytes = m["planner"]["estimated_host_kv_bytes"]
	rss_delta_kb = m["observed"]["host_rss_after_context_delta_kb"]
	kv_is_on_host = (point["backend"] == "cpu") or (placement == "cpu")
	if kv_is_on_host and kv_est_bytes is not None and rss_delta_kb is not None:
		kv_est_kb = kv_est_bytes / 1024.0
		err_kb = rss_delta_kb - kv_est_kb
		m["errors"]["host_kv_estimate_error_kb"] = round(err_kb, 1)
		if rss_delta_kb != 0:
			m["errors"]["host_kv_estimate_error_percent"] = round(
				100.0 * err_kb / rss_delta_kb, 3)
	return m


def main():
	out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
		"/tmp/membrane-planner-accuracy.json")
	bin_path = os.environ.get(
		"MEMBRANE_RUN_BIN", "./build-vulkan/tools/membrane-run/membrane-run")
	if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
		print(f"error: membrane-run binary not found/executable at "
			f"{bin_path}", file=sys.stderr)
		print("build it first: cmake --build build-vulkan "
			"--target membrane-run", file=sys.stderr)
		return 1

	commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
		capture_output=True, text=True, check=True).stdout.strip()
	run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

	measurements = []
	for point in POINTS:
		model_path = REPO_ROOT / point["model"]
		if not model_path.is_file():
			print(f"skipping {point['label']}: model not found at "
				f"{model_path}", file=sys.stderr)
			continue
		print(f"=== {point['label']} ===", file=sys.stderr)
		returncode, parsed, stderr, wall_seconds, cmd = run_point(
			bin_path, point)
		m = build_measurement(point, returncode, parsed, stderr,
			wall_seconds, cmd, commit)
		m["timestamp"] = run_timestamp
		measurements.append(m)
		print(f"    outcome={m['outcome']['classification']} "
			f"exit={returncode}", file=sys.stderr)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps({
		"schema_version": 1,
		"note": "Phase 19: planner memory-estimate-vs-observation "
			"measurements at a small, fixed set of controlled points. "
			"See docs/planner-accuracy.md.",
		"membrane_commit": commit,
		"generated_at": run_timestamp,
		"host": {"note": "generated on the developer machine that ran "
			"scripts/measure-planner-accuracy.py -- see gpu_policy's own "
			"device_total_bytes per measurement for the exact tested "
			"GPU's reported VRAM"},
		"measurements": measurements,
	}, indent=2) + "\n")
	print(f"wrote {len(measurements)} measurements to {out_path}",
		file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
