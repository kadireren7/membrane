#!/usr/bin/env python3
"""Milestone G3 -- Planner v2 real-workload evidence harness.

Runs a small, bounded, reproducible set of REAL `membrane plan` and
REAL `membrane-run` invocations against real local GGUF files, on this
real (memory-constrained, 5.6 GiB RAM) dev host, and records structured
JSON evidence -- see docs/planner-v2-evidence.md for what this is and
is not, and results/planner-v2-evidence/validation.json for the
generated artifact this script produces.

Design constraints (Part 2/19 of the G3 task):
  - No broad sweep: a small, fixed point set, not a parametric grid.
  - Short prompt, short generation length, low repetition count.
  - CPU (build-cpu-rc1) always; Vulkan (build-vulkan) only if that
    binary + a real Vulkan device are present, and only a handful of
    points, never a sweep.
  - Never downloads or quantizes a model itself (that already happened,
    once, as a disclosed manual step -- see docs/planner-v2-evidence.md
    "Models used").
  - An "infeasible" real-load attempt is bounded with `ulimit -v` and a
    hard timeout so a genuine OOM cannot destabilize the shared host.

This is a measurement script. It does not modify the planner, the
resolver, or the selection policy.
"""
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = REPO_ROOT / "models"
REG = REPO_ROOT / "scratch/g3-registry"

CPU_MEMBRANE = REPO_ROOT / "build-cpu-rc1/tools/membrane/membrane"
CPU_RUN = REPO_ROOT / "build-cpu-rc1/tools/membrane-run/membrane-run"
VK_MEMBRANE = REPO_ROOT / "build-vulkan/tools/membrane/membrane"
VK_RUN = REPO_ROOT / "build-vulkan/tools/membrane-run/membrane-run"

PROMPT = "The capital of France is"
GEN_TOKENS = 24
# Bound any real-load attempt to this much virtual memory so a real OOM
# on this 5.6 GiB shared host cannot destabilize other processes on it
# (Part 6 of the G3 task: "do not intentionally OOM the whole
# machine"). 1.5 GiB is comfortably above what any config in this
# harness needs to succeed, and comfortably below the host total.
ULIMIT_V_KB = 1536 * 1024


def read_meminfo():
	info = {}
	for line in Path("/proc/meminfo").read_text().splitlines():
		m = re.match(r"^(\w+):\s+(\d+)\s*kB", line)
		if m:
			info[m.group(1)] = int(m.group(2)) * 1024
	return {
		"total_bytes": info.get("MemTotal", 0),
		"available_bytes": info.get("MemAvailable", 0),
		"swap_total_bytes": info.get("SwapTotal", 0),
		"swap_free_bytes": info.get("SwapFree", 0),
	}


def read_gpu_meminfo():
	try:
		out = subprocess.run(
			["nvidia-smi", "--query-gpu=memory.total,memory.free,name",
			 "--format=csv,noheader,nounits"],
			capture_output=True, text=True, timeout=10,
		)
		if out.returncode != 0:
			return None
		total, free, name = [x.strip() for x in out.stdout.strip().split(",")]
		return {"device_name": name, "total_mib": int(total), "free_mib": int(free)}
	except Exception:
		return None


def _limit_virtual_memory():
	resource.setrlimit(resource.RLIMIT_AS, (ULIMIT_V_KB * 1024, ULIMIT_V_KB * 1024))


def run_membrane_plan(binary, model_name, extra_args=None, env=None):
	args = [str(binary), "plan", model_name, "--json"] + (extra_args or [])
	full_env = dict(os.environ)
	full_env["XDG_DATA_HOME"] = str(REG)
	if env:
		full_env.update(env)
	host_before = read_meminfo()
	t0 = time.time()
	proc = subprocess.run(args, capture_output=True, text=True, timeout=60, env=full_env)
	wall = time.time() - t0
	record = {
		"command": " ".join(args),
		"exit_code": proc.returncode,
		"wall_seconds": round(wall, 3),
		"host_available_bytes_before": host_before["available_bytes"],
		"stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else "",
	}
	try:
		record["json"] = json.loads(proc.stdout)
	except Exception:
		record["json"] = None
		record["stdout_tail"] = proc.stdout[-2000:]
	return record


def run_membrane_run(binary, model_path, ctx, gpu_layers, kv, gen_tokens=GEN_TOKENS,
			prompt=PROMPT, bound_memory=False, timeout=90, extra_args=None):
	args = [
		str(binary), "--model", str(model_path), "--prompt", prompt,
		"--ctx", str(ctx), "--gpu-layers", str(gpu_layers), "--kv", kv,
		"--gen-tokens", str(gen_tokens), "--json", "--quiet",
	] + (extra_args or [])
	host_before = read_meminfo()
	gpu_before = read_gpu_meminfo()
	t0 = time.time()
	try:
		proc = subprocess.run(
			args, capture_output=True, text=True, timeout=timeout,
			preexec_fn=_limit_virtual_memory if bound_memory else None,
		)
		timed_out = False
	except subprocess.TimeoutExpired as e:
		proc = type("P", (), {"returncode": -1, "stdout": "", "stderr": str(e)})()
		timed_out = True
	wall = time.time() - t0
	host_after = read_meminfo()
	gpu_after = read_gpu_meminfo()
	record = {
		"command": " ".join(args),
		"bounded_ulimit_v_mib": (ULIMIT_V_KB // 1024) if bound_memory else None,
		"exit_code": proc.returncode,
		"timed_out": timed_out,
		"wall_seconds": round(wall, 3),
		"host_available_bytes_before": host_before["available_bytes"],
		"host_available_bytes_after": host_after["available_bytes"],
		"gpu_before": gpu_before,
		"gpu_after": gpu_after,
		"stderr_tail": proc.stderr[-3000:] if proc.returncode != 0 else "",
	}
	try:
		record["json"] = json.loads(proc.stdout)
	except Exception:
		record["json"] = None
	return record


def main():
	results = {"generated_at_unix": int(time.time()), "commit": subprocess.run(
		["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
	).stdout.strip()}

	# ---- Part 6: planner decisions, CPU + Vulkan, two independent
	# snapshots minutes apart (host memory on this shared dev machine
	# genuinely fluctuates -- see docs/planner-v2-evidence.md).
	results["planner_decisions"] = {
		"cpu_snapshot_1": run_membrane_plan(CPU_MEMBRANE, "smollm2-135m-instruct"),
	}
	if VK_MEMBRANE.exists():
		results["planner_decisions"]["vulkan_snapshot_1"] = run_membrane_plan(
			VK_MEMBRANE, "smollm2-135m-instruct"
		)
	results["planner_decisions"]["cpu_360m_snapshot_1"] = run_membrane_plan(
		CPU_MEMBRANE, "smollm2-360m-instruct"
	)

	# ---- Parts 3/6/7/8: real runtime validation, CPU backend, family
	# smollm2-135m-instruct. A/B/C/D per Part 3 of the task.
	f16 = MODELS / "SmolLM2-135M-Instruct-f16.gguf"
	q8 = MODELS / "SmolLM2-135M-Instruct-Q8_0.gguf"
	q4 = MODELS / "SmolLM2-135M-Instruct-Q4_K_M.gguf"

	runtime_cpu = []
	REPS = 2

	def add(label, variant, path, ctx, gpu_layers, kv, bound_memory=False, reps=REPS):
		for i in range(reps):
			rec = run_membrane_run(CPU_RUN, path, ctx, gpu_layers, kv, bound_memory=bound_memory)
			rec["label"] = label
			rec["variant"] = variant
			rec["rep"] = i
			runtime_cpu.append(rec)

	# A. naive/default config: default ctx sizing (tiny, prompt+gen+8),
	# gpu-layers 0, kv native, on the F16 file as installed.
	for i in range(REPS):
		args = [str(CPU_RUN), "--model", str(f16), "--prompt", PROMPT,
			"--gen-tokens", str(GEN_TOKENS), "--gpu-layers", "0", "--json", "--quiet"]
		host_before = read_meminfo()
		t0 = time.time()
		proc = subprocess.run(args, capture_output=True, text=True, timeout=90)
		wall = time.time() - t0
		rec = {"command": " ".join(args), "exit_code": proc.returncode,
			"wall_seconds": round(wall, 3),
			"host_available_bytes_before": host_before["available_bytes"],
			"stderr_tail": proc.stderr[-3000:] if proc.returncode != 0 else ""}
		try:
			rec["json"] = json.loads(proc.stdout)
		except Exception:
			rec["json"] = None
		rec["label"] = "A_naive_default"
		rec["variant"] = "F16"
		rec["rep"] = i
		runtime_cpu.append(rec)

	# B. manual high-quality variant: explicit F16 at the planner's own
	# selected context (4096), so B and C differ ONLY in variant/quant,
	# isolating the effect of the planner's variant choice.
	add("B_manual_high_quality", "F16", f16, ctx=4096, gpu_layers=0, kv="native")

	# C. planner-selected configuration (Q8_0 @ ctx=4096, from the real
	# `membrane plan` run captured above).
	add("C_planner_selected", "Q8_0", q8, ctx=4096, gpu_layers=0, kv="native")

	# D. context-focused feasible alternative (Q4_K_M @ ctx=8192, the
	# G2 alternative with materially more feasible context).
	add("D_context_alternative", "Q4_K_M", q4, ctx=8192, gpu_layers=0, kv="native")

	# H2 / Part 6: bounded real-load attempt of the candidate the
	# planner marked infeasible (F16, HOST_MEMORY_LIMIT), at the same
	# ctx/gpu-layers/kv C used, under a hard virtual-memory ulimit so a
	# genuine OOM cannot destabilize the shared host.
	add("INFEASIBLE_CHECK_F16", "F16", f16, ctx=4096, gpu_layers=0, kv="native",
		bound_memory=True, reps=1)

	results["runtime_validation_cpu"] = runtime_cpu

	# ---- Part 10: small Vulkan evidence set, if a real Vulkan-capable
	# binary + device are present. Max 3 configs x 2 reps, one family.
	runtime_vulkan = []
	if VK_RUN.exists() and read_gpu_meminfo() is not None:
		vk_configs = [
			("VK_A_native_all_layers", "F16", f16, 2048, "all", "native"),
			("VK_B_q8_all_layers", "F16", f16, 2048, "all", "q8"),
			("VK_C_adaptive_all_layers", "Q8_0", q8, 4096, "all", "adaptive"),
		]
		for label, variant, path, ctx, gl, kv in vk_configs:
			for i in range(2):
				rec = run_membrane_run(VK_RUN, path, ctx, gl, kv)
				rec["label"] = label
				rec["variant"] = variant
				rec["rep"] = i
				runtime_vulkan.append(rec)
	results["runtime_validation_vulkan"] = runtime_vulkan
	results["vulkan_available"] = VK_RUN.exists() and read_gpu_meminfo() is not None

	out_path = REPO_ROOT / "scratch/evidence_harness_raw.json"
	out_path.write_text(json.dumps(results, indent=2))
	print(f"Wrote {out_path}", file=sys.stderr)
	print(json.dumps({"ok": True, "runtime_cpu_runs": len(runtime_cpu),
		"runtime_vulkan_runs": len(runtime_vulkan)}, indent=2))


if __name__ == "__main__":
	main()
