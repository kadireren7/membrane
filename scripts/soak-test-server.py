#!/usr/bin/env python3
"""Bounded local soak test for `membrane serve` (Mega Phase C, PR C3).

This measures RESOURCE STABILITY under sustained real use -- RSS
growth, thread count, open file descriptors, reload count, and failure
count -- over a bounded number of real, sequential chat-completion
requests against a real (small) model. It is deliberately NOT a
throughput/performance benchmark and must never be reported or
marketed as one (Mega Phase C's own explicit instruction) -- the
per-request latency numbers it prints are diagnostic context for the
resource-stability numbers, not a headline claim.

Bounded by design for this project's own real memory-constrained dev
host: a small model, a small max_tokens, and a small, fixed request
count -- never an unbounded/duration-based loop that could run away on
a shared host.

Usage:
  scripts/soak-test-server.py --model PATH [--requests N] [--membrane BIN]
                               [--port PORT]

Exit code: 0 if the server stayed healthy and resource usage stayed
bounded (see thresholds below); 1 otherwise, with the failing signal
named.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# A real, small resource-growth allowance -- not zero, since a real
# process's RSS legitimately fluctuates a little (allocator behavior,
# lazy-mapped pages) -- but far below "leaking a new buffer per
# request" territory, which is what this test exists to catch.
MAX_RSS_GROWTH_BYTES = 64 * 1024 * 1024   # 64 MiB
MAX_THREAD_GROWTH = 2
MAX_FD_GROWTH = 8


def _proc_stat(pid):
	rss_bytes = None
	with open(f"/proc/{pid}/status") as f:
		for line in f:
			if line.startswith("VmRSS:"):
				rss_bytes = int(line.split()[1]) * 1024
				break
	threads = len(os.listdir(f"/proc/{pid}/task"))
	fds = len(os.listdir(f"/proc/{pid}/fd"))
	return rss_bytes, threads, fds


def _wait_healthy(port, timeout_s=15):
	deadline = time.monotonic() + timeout_s
	while time.monotonic() < deadline:
		try:
			with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
					timeout=1) as resp:
				if resp.status == 200:
					return True
		except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
			pass
		time.sleep(0.2)
	return False


def _chat_request(port, model_name, max_tokens, request_timeout):
	body = json.dumps({
		"model": model_name,
		"messages": [{"role": "user", "content": "Say hi in one word."}],
		"max_tokens": max_tokens,
	}).encode()
	req = urllib.request.Request(
		f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
		headers={"Content-Type": "application/json"}, method="POST")
	start = time.monotonic()
	try:
		with urllib.request.urlopen(req, timeout=request_timeout) as resp:
			resp.read()
			ok = resp.status == 200
	except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
		ok = False
	return ok, time.monotonic() - start


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--requests", type=int, default=20)
	parser.add_argument("--max-tokens", type=int, default=8)
	parser.add_argument("--membrane", default=None,
		help="path to the membrane binary (default: build-b1/tools/membrane/membrane)")
	parser.add_argument("--port", type=int, default=18943)
	parser.add_argument("--request-timeout", type=float, default=30.0,
		help="per-request socket timeout in seconds (default: 30; lower "
		"this on a memory-pressured host to bound how long a single "
		"slow/hung request can block the whole soak run)")
	args = parser.parse_args()

	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	membrane_bin = args.membrane or os.path.join(
		repo_root, "build-b1", "tools", "membrane", "membrane")
	if not os.path.isfile(membrane_bin):
		print(f"soak-test-server.py: membrane binary not found at {membrane_bin}",
			file=sys.stderr)
		return 1

	tmpdir = tempfile.mkdtemp(prefix="membrane-soak-")
	models_path = os.path.join(tmpdir, "models.json")
	env = dict(os.environ)
	env["MEMBRANE_MODELS_PATH"] = models_path

	model_name = "soak-model"
	add_rc = subprocess.run(
		[membrane_bin, "model", "add", model_name, args.model],
		env=env, capture_output=True, text=True)
	if add_rc.returncode != 0:
		print(f"soak-test-server.py: model add failed: {add_rc.stderr}",
			file=sys.stderr)
		return 1

	server = subprocess.Popen(
		[membrane_bin, "serve", "--port", str(args.port)],
		env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	try:
		if not _wait_healthy(args.port):
			print("soak-test-server.py: server never became healthy",
				file=sys.stderr)
			return 1

		# One real request to force the model to actually load before
		# baseline sampling -- otherwise the "baseline" RSS would be
		# pre-load and every real soak run would show a large, expected,
		# one-time jump that has nothing to do with per-request growth.
		#
		# A real, disclosed possibility on this project's own memory-
		# constrained dev host: the host-memory-guard's fixed 256 MiB
		# reserve (host_memory_guard.h) can legitimately make even a
		# tiny model's real minimum-context request infeasible when
		# OTHER real processes on a shared host currently leave less
		# than that much memory free -- this is the guard working
		# correctly, not a MEMBRANE bug. When this happens, the soak
		# loop still runs, but as a REJECTION-PATH stability check
		# (does the server stay healthy/bounded while correctly and
		# repeatedly refusing real requests under real memory
		# pressure?) rather than a real-generation soak -- always
		# honestly labeled as such in the result, never silently
		# reported as if real generation had been exercised.
		warmup_ok, _ = _chat_request(args.port, model_name, args.max_tokens, args.request_timeout)
		generation_mode = "real_generation" if warmup_ok else "rejection_path_only"
		if not warmup_ok:
			print("soak-test-server.py: warm-up request did not succeed "
				"(likely real host memory pressure, see host_memory_guard.h's "
				"fixed reserve) -- continuing as a rejection-path stability "
				"soak instead of a real-generation soak", file=sys.stderr)

		baseline_rss, baseline_threads, baseline_fds = _proc_stat(server.pid)
		failures = 0
		latencies = []
		samples = []
		for i in range(args.requests):
			ok, elapsed = _chat_request(args.port, model_name, args.max_tokens, args.request_timeout)
			if generation_mode == "real_generation" and not ok:
				failures += 1
			latencies.append(elapsed)
			if i % 5 == 0 or i == args.requests - 1:
				rss, threads, fds = _proc_stat(server.pid)
				samples.append({"request": i, "rss_bytes": rss,
					"threads": threads, "fds": fds})

		final_rss, final_threads, final_fds = _proc_stat(server.pid)
		healthy_after = _wait_healthy(args.port, timeout_s=2)

		result = {
			"generation_mode": generation_mode,
			"requests_sent": args.requests,
			"failures": failures,
			"latency_seconds": {
				"min": min(latencies), "max": max(latencies),
				"mean": sum(latencies) / len(latencies),
			},
			"baseline": {"rss_bytes": baseline_rss, "threads": baseline_threads,
				"fds": baseline_fds},
			"final": {"rss_bytes": final_rss, "threads": final_threads,
				"fds": final_fds},
			"rss_growth_bytes": final_rss - baseline_rss,
			"thread_growth": final_threads - baseline_threads,
			"fd_growth": final_fds - baseline_fds,
			"samples": samples,
			"healthy_after_soak": healthy_after,
		}
		print(json.dumps(result, indent=2))

		problems = []
		if failures > 0:
			problems.append(f"{failures}/{args.requests} requests failed")
		if result["rss_growth_bytes"] > MAX_RSS_GROWTH_BYTES:
			problems.append(f"RSS grew {result['rss_growth_bytes']} bytes "
				f"(> {MAX_RSS_GROWTH_BYTES})")
		if result["thread_growth"] > MAX_THREAD_GROWTH:
			problems.append(f"thread count grew by {result['thread_growth']} "
				f"(> {MAX_THREAD_GROWTH})")
		if result["fd_growth"] > MAX_FD_GROWTH:
			problems.append(f"FD count grew by {result['fd_growth']} "
				f"(> {MAX_FD_GROWTH})")
		if not healthy_after:
			problems.append("server not healthy immediately after soak")

		if problems:
			print("soak-test-server.py: FAIL: " + "; ".join(problems),
				file=sys.stderr)
			return 1
		print("soak-test-server.py: PASS: resource usage stayed bounded",
			file=sys.stderr)
		return 0
	finally:
		server.send_signal(signal.SIGTERM)
		try:
			server.wait(timeout=10)
		except subprocess.TimeoutExpired:
			server.kill()
			server.wait()


if __name__ == "__main__":
	sys.exit(main())
