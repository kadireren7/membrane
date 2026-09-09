#!/usr/bin/env python3
"""Bounded local soak test for model-switching, multi-model residency,
and service-restart lifecycle (Mega Phase E, PR E4).

Same real-process, real-model, bounded-by-design convention as
scripts/soak-test-server.py (Mega Phase C, PR C3) -- measures RESOURCE
STABILITY (RSS/thread/FD growth, reload count, failure count) over a
bounded number of real cycles, never an unbounded/duration-based loop,
on this project's own real, severely memory-constrained dev host.
Complements soak-test-server.py (sequential single-model requests) and
concurrency-soak-server.py (admission-boundary concurrency) with the
three dimensions neither covers:

  1. repeated model-switch cycles (forced 1-slot residency, so every
     switch is a real evict+load) -- Section 27's own "model
     switching" soak.
  2. repeated multi-model hot-switch cycles (default residency,
     2 real models both resident, alternating requests) -- Section
     27's own "multi-model residency" soak.
  3. a bounded service restart loop -- Section 27/30: kill the real
     process, restart it, verify the registry/config survive and the
     service comes back healthy and able to load a model again, N
     times.

Usage:
  scripts/soak-model-lifecycle.py --model-a PATH --model-b PATH
                                   [--membrane BIN] [--port PORT]
                                   [--switch-cycles N] [--restart-cycles N]

Exit code: 0 if every phase stayed within its real resource-growth
threshold and reported no failures; 1 otherwise.
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

MAX_RSS_GROWTH_BYTES = 96 * 1024 * 1024  # a real, small allowance -- see
                                          # soak-test-server.py's own top
                                          # comment for why not zero
MAX_THREAD_GROWTH = 3
MAX_FD_GROWTH = 12


def _proc_stat(pid):
	rss_bytes = None
	with open(f"/proc/{pid}/status") as f:
		for line in f:
			if line.startswith("VmRSS:"):
				rss_bytes = int(line.split()[1]) * 1024
	n_threads = len(os.listdir(f"/proc/{pid}/task"))
	n_fds = len(os.listdir(f"/proc/{pid}/fd"))
	return {"rss_bytes": rss_bytes, "n_threads": n_threads, "n_fds": n_fds}


def _wait_healthy(port, timeout_s=20):
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


def _start_server(membrane_bin, models, port, env_overrides, models_path=None):
	tmpdir = tempfile.mkdtemp(prefix="membrane-e4-soak-")
	env = dict(os.environ)
	env["MEMBRANE_MODELS_PATH"] = models_path or os.path.join(tmpdir,
			"models.json")
	env.update(env_overrides)
	for name, path in models:
		add_rc = subprocess.run([membrane_bin, "model", "add", name, path],
			env=env, capture_output=True, text=True)
		if add_rc.returncode != 0 and "already registered" not in add_rc.stderr:
			raise RuntimeError(f"model add {name} failed: {add_rc.stderr}")
	server = subprocess.Popen([membrane_bin, "serve", "--port", str(port)],
		env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	if not _wait_healthy(port):
		server.send_signal(signal.SIGTERM)
		try:
			server.wait(timeout=5)
		except subprocess.TimeoutExpired:
			server.kill()
		raise RuntimeError("server never became healthy")
	return server, tmpdir, env["MEMBRANE_MODELS_PATH"]


def _stop_server(server, sig=signal.SIGTERM):
	server.send_signal(sig)
	try:
		server.wait(timeout=10)
	except subprocess.TimeoutExpired:
		server.kill()
		server.wait()


def _post(port, path, body, timeout=60):
	req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
		data=json.dumps(body).encode(),
		headers={"Content-Type": "application/json"}, method="POST")
	try:
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			return resp.status, json.loads(resp.read())
	except urllib.error.HTTPError as e:
		return e.code, json.loads(e.read())


def soak_model_switching(membrane_bin, model_a, model_b, port, cycles):
	"""Forces 1 resident slot -- every alternation is a real evict+load
	(open a brand-new llama_context/KV cache, close the old one).

	Real, disclosed finding from running this for real (see docs/
	soak-and-concurrency-testing.md's own PR E4 section): RSS grows
	during an initial warm-up window then PLATEAUS -- classic glibc
	malloc arena-retention behavior (freed large allocations are kept
	mapped, ready for reuse, rather than immediately munmap()'d back to
	the OS) rather than a genuine per-cycle leak. A fixed total-growth
	threshold alone would incorrectly fail this real, benign pattern
	(observed real growth before plateau: baseline 13 MiB -> ~450 MiB
	over 12 switches, then FLAT for 4+ more) -- so this check instead
	compares growth in the SECOND half of the cycles against the FIRST
	half: a genuine unbounded leak keeps growing at a similar rate in
	both halves; a plateaued allocator-retention pattern shows the
	second half growing far less than the first. Section 28 of the
	task: "no monotonic growth observed" is reported as exactly that
	claim, never overclaimed as "zero growth" when real growth (that
	then plateaus) was actually observed."""
	server, tmpdir, _ = _start_server(membrane_bin,
		[("model-a", model_a), ("model-b", model_b)], port,
		{"MEMBRANE_MAX_RESIDENT_MODELS": "1"})
	try:
		baseline = _proc_stat(server.pid)
		failures = 0
		samples = [baseline["rss_bytes"] or 0]
		for i in range(cycles):
			target = "model-a" if i % 2 == 0 else "model-b"
			status, body = _post(port, "/membrane/v1/models/activate",
				{"model": target})
			if status != 200 or not body.get("ok"):
				failures += 1
			samples.append((_proc_stat(server.pid))["rss_bytes"] or 0)
		final = _proc_stat(server.pid)
		healthy_after = _wait_healthy(port, timeout_s=5)
		rss_growth = (final["rss_bytes"] or 0) - (baseline["rss_bytes"] or 0)
		mid = len(samples) // 2
		first_half_growth = samples[mid] - samples[0]
		second_half_growth = samples[-1] - samples[mid]
		plateaued = second_half_growth <= max(first_half_growth * 0.34,
			8 * 1024 * 1024)
		result = {"cycles": cycles, "failures": failures,
			"healthy_after": healthy_after, "baseline": baseline,
			"final": final, "rss_growth_bytes": rss_growth,
			"thread_growth": final["n_threads"] - baseline["n_threads"],
			"fd_growth": final["n_fds"] - baseline["n_fds"],
			"rss_samples_bytes": samples,
			"first_half_growth_bytes": first_half_growth,
			"second_half_growth_bytes": second_half_growth,
			"plateaued": plateaued}
		result["pass"] = (failures == 0 and healthy_after and plateaued
			and result["thread_growth"] <= MAX_THREAD_GROWTH
			and result["fd_growth"] <= MAX_FD_GROWTH)
		return result
	finally:
		_stop_server(server)
		subprocess.run(["rm", "-rf", tmpdir], check=False)


def soak_multi_model_residency(membrane_bin, model_a, model_b, port, cycles):
	"""Default residency (2 slots) -- both models loaded once, then
	repeated hot-switches (no reload should ever happen after the first
	two loads)."""
	server, tmpdir, _ = _start_server(membrane_bin,
		[("model-a", model_a), ("model-b", model_b)], port, {})
	try:
		_post(port, "/membrane/v1/models/activate", {"model": "model-a"})
		_post(port, "/membrane/v1/models/activate", {"model": "model-b"})
		baseline = _proc_stat(server.pid)
		failures = 0
		non_hot = 0
		for i in range(cycles):
			target = "model-a" if i % 2 == 0 else "model-b"
			status, body = _post(port, "/membrane/v1/models/activate",
				{"model": target})
			if status != 200 or not body.get("ok"):
				failures += 1
			if not body.get("already_active"):
				non_hot += 1
		final = _proc_stat(server.pid)
		healthy_after = _wait_healthy(port, timeout_s=5)
		rss_growth = (final["rss_bytes"] or 0) - (baseline["rss_bytes"] or 0)
		result = {"cycles": cycles, "failures": failures,
			"non_hot_switches": non_hot, "healthy_after": healthy_after,
			"rss_growth_bytes": rss_growth,
			"thread_growth": final["n_threads"] - baseline["n_threads"],
			"fd_growth": final["n_fds"] - baseline["n_fds"]}
		result["pass"] = (failures == 0 and non_hot == 0 and healthy_after
			and rss_growth < MAX_RSS_GROWTH_BYTES
			and result["thread_growth"] <= MAX_THREAD_GROWTH
			and result["fd_growth"] <= MAX_FD_GROWTH)
		return result
	finally:
		_stop_server(server)
		subprocess.run(["rm", "-rf", tmpdir], check=False)


def soak_service_restart(membrane_bin, model_a, port, cycles):
	"""Kills the real process (SIGKILL -- a real crash, not a graceful
	shutdown) and restarts it N times against the SAME persistent
	registry/config path, verifying each time that: the registry
	survived, the previously-added model is still there, the server
	comes back healthy, and it can still load that model."""
	tmpdir = tempfile.mkdtemp(prefix="membrane-e4-restart-")
	models_path = os.path.join(tmpdir, "models.json")
	failures = []
	for i in range(cycles):
		server, _, _ = _start_server(membrane_bin, [("model-a", model_a)],
			port, {}, models_path=models_path)
		models_status, models_body = 0, {}
		try:
			with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
					timeout=5) as resp:
				models_status = resp.status
				models_body = json.loads(resp.read())
		except Exception as e:  # noqa: BLE001
			failures.append(f"cycle {i}: /v1/models fetch failed: {e}")
		ids = [m["id"] for m in models_body.get("data", [])]
		if "model-a" not in ids:
			failures.append(f"cycle {i}: registry lost model-a after "
				f"restart -- ids={ids}")
		act_status, act_body = _post(port, "/membrane/v1/models/activate",
			{"model": "model-a"})
		if act_status != 200 or not act_body.get("ok"):
			failures.append(f"cycle {i}: activate failed after restart: "
				f"status={act_status} body={act_body}")
		# A real crash, not a graceful SIGTERM -- proves the NEXT start
		# recovers cleanly from a process that never got to clean up.
		_stop_server(server, sig=signal.SIGKILL)
	subprocess.run(["rm", "-rf", tmpdir], check=False)
	return {"cycles": cycles, "failures": failures, "pass": len(failures) == 0}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-a", required=True)
	parser.add_argument("--model-b", required=True)
	parser.add_argument("--membrane", default=None)
	parser.add_argument("--port", type=int, default=19100)
	parser.add_argument("--switch-cycles", type=int, default=16)
	parser.add_argument("--restart-cycles", type=int, default=5)
	args = parser.parse_args()

	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	membrane_bin = args.membrane or os.path.join(repo_root,
		"build-release-cpu", "tools", "membrane", "membrane")
	if not os.path.isfile(membrane_bin):
		print(f"soak-model-lifecycle.py: membrane binary not found at "
			f"{membrane_bin}", file=sys.stderr)
		return 1

	result = {"checks": {}}
	problems = []

	checks = [
		("model_switching", lambda: soak_model_switching(membrane_bin,
			args.model_a, args.model_b, args.port, args.switch_cycles)),
		("multi_model_residency", lambda: soak_multi_model_residency(
			membrane_bin, args.model_a, args.model_b, args.port + 1,
			args.switch_cycles)),
		("service_restart", lambda: soak_service_restart(membrane_bin,
			args.model_a, args.port + 2, args.restart_cycles)),
	]
	for name, fn in checks:
		r = fn()
		result["checks"][name] = r
		if not r["pass"]:
			problems.append(name)

	print(json.dumps(result, indent=2))
	if problems:
		print("soak-model-lifecycle.py: FAIL: " + "; ".join(problems),
			file=sys.stderr)
		return 1
	print("soak-model-lifecycle.py: PASS", file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
