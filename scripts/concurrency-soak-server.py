#!/usr/bin/env python3
"""Real, external-process concurrency soak for `membrane serve` around
the bounded-admission limit (Mega Phase C, PR C3).

`test_server.cpp`'s own `test_concurrent_requests_are_thread_safe`
already fires 16 real, simultaneous connections at an in-process test
server, TSan-instrumented, in CI (`server-thread-sanitizer` job). This
script adds a complementary, real EXTERNAL-PROCESS check: a real
`membrane serve` binary, launched as its own OS process, hit with real
concurrent HTTP connections specifically clustered AROUND the bounded-
admission limit (8 by default) rather than all far below or all far
above it -- checking that admission succeeds up to the limit and fails
closed (503 SERVER_BUSY with Retry-After) beyond it, under real
concurrent load, with no crash and no hang.

Run the server binary itself under a TSan build (see docs/soak-and-
concurrency-testing.md) to also catch a real data race, if one exists.

Usage:
  scripts/concurrency-soak-server.py --model PATH [--concurrency N]
                                      [--membrane BIN] [--port PORT]

Exit code: 0 if the server handled the concurrent burst without
crashing and produced a sane admit/reject split; 1 otherwise.
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
from concurrent.futures import ThreadPoolExecutor

ADMISSION_LIMIT = 8  # request_admission.h's own real, current bound


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


def _one_request(port, model_name, request_timeout):
	body = json.dumps({
		"model": model_name,
		"messages": [{"role": "user", "content": "hi"}],
		"max_tokens": 4,
	}).encode()
	req = urllib.request.Request(
		f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
		headers={"Content-Type": "application/json"}, method="POST")
	try:
		with urllib.request.urlopen(req, timeout=request_timeout) as resp:
			code = None
			return resp.status, dict(resp.getheaders()), code
	except urllib.error.HTTPError as e:
		code = None
		try:
			payload = json.loads(e.read())
			code = payload.get("error", {}).get("code")
		except (ValueError, AttributeError):
			pass
		return e.code, dict(e.headers or {}), code
	except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
		return None, {"exception": str(e)}, None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--concurrency", type=int, default=2 * ADMISSION_LIMIT)
	parser.add_argument("--membrane", default=None)
	parser.add_argument("--port", type=int, default=18960)
	parser.add_argument("--request-timeout", type=float, default=10.0)
	parser.add_argument("--verbose", action="store_true",
		help="include each request's raw status/headers in the JSON output")
	args = parser.parse_args()

	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	membrane_bin = args.membrane or os.path.join(
		repo_root, "build-b1", "tools", "membrane", "membrane")
	if not os.path.isfile(membrane_bin):
		print(f"concurrency-soak-server.py: membrane binary not found at "
			f"{membrane_bin}", file=sys.stderr)
		return 1

	tmpdir = tempfile.mkdtemp(prefix="membrane-concurrency-soak-")
	env = dict(os.environ)
	env["MEMBRANE_MODELS_PATH"] = os.path.join(tmpdir, "models.json")
	model_name = "concurrency-soak-model"

	add_rc = subprocess.run(
		[membrane_bin, "model", "add", model_name, args.model],
		env=env, capture_output=True, text=True)
	if add_rc.returncode != 0:
		print(f"concurrency-soak-server.py: model add failed: {add_rc.stderr}",
			file=sys.stderr)
		return 1

	server = subprocess.Popen(
		[membrane_bin, "serve", "--port", str(args.port)],
		env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		text=True)
	try:
		if not _wait_healthy(args.port):
			print("concurrency-soak-server.py: server never became healthy",
				file=sys.stderr)
			return 1

		with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
			futures = [pool.submit(_one_request, args.port, model_name,
				args.request_timeout) for _ in range(args.concurrency)]
			outcomes = [f.result() for f in futures]

		healthy_after = _wait_healthy(args.port, timeout_s=5)
		status_counts = {}
		code_counts = {}
		# Retry-After is documented (docs/server.md) as part of the
		# SERVER_BUSY (admission-gate) contract specifically -- a 503
		# NO_FEASIBLE_CONTEXT is a real, distinct, DIFFERENT 503 (no
		# Retry-After promised or needed, since retrying the exact same
		# request would not help). Checking every 503 alike was a real
		# bug in an earlier version of this script -- it flagged
		# NO_FEASIBLE_CONTEXT responses as broken SERVER_BUSY responses,
		# when they are simply a different, correctly-shaped error.
		server_busy_missing_retry_after = 0
		exceptions = 0
		for status, headers, code in outcomes:
			status_counts[status] = status_counts.get(status, 0) + 1
			if code is not None:
				code_counts[code] = code_counts.get(code, 0) + 1
			if status is None:
				exceptions += 1
			if code == "SERVER_BUSY" and "Retry-After" not in headers:
				server_busy_missing_retry_after += 1

		result = {
			"concurrency": args.concurrency,
			"admission_limit": ADMISSION_LIMIT,
			"status_counts": {str(k): v for k, v in status_counts.items()},
			"error_code_counts": code_counts,
			"server_busy_missing_retry_after": server_busy_missing_retry_after,
			"healthy_after_burst": healthy_after,
		}
		if args.verbose:
			result["raw_outcomes"] = [
				{"status": s, "headers": h, "code": c} for s, h, c in outcomes]
		print(json.dumps(result, indent=2))

		problems = []
		if exceptions > 0:
			problems.append(f"{exceptions} request(s) raised an exception "
				"instead of getting a real HTTP response (see "
				"docs/soak-and-concurrency-testing.md's disclosed "
				"host-memory-pressure timing finding)")
		if not healthy_after:
			problems.append("server not healthy immediately after the "
				"concurrent burst")
		if server_busy_missing_retry_after > 0:
			problems.append(f"{server_busy_missing_retry_after} SERVER_BUSY "
				"response(s) missing Retry-After")
		if problems:
			print("concurrency-soak-server.py: FAIL: " + "; ".join(problems),
				file=sys.stderr)
			return 1
		print("concurrency-soak-server.py: PASS: server handled the "
			"concurrent burst cleanly", file=sys.stderr)
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
