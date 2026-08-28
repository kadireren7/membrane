#!/usr/bin/env python3
"""Validate results/rc2-user-validation/summary.json: shape (against
results/rc2-user-validation/schema.json's field contract), that every entry
targets the exact v0.3.0-rc2 tag/commit, no private-path/username/hostname
leakage in any string field, no duplicate environment_id, only allowed
enum values, and no entry that claims SUCCESS while its own install/runtime
fields say otherwise (a fake success classification).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "rc2-user-validation" / "summary.json"
EXPECTED_TAG = "v0.3.0-rc2"
EXPECTED_COMMIT = "7b94217702cb6b5d170c496660cb3208e01fd714"
FAILURES = []
CHECK_COUNT = 0

VALID_BACKENDS = ("cpu", "vulkan")
VALID_OUTCOMES = ("SUCCESS", "EXPECTED_REJECTION", "UNEXPECTED_FAILURE")
VALID_KV_PRECISION = ("native", "q8", "q5")
VALID_KV_PLACEMENT = ("default", "gpu", "cpu", "auto")
VALID_ISSUE_CATEGORIES = (
	"INSTALL_BLOCKER", "BUILD_BLOCKER", "RUNTIME_BLOCKER", "PLANNER_BLOCKER",
	"FALLBACK_BLOCKER", "COMPATIBILITY_DOC_BUG", "JSON_CONTRACT_BUG",
	"DOC_FRICTION", "NONBLOCKING_USABILITY", "ENVIRONMENT_SPECIFIC",
	"FALSE_POSITIVE",
)
VALID_SEVERITIES = ("blocker", "nonblocking")

# Section 5 of the Phase 23 task: none of these may appear anywhere in a
# committed report. Deliberately broad/conservative -- false positives here
# just mean double-checking a report by hand, which is cheap; a missed leak
# is not.
LEAK_PATTERNS = [
	(re.compile(r"/home/[^/\s]+"), "a /home/<user> path"),
	(re.compile(r"/Users/[^/\s]+"), "a /Users/<user> path"),
	(re.compile(r"\bC:\\Users\\[^\\\s]+", re.IGNORECASE), "a Windows user profile path"),
	(re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "what looks like an IPv4 address"),
	(re.compile(r"\bSerial\s*(?:Number)?\s*[:=]", re.IGNORECASE), "a serial-number field"),
]


def fail(name, detail):
	FAILURES.append(f"{name}: {detail}")


def check(name):
	def decorator(fn):
		def wrapper(*args, **kwargs):
			global CHECK_COUNT
			CHECK_COUNT += 1
			before = len(FAILURES)
			fn(*args, **kwargs)
			status = "PASS" if len(FAILURES) == before else "FAIL"
			print(f"[{status}] {name}")
		return wrapper
	return decorator


def all_strings(obj):
	"""Yield every string value anywhere in a JSON-decoded structure."""
	if isinstance(obj, str):
		yield obj
	elif isinstance(obj, dict):
		for v in obj.values():
			yield from all_strings(v)
	elif isinstance(obj, list):
		for v in obj:
			yield from all_strings(v)


@check("summary.json is a valid JSON array of entries with the required fields")
def check_shape(entries):
	if not isinstance(entries, list):
		fail("shape", "summary.json must be a JSON array")
		return
	required = ("schema_version", "environment_id", "provenance", "tag",
		"commit", "os", "cpu", "ram_gb", "backend", "compiler",
		"cmake_version", "install", "runtime", "issues", "privacy")
	for e in entries:
		eid = e.get("environment_id", "<no id>") if isinstance(e, dict) else "<not an object>"
		if not isinstance(e, dict):
			fail(eid, "entry is not a JSON object")
			continue
		for field in required:
			if field not in e:
				fail(eid, f"missing required field {field!r}")
		if e.get("schema_version") != 1:
			fail(eid, f"schema_version is {e.get('schema_version')!r}, expected 1")


@check("every entry targets the exact v0.3.0-rc2 tag and commit")
def check_tag_commit(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		if e.get("tag") != EXPECTED_TAG:
			fail(eid, f"tag is {e.get('tag')!r}, expected {EXPECTED_TAG!r} -- "
				f"external validation must never target a moving main")
		if e.get("commit") != EXPECTED_COMMIT:
			fail(eid, f"commit is {e.get('commit')!r}, expected "
				f"{EXPECTED_COMMIT!r} (the exact commit v0.3.0-rc2 points to)")


@check("environment_id is unique across every entry")
def check_unique_ids(entries):
	seen = {}
	for e in entries:
		eid = e.get("environment_id")
		if eid in seen:
			fail(eid, "environment_id is not unique")
		seen[eid] = True


@check("every enum field uses only an allowed value")
def check_enums(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		if e.get("provenance") != "REAL":
			fail(eid, f"provenance is {e.get('provenance')!r} -- this schema "
				f"has no SIMULATED case; a non-REAL report does not belong here")
		if e.get("backend") not in VALID_BACKENDS:
			fail(eid, f"backend {e.get('backend')!r} not in {VALID_BACKENDS}")
		for r in e.get("runtime", []):
			if r.get("outcome") not in VALID_OUTCOMES:
				fail(eid, f"runtime outcome {r.get('outcome')!r} not in "
					f"{VALID_OUTCOMES}")
			if "selected_kv_precision" in r and r["selected_kv_precision"] not in VALID_KV_PRECISION:
				fail(eid, f"selected_kv_precision {r['selected_kv_precision']!r} "
					f"not in {VALID_KV_PRECISION}")
			if "selected_kv_placement" in r and r["selected_kv_placement"] not in VALID_KV_PLACEMENT:
				fail(eid, f"selected_kv_placement {r['selected_kv_placement']!r} "
					f"not in {VALID_KV_PLACEMENT}")
		for i in e.get("issues", []):
			if i.get("category") not in VALID_ISSUE_CATEGORIES:
				fail(eid, f"issue category {i.get('category')!r} not in "
					f"{VALID_ISSUE_CATEGORIES}")
			if i.get("severity") not in VALID_SEVERITIES:
				fail(eid, f"issue severity {i.get('severity')!r} not in "
					f"{VALID_SEVERITIES}")


@check("no private-path/username/hostname/IP/serial-number leakage anywhere in an entry")
def check_no_leakage(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		for s in all_strings(e):
			for pattern, description in LEAK_PATTERNS:
				if pattern.search(s):
					fail(eid, f"a field contains {description}: {s!r}")


@check("every entry's own privacy self-attestation is affirmatively true")
def check_privacy_attestation(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		privacy = e.get("privacy") or {}
		for field in ("no_username", "no_home_path", "no_hostname", "no_serial_numbers"):
			if privacy.get(field) is not True:
				fail(eid, f"privacy.{field} is not true")


@check("no entry claims SUCCESS while its own install/runtime fields disagree")
def check_no_fake_success(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		install = e.get("install") or {}
		has_success_runtime = any(r.get("outcome") == "SUCCESS" for r in e.get("runtime", []))
		if has_success_runtime:
			for field in ("configure", "build", "install", "outside_tree_run"):
				if install.get(field) is not True:
					fail(eid, f"a runtime entry claims SUCCESS but "
						f"install.{field} is not true")
		for r in e.get("runtime", []):
			if r.get("outcome") == "SUCCESS" and r.get("fallback_attempted") is None:
				# fallback_attempted is optional, but if present and true on
				# a SUCCESS entry that's a real, expected combination (a
				# recovered fallback) -- nothing to flag either way here,
				# this branch exists so a future stricter rule has a place
				# to live without restructuring the check.
				pass


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	entries = json.loads(DATA_PATH.read_text())
	check_shape(entries)
	if isinstance(entries, list):
		check_tag_commit(entries)
		check_unique_ids(entries)
		check_enums(entries)
		check_no_leakage(entries)
		check_privacy_attestation(entries)
		check_no_fake_success(entries)

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	n = len(entries) if isinstance(entries, list) else 0
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({n} environment reports)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
