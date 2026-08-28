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
	(re.compile(r"(?<![:\w])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![:\w])"),
		"what looks like an IPv6 address"),
	(re.compile(r"\bSerial\s*(?:Number)?\s*[:=]", re.IGNORECASE), "a serial-number field"),
	(re.compile(r"\b(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET|PASSWORD|BEARER|"
		r"AUTH[_-]?TOKEN)\s*[:=]\s*\S+", re.IGNORECASE),
		"what looks like a secret/credential (KEY=/TOKEN=/SECRET=/... style)"),
	(re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "what looks like a GitHub personal access token"),
]

# Section 5: reporters are told to sanitize any model path down to just the
# filename -- a "command" field should therefore never contain an absolute
# filesystem path at all (a repo-relative path like "models/foo.gguf" is
# fine and expected). Separate from LEAK_PATTERNS above (which only catch
# specific known-sensitive path prefixes): this is a blanket rule for this
# one field.
COMMAND_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![:\w])/[\w.\-]+(?:/[\w.\-]+)+")


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


INSTALL_FIELDS = ("configure", "build", "install", "outside_tree_run", "uninstall")
RUNTIME_REQUIRED_FIELDS = ("model", "architecture", "context", "command", "outcome")
ISSUE_REQUIRED_FIELDS = ("category", "severity", "notes")
PRIVACY_FIELDS = ("no_username", "no_home_path", "no_hostname",
	"no_serial_numbers", "no_ip_addresses", "no_secrets")
ROOT_REQUIRED_FIELDS = ("schema_version", "environment_id", "provenance", "tag",
	"commit", "os", "cpu", "ram_gb", "backend", "compiler",
	"cmake_version", "install", "runtime", "issues", "privacy")


@check("summary.json entries satisfy every nested requirement in schema.json "
	"(not just root-level field presence)")
def check_shape(entries):
	if not isinstance(entries, list):
		fail("shape", "summary.json must be a JSON array")
		return
	for e in entries:
		eid = e.get("environment_id", "<no id>") if isinstance(e, dict) else "<not an object>"
		if not isinstance(e, dict):
			fail(eid, "entry is not a JSON object")
			continue
		for field in ROOT_REQUIRED_FIELDS:
			if field not in e:
				fail(eid, f"missing required field {field!r}")
		if e.get("schema_version") != 1:
			fail(eid, f"schema_version is {e.get('schema_version')!r}, expected 1")
		if not isinstance(e.get("ram_gb"), (int, float)):
			fail(eid, f"ram_gb is {e.get('ram_gb')!r}, expected a number")

		install = e.get("install")
		if not isinstance(install, dict):
			fail(eid, f"install must be an object, got {type(install).__name__}")
		else:
			for field in INSTALL_FIELDS:
				if field not in install:
					fail(eid, f"install missing required field {field!r}")
				elif not isinstance(install[field], bool):
					fail(eid, f"install.{field} must be a boolean, got "
						f"{type(install[field]).__name__}")
			extra = set(install) - set(INSTALL_FIELDS)
			if extra:
				fail(eid, f"install has unexpected field(s): {sorted(extra)!r}")

		runtime = e.get("runtime")
		if not isinstance(runtime, list):
			fail(eid, f"runtime must be an array, got {type(runtime).__name__}")
		else:
			for i, r in enumerate(runtime):
				rid = f"{eid}.runtime[{i}]"
				if not isinstance(r, dict):
					fail(rid, "runtime entry is not a JSON object")
					continue
				for field in RUNTIME_REQUIRED_FIELDS:
					if field not in r:
						fail(rid, f"missing required field {field!r}")
				if "context" in r and not isinstance(r["context"], int):
					fail(rid, f"context must be an integer, got "
						f"{type(r['context']).__name__}")
				if "fallback_attempted" in r and not isinstance(
						r["fallback_attempted"], bool):
					fail(rid, "fallback_attempted must be a boolean when present")

		issues = e.get("issues")
		if not isinstance(issues, list):
			fail(eid, f"issues must be an array, got {type(issues).__name__}")
		else:
			for i, iss in enumerate(issues):
				iid = f"{eid}.issues[{i}]"
				if not isinstance(iss, dict):
					fail(iid, "issue entry is not a JSON object")
					continue
				for field in ISSUE_REQUIRED_FIELDS:
					if field not in iss:
						fail(iid, f"missing required field {field!r}")

		privacy = e.get("privacy")
		if not isinstance(privacy, dict):
			fail(eid, f"privacy must be an object, got {type(privacy).__name__}")
		else:
			for field in PRIVACY_FIELDS:
				if field not in privacy:
					fail(eid, f"privacy missing required field {field!r}")
			extra = set(privacy) - set(PRIVACY_FIELDS)
			if extra:
				fail(eid, f"privacy has unexpected field(s): {sorted(extra)!r}")


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
		for field in PRIVACY_FIELDS:
			if privacy.get(field) is not True:
				fail(eid, f"privacy.{field} is not true")


@check("no 'command' field contains an absolute filesystem path")
def check_command_no_absolute_path(entries):
	for e in entries:
		eid = e.get("environment_id", "<no id>")
		for r in e.get("runtime", []) if isinstance(e.get("runtime"), list) else []:
			if not isinstance(r, dict):
				continue
			cmd = r.get("command")
			if isinstance(cmd, str) and COMMAND_ABSOLUTE_PATH_PATTERN.search(cmd):
				fail(eid, f"command field contains an absolute path (model "
					f"paths must be sanitized to just the filename): {cmd!r}")


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


DOC_PATH = REPO_ROOT / "docs" / "rc2-user-validation.md"
DOC_TABLE_ROW_PATTERN = re.compile(
	r"^\|\s*(\S+)\s*\|.*\|\s*(VALIDATED|NOT_YET_VALIDATED)\b")


@check("docs/rc2-user-validation.md's status table matches summary.json (no drift)")
def check_doc_matches_summary(entries):
	if not DOC_PATH.exists():
		fail("doc table", f"{DOC_PATH} does not exist")
		return
	summary_ids = {e.get("environment_id") for e in entries if isinstance(e, dict)}
	doc_validated_ids = set()
	for line in DOC_PATH.read_text().splitlines():
		m = DOC_TABLE_ROW_PATTERN.match(line.strip())
		if not m:
			continue
		row_id, status = m.group(1), m.group(2)
		if status == "VALIDATED":
			doc_validated_ids.add(row_id)
			if row_id not in summary_ids:
				fail("doc table", f"row {row_id!r} is marked VALIDATED in "
					f"{DOC_PATH.name} but has no matching entry in "
					f"summary.json")
	missing_from_doc = summary_ids - doc_validated_ids
	if missing_from_doc:
		fail("doc table", f"summary.json has entries not reflected as "
			f"VALIDATED in {DOC_PATH.name}: {sorted(missing_from_doc)!r}")


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
		check_command_no_absolute_path(entries)
		check_privacy_attestation(entries)
		check_no_fake_success(entries)
		check_doc_matches_summary(entries)

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
