#!/usr/bin/env python3
"""Validate docs/compatibility.json: schema/required fields, evidence paths
actually exist, and a small set of status-integrity invariants (no
SUPPORTED claim with empty evidence, no UNSUPPORTED claim with no concrete
reason, no NOT_YET_VALIDATED row citing a results/ evidence file as if it
were direct proof of support).

This is a product-compatibility-claim validator, separate from
scripts/verify-results.py (which checks research/benchmark evidence
artifacts). Keeping them separate on purpose -- see docs/compatibility.md.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "docs" / "compatibility.json"
FAILURES = []
CHECK_COUNT = 0

REQUIRED_ROW_FIELDS = [
	"id", "model_family", "model_arch", "backend", "kv_precision",
	"kv_placement", "gpu_layers_mode", "context_class", "hardware_scope",
	"status", "reason_code", "evidence", "notes",
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


def evidence_file_exists(entry):
	"""evidence entries may carry a '#anchor' suffix naming a section/test
	inside the file -- only the path portion is checked on disk."""
	path_part = entry.split("#", 1)[0]
	return (REPO_ROOT / path_part).exists()


@check("compatibility.json is valid JSON with the expected top-level shape")
def check_top_level(matrix):
	for field in ("schema_version", "status_values", "hardware_scope_values", "rows"):
		if field not in matrix:
			fail("top-level shape", f"missing required field '{field}'")
	if matrix.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version {matrix.get('schema_version')!r}")
	if not isinstance(matrix.get("rows"), list) or len(matrix["rows"]) == 0:
		fail("top-level shape", "'rows' must be a non-empty list")


@check("every row has all required fields, correctly typed")
def check_required_fields(rows):
	for row in rows:
		rid = row.get("id", "<no id>")
		for field in REQUIRED_ROW_FIELDS:
			if field not in row:
				fail(rid, f"missing required field '{field}'")
		if "evidence" in row and not isinstance(row["evidence"], list):
			fail(rid, "'evidence' must be a list")


@check("row ids are unique")
def check_unique_ids(rows):
	seen = {}
	for row in rows:
		rid = row.get("id")
		if rid in seen:
			fail(rid, "duplicate row id")
		seen[rid] = True


@check("every row's status is one of the declared enum values")
def check_status_enum(rows, status_values):
	for row in rows:
		if row.get("status") not in status_values:
			fail(row.get("id"), f"status {row.get('status')!r} not in {status_values}")


@check("every row's hardware_scope is one of the declared enum values")
def check_hardware_scope_enum(rows, hardware_scope_values):
	for row in rows:
		if row.get("hardware_scope") not in hardware_scope_values:
			fail(row.get("id"),
				f"hardware_scope {row.get('hardware_scope')!r} not in {hardware_scope_values}")


@check("every evidence path exists on disk (before any '#anchor')")
def check_evidence_paths_exist(rows):
	for row in rows:
		for entry in row.get("evidence", []):
			if not evidence_file_exists(entry):
				fail(row.get("id"), f"evidence path does not exist: {entry}")


@check("SUPPORTED rows carry at least one evidence citation")
def check_supported_has_evidence(rows):
	for row in rows:
		if row.get("status") == "SUPPORTED" and not row.get("evidence"):
			fail(row.get("id"), "SUPPORTED row has empty evidence list")


@check("UNSUPPORTED rows state a concrete reason_code and notes")
def check_unsupported_has_reason(rows):
	for row in rows:
		if row.get("status") != "UNSUPPORTED":
			continue
		if not row.get("reason_code"):
			fail(row.get("id"), "UNSUPPORTED row has empty reason_code")
		if not row.get("notes"):
			fail(row.get("id"), "UNSUPPORTED row has empty notes")


@check("NOT_YET_VALIDATED rows never cite a results/ evidence file as proof of support")
def check_not_yet_validated_no_overclaim(rows):
	for row in rows:
		if row.get("status") != "NOT_YET_VALIDATED":
			continue
		for entry in row.get("evidence", []):
			if entry.startswith("results/"):
				fail(row.get("id"),
					f"NOT_YET_VALIDATED row cites a results/ evidence file "
					f"({entry}) -- that reads as direct validation proof, "
					f"which contradicts the status")


@check("compat_check.c's llama-only architecture gate matches what MC-17/MC-18 claim")
def check_compat_check_invariant():
	src = (REPO_ROOT / "tools" / "membrane-run" / "compat_check.c").read_text()
	if 'strcmp(arch_name, "llama")' not in src:
		fail("compat_check.c invariant",
			"expected exact-match gate on arch_name==\"llama\" not found -- "
			"docs/compatibility.json's MC-17/MC-18 rows describe this exact "
			"gate and need re-review if it changed")


@check("no CUDA build option exists, matching MC-23's claim")
def check_no_cuda_option():
	src = (REPO_ROOT / "CMakeLists.txt").read_text()
	if re.search(r"CUDA", src, re.IGNORECASE):
		fail("CUDA-absence invariant",
			"CMakeLists.txt now mentions CUDA -- MC-23 claims CUDA is not "
			"a build option anywhere in this repo and needs re-review")


def main():
	if not MATRIX_PATH.exists():
		print(f"FATAL: {MATRIX_PATH} does not exist", file=sys.stderr)
		return 1
	matrix = json.loads(MATRIX_PATH.read_text())
	check_top_level(matrix)
	rows = matrix.get("rows", [])
	check_required_fields(rows)
	check_unique_ids(rows)
	check_status_enum(rows, matrix.get("status_values", []))
	check_hardware_scope_enum(rows, matrix.get("hardware_scope_values", []))
	check_evidence_paths_exist(rows)
	check_supported_has_evidence(rows)
	check_unsupported_has_reason(rows)
	check_not_yet_validated_no_overclaim(rows)
	check_compat_check_invariant()
	check_no_cuda_option()

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({len(rows)} compatibility rows)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
