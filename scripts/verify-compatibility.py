#!/usr/bin/env python3
"""Validate docs/compatibility.json: schema/required fields, evidence paths
actually exist as real in-repository files, and a small set of
status-integrity invariants (no SUPPORTED claim with only a documentation
citation, no UNSUPPORTED claim with no concrete reason/evidence, no
NOT_YET_VALIDATED row citing a results/ evidence file as if it were direct
proof of support).

This is a product-compatibility-claim validator, separate from
scripts/verify-results.py (which checks research/benchmark evidence
artifacts). Kept as a separate script/file on purpose -- see
docs/compatibility.md's "How compatibility is verified" -- but
verify-results.py runs this script too, as an unnumbered final step, so
CONTRIBUTING.md's "every docs/ claim must be checkable by
scripts/verify-results.py" stays true without merging the two verifiers'
logic together.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "docs" / "compatibility.json"
FAILURES = []
CHECK_COUNT = 0

# Fixed, canonical enums -- deliberately NOT read from the matrix file
# itself, so a careless or malicious edit to compatibility.json's own
# status_values/hardware_scope_values can't silently redefine what
# "valid" means and pass its own new value.
CANONICAL_STATUS_VALUES = ["SUPPORTED", "UNSUPPORTED", "NOT_YET_VALIDATED"]
CANONICAL_HARDWARE_SCOPE_VALUES = ["tested", "backend-level", "not-hardware-specific"]

REQUIRED_ROW_STRING_FIELDS = [
	"id", "model_family", "model_arch", "backend", "kv_precision",
	"kv_placement", "gpu_layers_mode", "context_class", "hardware_scope",
	"status", "reason_code", "notes",
]

# Evidence citing only a documentation file repeats a claim rather than
# independently demonstrating it -- SUPPORTED rows must cite at least one
# entry outside this set (a results/ artifact, a test file, or a product
# source file).
DOC_ONLY_EVIDENCE_PREFIXES = ("README.md", "docs/")


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


def resolve_evidence_path(entry):
	"""Returns (ok, normalized_relative_posix_path_or_None). An entry may
	carry a '#anchor' suffix naming a section/test inside the file -- only
	the path portion is resolved/checked on disk. Rejects anything that
	isn't a real, in-repository regular file: absolute paths, '..'
	traversal escaping the repo, directories, and empty paths (e.g. a
	bare '#anchor' with nothing before it) all fail closed rather than
	silently resolving to the repo root."""
	path_part = entry.split("#", 1)[0]
	if not path_part or path_part.startswith("/"):
		return (False, None)
	candidate = (REPO_ROOT / path_part)
	try:
		resolved = candidate.resolve()
		resolved.relative_to(REPO_ROOT.resolve())
	except (ValueError, OSError):
		return (False, None)
	if not resolved.is_file():
		return (False, None)
	return (True, resolved.relative_to(REPO_ROOT.resolve()).as_posix())


@check("compatibility.json is valid JSON with the expected top-level shape")
def check_top_level(matrix):
	for field in ("schema_version", "status_values", "hardware_scope_values", "rows"):
		if field not in matrix:
			fail("top-level shape", f"missing required field '{field}'")
	if matrix.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version {matrix.get('schema_version')!r}")
	if not isinstance(matrix.get("rows"), list) or len(matrix["rows"]) == 0:
		fail("top-level shape", "'rows' must be a non-empty list")


@check("declared status_values/hardware_scope_values match the canonical fixed enums")
def check_declared_enums_match_canonical(matrix):
	declared_status = matrix.get("status_values")
	if declared_status != CANONICAL_STATUS_VALUES:
		fail("declared enums",
			f"status_values {declared_status!r} does not match the canonical "
			f"fixed set {CANONICAL_STATUS_VALUES!r} -- the matrix cannot "
			f"redefine what a valid status is")
	declared_scope = matrix.get("hardware_scope_values")
	if declared_scope != CANONICAL_HARDWARE_SCOPE_VALUES:
		fail("declared enums",
			f"hardware_scope_values {declared_scope!r} does not match the "
			f"canonical fixed set {CANONICAL_HARDWARE_SCOPE_VALUES!r}")


@check("every row is an object with all required fields, correctly typed")
def check_required_fields(rows):
	for i, row in enumerate(rows):
		label = row.get("id", f"<row {i}, no id>") if isinstance(row, dict) else f"<row {i}>"
		if not isinstance(row, dict):
			fail(label, "row is not a JSON object")
			continue
		for field in REQUIRED_ROW_STRING_FIELDS:
			if field not in row:
				fail(label, f"missing required field '{field}'")
			elif not isinstance(row[field], str):
				fail(label, f"field '{field}' must be a string, got {type(row[field]).__name__}")
		if "evidence" not in row:
			fail(label, "missing required field 'evidence'")
		elif not isinstance(row["evidence"], list):
			fail(label, "'evidence' must be a list")
		else:
			for entry in row["evidence"]:
				if not isinstance(entry, str):
					fail(label,
						f"evidence entry must be a string, got {type(entry).__name__}: {entry!r}")


def typed_rows(rows):
	"""Rows that passed check_required_fields' type checks -- downstream
	checks that index into row fields assume this shape and would raise
	on a malformed row otherwise."""
	out = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		if all(isinstance(row.get(f), str) for f in REQUIRED_ROW_STRING_FIELDS):
			if isinstance(row.get("evidence"), list) and all(
					isinstance(e, str) for e in row["evidence"]):
				out.append(row)
	return out


@check("row ids are unique")
def check_unique_ids(rows):
	seen = {}
	for row in rows:
		rid = row.get("id")
		if rid in seen:
			fail(rid, "duplicate row id")
		seen[rid] = True


@check("every row's status is one of the canonical enum values")
def check_status_enum(rows):
	for row in rows:
		if row["status"] not in CANONICAL_STATUS_VALUES:
			fail(row["id"], f"status {row['status']!r} not in {CANONICAL_STATUS_VALUES}")


@check("every row's hardware_scope is one of the canonical enum values")
def check_hardware_scope_enum(rows):
	for row in rows:
		if row["hardware_scope"] not in CANONICAL_HARDWARE_SCOPE_VALUES:
			fail(row["id"],
				f"hardware_scope {row['hardware_scope']!r} not in "
				f"{CANONICAL_HARDWARE_SCOPE_VALUES}")


@check("every evidence path resolves to a real in-repository file (before any '#anchor')")
def check_evidence_paths_exist(rows):
	for row in rows:
		for entry in row["evidence"]:
			ok, _ = resolve_evidence_path(entry)
			if not ok:
				fail(row["id"],
					f"evidence path does not resolve to a real in-repository "
					f"file: {entry}")


@check("SUPPORTED rows cite at least one non-documentation evidence entry")
def check_supported_has_evidence(rows):
	for row in rows:
		if row["status"] != "SUPPORTED":
			continue
		if not row["evidence"]:
			fail(row["id"], "SUPPORTED row has empty evidence list")
			continue
		if all(e.startswith(DOC_ONLY_EVIDENCE_PREFIXES) for e in row["evidence"]):
			fail(row["id"],
				"SUPPORTED row's evidence is documentation-only (README.md/"
				"docs/*) -- needs at least one direct product-path citation "
				"(a results/ artifact, a test file, or a source file)")


@check("UNSUPPORTED rows carry non-empty evidence, a concrete reason_code, and notes")
def check_unsupported_has_reason(rows):
	for row in rows:
		if row["status"] != "UNSUPPORTED":
			continue
		if not row["evidence"]:
			fail(row["id"], "UNSUPPORTED row has empty evidence list")
		if not row["reason_code"]:
			fail(row["id"], "UNSUPPORTED row has empty reason_code")
		if not row["notes"]:
			fail(row["id"], "UNSUPPORTED row has empty notes")


@check("NOT_YET_VALIDATED rows never cite a results/ evidence file as proof of support")
def check_not_yet_validated_no_overclaim(rows):
	for row in rows:
		if row["status"] != "NOT_YET_VALIDATED":
			continue
		for entry in row["evidence"]:
			ok, normalized = resolve_evidence_path(entry)
			if ok and normalized.startswith("results/"):
				fail(row["id"],
					f"NOT_YET_VALIDATED row cites a results/ evidence file "
					f"({entry}) -- that reads as direct validation proof, "
					f"which contradicts the status")


def strip_c_comments(src):
	"""Removes // and /* */ comments (a simple state machine, not a full
	C tokenizer -- good enough to keep check_compat_check_invariant from
	matching text that only exists inside a comment)."""
	out = []
	i = 0
	n = len(src)
	while i < n:
		if src[i:i + 2] == "//":
			i = src.find("\n", i)
			if i == -1:
				break
		elif src[i:i + 2] == "/*":
			end = src.find("*/", i + 2)
			i = n if end == -1 else end + 2
		else:
			out.append(src[i])
			i += 1
	return "".join(out)


@check("compat_check.c's architecture allowlist matches what MC-17/MC-18 claim "
	"(Phase 26: exactly llama + qwen2, an explicit list, not a widened "
	"strcmp)")
def check_compat_check_invariant():
	path = REPO_ROOT / "tools" / "membrane-run" / "compat_check.c"
	src = strip_c_comments(path.read_text())
	if "MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST" not in src:
		fail("compat_check.c invariant",
			"expected the named architecture allowlist array not found "
			"outside comments -- docs/compatibility.json's MC-17/MC-18/"
			"MC-19 rows describe this exact gate and need re-review if it "
			"changed.")
		return
	for arch in ('"llama"', '"qwen2"'):
		if arch not in src:
			fail("compat_check.c invariant",
				f"expected allowlist entry {arch} not found outside "
				f"comments -- docs/compatibility.json's MC-13..MC-19 rows "
				f"describe both architectures and need re-review if "
				f"either was removed.")
	# The gate itself must actually consult the allowlist, not just
	# define it unused -- a real behavioral proof, not just a static
	# grep, still lives in tools/membrane-run/test_compat_check.c's
	# test_q8_qwen2_real_shape_accepted/test_q8_gemma3_still_rejected/
	# etc, which run as `ctest -R test_compat_check` inside CI's
	# build-and-test job on every push/PR.
	if "membrane_arch_supports_compressed_kv(arch_name)" not in src:
		fail("compat_check.c invariant",
			"membrane_check_kv_compat() no longer appears to call the "
			"allowlist helper -- the gate may have been bypassed")


def list_cmake_files_under(root):
	"""Every CMakeLists.txt/*.cmake file under root, as root-relative
	posix path strings, found via a plain filesystem walk -- not `git
	ls-files`, so this same function is directly testable against an
	arbitrary temp directory (see test_cuda_scan_detects_nested_reference
	below), not just a real git checkout."""
	files = list(root.rglob("CMakeLists.txt")) + list(root.rglob("*.cmake"))
	return sorted(p.relative_to(root).as_posix() for p in files)


def exclude_third_party(rel_paths):
	"""Pure: drops anything rooted under third_party/ (vendored code, not
	this project's own build configuration -- MC-23's claim is scoped to
	this repo's own CMake, not llama.cpp's)."""
	return [p for p in rel_paths if not p.startswith("third_party/")]


def scan_cmake_files_for_cuda(root, rel_paths):
	"""Pure-ish (only reads files, never writes): returns the subset of
	rel_paths whose content contains a case-insensitive 'cuda' reference.
	Takes an explicit root + rel_paths rather than walking anything
	itself, so a test can hand it a temp directory's file list directly."""
	hits = []
	for rel_path in rel_paths:
		if re.search(r"cuda", (root / rel_path).read_text(), re.IGNORECASE):
			hits.append(rel_path)
	return hits


@check("no CUDA build option exists in any tracked CMake file, matching MC-23's claim")
def check_no_cuda_option():
	tracked = subprocess.run(
		["git", "-C", str(REPO_ROOT), "ls-files",
			"*CMakeLists.txt", "*.cmake"],
		capture_output=True, text=True, check=True,
	).stdout.splitlines()
	rel_paths = exclude_third_party(tracked)
	for rel_path in scan_cmake_files_for_cuda(REPO_ROOT, rel_paths):
		fail("CUDA-absence invariant",
			f"{rel_path} mentions CUDA -- MC-23 claims CUDA is not a "
			f"build option anywhere in this repo's own CMake and needs "
			f"re-review")


@check("CUDA scan itself detects a nested reference and excludes third_party (self-test)")
def check_cuda_scan_self_test():
	"""Regression test for check_no_cuda_option()'s own scanning logic --
	proves the scan actually recurses into nested subdirectories and
	actually excludes third_party/, using a throwaway temp directory tree
	(never the real repository, never committed anywhere). Without this,
	a refactor could silently make the scan non-recursive or drop the
	third_party exclusion and every check above would keep passing simply
	because MEMBRANE's own real CMake files happen to be clean."""
	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp)
		clean = root / "CMakeLists.txt"
		clean.parent.mkdir(parents=True, exist_ok=True)
		clean.write_text("project(fixture)\n")

		nested = root / "some" / "subdir" / "CMakeLists.txt"
		nested.parent.mkdir(parents=True, exist_ok=True)
		nested.write_text('option(MEMBRANE_ENABLE_CUDA "test" OFF)\n')

		vendored = root / "third_party" / "vendor" / "CMakeLists.txt"
		vendored.parent.mkdir(parents=True, exist_ok=True)
		vendored.write_text("find_package(CUDA REQUIRED)\n")

		all_files = list_cmake_files_under(root)
		if sorted(all_files) != sorted([
				"CMakeLists.txt", "some/subdir/CMakeLists.txt",
				"third_party/vendor/CMakeLists.txt"]):
			fail("CUDA scan self-test",
				f"list_cmake_files_under() found {all_files!r}, expected "
				f"all three fixture files -- the filesystem walk itself "
				f"is broken")
			return

		non_third_party = exclude_third_party(all_files)
		if "third_party/vendor/CMakeLists.txt" in non_third_party:
			fail("CUDA scan self-test",
				"exclude_third_party() did not drop the third_party "
				"fixture file")
			return

		hits = scan_cmake_files_for_cuda(root, non_third_party)
		if "some/subdir/CMakeLists.txt" not in hits:
			fail("CUDA scan self-test",
				"scan_cmake_files_for_cuda() failed to detect a CUDA "
				"reference in a NESTED (non-root) CMake file -- this is "
				"exactly the gap the real check_no_cuda_option() must not "
				"have")
		if "CMakeLists.txt" in hits:
			fail("CUDA scan self-test",
				"scan_cmake_files_for_cuda() false-positived on the clean "
				"fixture file")
		if len(hits) != 1:
			fail("CUDA scan self-test",
				f"expected exactly one hit (the nested fixture file), got "
				f"{hits!r} -- the third_party fixture file must never even "
				f"be scanned, since it was excluded before scan_cmake_"
				f"files_for_cuda() was called")


def main():
	if not MATRIX_PATH.exists():
		print(f"FATAL: {MATRIX_PATH} does not exist", file=sys.stderr)
		return 1
	matrix = json.loads(MATRIX_PATH.read_text())
	check_top_level(matrix)
	check_declared_enums_match_canonical(matrix)
	raw_rows = matrix.get("rows", [])
	check_required_fields(raw_rows)
	rows = typed_rows(raw_rows)
	if len(rows) != len(raw_rows):
		# Malformed rows already failed check_required_fields above;
		# downstream checks only run against well-typed rows so they
		# don't crash on the malformed ones.
		pass
	check_unique_ids(rows)
	check_status_enum(rows)
	check_hardware_scope_enum(rows)
	check_evidence_paths_exist(rows)
	check_supported_has_evidence(rows)
	check_unsupported_has_reason(rows)
	check_not_yet_validated_no_overclaim(rows)
	check_compat_check_invariant()
	check_no_cuda_option()
	check_cuda_scan_self_test()

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({len(raw_rows)} compatibility rows)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
