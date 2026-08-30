#!/usr/bin/env python3
"""Validate results/compat-expansion/validation.json: schema, that every
row's architecture/backend/precision values are real enums, that KV byte
counts actually match the documented Q8_0/Q5_1 block-format reduction
formulas (not just internally self-consistent numbers), that every row
carries real hardware/commit provenance, that no row promotes an
unallowlisted architecture to supported evidence, and that
docs/compatibility.json's MC-17/MC-18/MC-19 rows (the rows this phase's
evidence backs) actually cite this file.

This is a measurement-evidence validator, same one-file-per-concern
convention as every other scripts/verify-*.py in this project. It
validates the COMMITTED evidence file; Phase 26 ran no separate
regeneration script for it (a small, one-off architecture-expansion
investigation, not a repeatable sweep -- see docs/compat-expansion.md
for full methodology).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "compat-expansion" / "validation.json"
COMPAT_JSON_PATH = REPO_ROOT / "docs" / "compatibility.json"
DOC_PATH = REPO_ROOT / "docs" / "compat-expansion.md"
COMPAT_CHECK_C_PATH = REPO_ROOT / "tools" / "membrane-run" / "compat_check.c"
FAILURES = []
CHECK_COUNT = 0

VALID_EVIDENCE_CLASSES = frozenset({"SOURCE_PROVEN", "REAL", "SIMULATED"})
VALID_BACKENDS = frozenset({"cpu", "vulkan"})
VALID_PRECISIONS = frozenset({"native", "q8", "q5", "adaptive"})
VALID_PLACEMENTS = frozenset({"default", "gpu", "cpu", "auto"})
REQUIRED_ROW_FIELDS = (
	"id", "evidence_class", "architecture", "model", "backend", "precision",
	"placement", "context", "command", "exit_code", "quality_result",
	"kv_bytes", "resolved_plan", "hardware", "commit", "notes",
)

# Real ggml block-format byte cost per element, vs F16's 2 bytes/element
# -- Q8_0: 34 bytes per 32-element block (1 fp16 scale + 32 int8 values).
# Q5_1: 24 bytes per 32-element block (2 fp16 scale/min + 4 bytes high
# bits + 16 bytes low nibbles for 32 elements). Both confirmed against
# ggml's own quantize.c block struct sizes, same formula
# docs/live-runtime.md already documents for the llama-only case.
EXPECTED_Q8_RATIO = 34 / 32 / 2
EXPECTED_Q5_RATIO = 24 / 32 / 2
RATIO_TOLERANCE = 0.002


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


def get_allowlisted_architectures():
	"""Parses compat_check.c's own MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST
	array literally, so this validator's notion of "which architectures
	are actually allowlisted" can never silently drift from the real
	source (Phase 26's own "no false compatibility claim" discipline)."""
	if not COMPAT_CHECK_C_PATH.exists():
		return set()
	src = COMPAT_CHECK_C_PATH.read_text()
	m = re.search(
		r"MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST\[\]\s*=\s*\{([^}]*)\}",
		src, re.DOTALL)
	if not m:
		return set()
	return set(re.findall(r'"([a-zA-Z0-9_.]+)"', m.group(1)))


@check("validation.json is valid JSON with the expected top-level shape")
def check_top_level(data):
	for field in ("schema_version", "membrane_commit", "generated_at",
			"expansion_decision", "rows"):
		if field not in data:
			fail("top-level shape", f"missing required field {field!r}")
	if data.get("schema_version") != 1:
		fail("top-level shape", f"unexpected schema_version "
			f"{data.get('schema_version')!r}")
	commit = data.get("membrane_commit", "")
	if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
		fail("top-level shape", f"membrane_commit {commit!r} is not a "
			f"40-hex SHA")
	valid_decisions = frozenset({
		"EXPAND_QWEN2_COMPRESSION", "EXPAND_OTHER_ARCH", "EXPAND_BACKEND",
		"NO_SAFE_EXPANSION_YET",
	})
	if data.get("expansion_decision") not in valid_decisions:
		fail("top-level shape", f"expansion_decision "
			f"{data.get('expansion_decision')!r} is not one of "
			f"{sorted(valid_decisions)}")
	if not isinstance(data.get("rows"), list) or not data["rows"]:
		fail("top-level shape", "'rows' must be a non-empty list")


# CodeRabbit review (PR #36): a non-empty 'rows' list alone does not
# substantiate docs/compatibility.md's claim of 8 successful real Qwen2
# validations -- a row could be deleted, or silently downgraded to a
# failed/non-REAL/empty-quality entry, and this file would still pass.
# Pin the exact coverage this phase's own real experiment produced (one
# row id per (backend, precision, placement) combination Sections
# 7/10/11/12 of the Phase 26 task required) so removing or degrading any
# one of them is a real, loud failure, not silently unnoticed.
REQUIRED_ROW_COVERAGE = {
	"CE-01": ("cpu", "native", "default"),
	"CE-02": ("cpu", "q8", "default"),
	"CE-03": ("cpu", "q5", "default"),
	"CE-04": ("cpu", "adaptive", "default"),
	"CE-05": ("vulkan", "q8", "default"),
	"CE-06": ("vulkan", "q5", "default"),
	"CE-07": ("vulkan", "q8", "cpu"),
	"CE-08": ("vulkan", "adaptive", "default"),
}


@check("the exact required (backend, precision, placement) coverage is "
	"present, every required row is REAL with a successful exit and a "
	"real quality result -- not just a non-empty rows list "
	"(CodeRabbit review, PR #36)")
def check_required_coverage(rows):
	by_id = {r.get("id"): r for r in rows}
	missing_ids = set(REQUIRED_ROW_COVERAGE) - set(by_id)
	if missing_ids:
		fail("required coverage", f"missing required row id(s): "
			f"{sorted(missing_ids)} -- docs/compatibility.md's "
			f"MC-17/MC-18/MC-19 SUPPORTED claim depends on this exact "
			f"coverage existing")
	for rid, (backend, precision, placement) in REQUIRED_ROW_COVERAGE.items():
		row = by_id.get(rid)
		if row is None:
			continue
		if (row.get("backend"), row.get("precision"), row.get("placement")) \
				!= (backend, precision, placement):
			fail(rid, f"expected (backend, precision, placement) = "
				f"{(backend, precision, placement)!r}, found "
				f"{(row.get('backend'), row.get('precision'), row.get('placement'))!r} "
				f"-- a required row was repurposed to cover something "
				f"else")
		if row.get("evidence_class") != "REAL":
			fail(rid, f"evidence_class is {row.get('evidence_class')!r}, "
				f"expected REAL -- this row backs a SUPPORTED "
				f"compatibility claim and cannot be SIMULATED")
		if row.get("exit_code") != 0:
			fail(rid, f"exit_code is {row.get('exit_code')!r}, expected 0 "
				f"-- a required row must be a real success, not a "
				f"failed run")
		quality = row.get("quality_result")
		if not isinstance(quality, dict) \
				or not (quality.get("generated_text") or "").strip():
			fail(rid, "quality_result.generated_text is missing/empty -- "
				"a required row must show real, non-empty generated "
				"output, not just an exit code")


@check("every row has the full required schema and real enum values")
def check_row_schema(rows, allowlisted):
	for row in rows:
		rid = row.get("id", "<no id>")
		for field in REQUIRED_ROW_FIELDS:
			if field not in row:
				fail(rid, f"missing required field {field!r}")
		if row.get("evidence_class") not in VALID_EVIDENCE_CLASSES:
			fail(rid, f"evidence_class {row.get('evidence_class')!r} not "
				f"in {sorted(VALID_EVIDENCE_CLASSES)}")
		if row.get("backend") not in VALID_BACKENDS:
			fail(rid, f"backend {row.get('backend')!r} not in "
				f"{sorted(VALID_BACKENDS)}")
		if row.get("precision") not in VALID_PRECISIONS:
			fail(rid, f"precision {row.get('precision')!r} not in "
				f"{sorted(VALID_PRECISIONS)}")
		if row.get("placement") not in VALID_PLACEMENTS:
			fail(rid, f"placement {row.get('placement')!r} not in "
				f"{sorted(VALID_PLACEMENTS)}")
		if not re.fullmatch(r"[0-9a-f]{40}", row.get("commit", "") or ""):
			fail(rid, f"commit {row.get('commit')!r} is not a 40-hex SHA")
		if not row.get("hardware"):
			fail(rid, "missing/empty 'hardware' -- every claim must be "
				"hardware-scoped, never generalized")


@check("no REAL row promotes an architecture that isn't actually on "
	"compat_check.c's own allowlist to supported evidence")
def check_no_unallowlisted_promotion(rows, allowlisted):
	if not allowlisted:
		fail("allowlist", "could not parse "
			"MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST from compat_check.c -- "
			"cannot cross-check row architectures against real source")
		return
	for row in rows:
		if row.get("evidence_class") != "REAL":
			continue
		if row.get("precision") == "native":
			continue	# native has no architecture gate at all
		arch = row.get("architecture")
		if arch not in allowlisted:
			fail(row.get("id", "<no id>"), f"claims REAL evidence for "
				f"compressed KV ({row.get('precision')!r}) on architecture "
				f"{arch!r}, which is NOT in compat_check.c's own real "
				f"allowlist {sorted(allowlisted)} -- this would be a false "
				f"compatibility claim the actual gate does not honor")


@check("q8/q5 KV byte reductions match the real ggml Q8_0/Q5_1 block-"
	"format formulas, recomputed independently -- not just self-"
	"consistent numbers")
def check_kv_byte_arithmetic(rows):
	by_key = {}
	for row in rows:
		key = (row.get("architecture"), row.get("model"), row.get("backend"),
			row.get("context"), row.get("placement"))
		by_key.setdefault(key, {})[row.get("precision")] = row

	for key, by_precision in by_key.items():
		native = by_precision.get("native")
		if native is None:
			continue
		native_kv = native.get("kv_bytes")
		if not isinstance(native_kv, (int, float)) or native_kv <= 0:
			fail(native.get("id", "<no id>"), f"native kv_bytes "
				f"{native_kv!r} is not a positive number -- cannot check "
				f"any compressed row's byte ratio against it")
			continue
		for precision, expected_ratio in (
				("q8", EXPECTED_Q8_RATIO), ("q5", EXPECTED_Q5_RATIO)):
			row = by_precision.get(precision)
			if row is None:
				# A native baseline exists for this exact (architecture,
				# model, backend, context, placement) key but its q8/q5
				# counterpart is missing -- CodeRabbit review (PR #36):
				# this used to `continue` silently, letting a deleted
				# compressed row go unnoticed by this check even though
				# the byte-ratio claim depends on it existing.
				fail(f"{key!r}", f"a native row exists for this "
					f"(architecture, model, backend, context, placement) "
					f"but no matching {precision!r} row was found -- "
					f"cannot verify the {precision} byte-ratio claim")
				continue
			kv = row.get("kv_bytes")
			rid = row.get("id", "<no id>")
			if not isinstance(kv, (int, float)) or kv <= 0:
				fail(rid, f"kv_bytes {kv!r} is not a positive number")
				continue
			real_ratio = kv / native_kv
			if abs(real_ratio - expected_ratio) > RATIO_TOLERANCE:
				fail(rid, f"{precision} kv_bytes {kv} is "
					f"{round(real_ratio, 6)} of native's {native_kv}, "
					f"expected ~{round(expected_ratio, 6)} (the real "
					f"ggml block-format ratio) -- outside tolerance "
					f"{RATIO_TOLERANCE}, suggesting compression was not "
					f"actually applied or the byte count was fabricated")


@check("docs/compatibility.json's MC-17/MC-18/MC-19 rows are SUPPORTED "
	"and cite this evidence file")
def check_compatibility_json_cites_this_file():
	if not COMPAT_JSON_PATH.exists():
		fail("compatibility.json", f"{COMPAT_JSON_PATH} does not exist")
		return
	compat = json.loads(COMPAT_JSON_PATH.read_text())
	by_id = {r.get("id"): r for r in compat.get("rows", [])}
	for rid in ("MC-17", "MC-18", "MC-19"):
		row = by_id.get(rid)
		if row is None:
			fail("compatibility.json", f"{rid} not found")
			continue
		if row.get("status") != "SUPPORTED":
			fail("compatibility.json", f"{rid}.status is "
				f"{row.get('status')!r}, expected SUPPORTED after Phase "
				f"26's real Qwen2 evidence")
		if not any("compat-expansion/validation.json" in e
				for e in row.get("evidence", [])):
			fail("compatibility.json", f"{rid} does not cite "
				f"results/compat-expansion/validation.json in its "
				f"'evidence' list")


@check("docs/compat-expansion.md exists and names the expansion decision")
def check_doc_exists(data):
	if not DOC_PATH.exists():
		fail("doc claims", f"{DOC_PATH} does not exist")
		return
	text = DOC_PATH.read_text()
	decision = data.get("expansion_decision", "")
	if decision and decision not in text:
		fail("doc claims", f"expansion_decision {decision!r} is never "
			f"mentioned in {DOC_PATH.name}")


def main():
	if not DATA_PATH.exists():
		print(f"FATAL: {DATA_PATH} does not exist", file=sys.stderr)
		return 1
	data = json.loads(DATA_PATH.read_text())
	check_top_level(data)
	rows = data.get("rows", [])
	if not isinstance(rows, list):
		rows = []
	allowlisted = get_allowlisted_architectures()
	check_required_coverage(rows)
	check_row_schema(rows, allowlisted)
	check_no_unallowlisted_promotion(rows, allowlisted)
	check_kv_byte_arithmetic(rows)
	check_compatibility_json_cites_this_file()
	check_doc_exists(data)

	print()
	if FAILURES:
		print(f"{len(FAILURES)} failure(s) out of {CHECK_COUNT} checks:")
		for f in FAILURES:
			print(f"  - {f}")
		return 1
	print(f"{CHECK_COUNT}/{CHECK_COUNT} checks passed ({len(rows)} rows)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
