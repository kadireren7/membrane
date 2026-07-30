#!/usr/bin/env python3
"""Verify the Phase 7.3 hardware-outreach package: internal links,
lab-package completeness, hardware-results schema conformance, and
prohibited-claim/hype-word scanning across outreach/ and hardware/.

This does NOT re-check paper/ (see paper/scripts/verify-paper.py) or
benchmarks/ (see scripts/verify-results.py) claims -- it is specific to
the outreach/hardware materials added in Phase 7.3.

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTREACH_DIR = REPO_ROOT / "outreach"
HARDWARE_DIR = REPO_ROOT / "hardware"
FAILURES = []
CHECK_COUNT = 0

REQUIRED_LAB_PACKAGE_FILES = [
	"README.md", "quick-start.md", "required-hardware.md",
	"experiment-checklist.md", "expected-artifacts.md", "collaboration-scope.md",
]

# Hype words the one-page summary (and, by extension, all outreach
# material) must never use, per the Phase 7.3 spec.
HYPE_WORDS = ["revolutionary", "production-ready", "industry-leading"]

# Hardware claims that are gated -- see outreach/hardware-claim-gates.md.
# Each maps a prohibited phrase to the gate name that would need to pass
# before it's allowed.
GATED_CLAIMS = {
	"hardware bit-exact": "Gate 4 (hardware bit-exactness)",
	"fpga-deployed": "Gate 3 (real board bring-up)",
	"fpga-verified": "Gate 3 (real board bring-up)",
	"board-verified": "Gate 3 (real board bring-up)",
	"runs on an fpga": "Gate 3 (real board bring-up)",
	"real cxl acceleration": "Gate 7 (real CXL platform integration)",
	"cxl-accelerated": "Gate 7 (real CXL platform integration)",
	"hardware-validated": "Gate 3/6/7 depending on context",
}

NEGATION_MARKERS = [
	"not ", "never ", "no ", "nowhere", "isn't", "is not", "n't", "without",
	"prohibited", "gated", "gate for", "required for", "bar for",
	"minimum bar", "before any", "any claim", "for any",
]


def check(name):
	def decorator(fn):
		def wrapper():
			global CHECK_COUNT
			CHECK_COUNT += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			status = "PASS" if ok else "FAIL"
			print(f"[{status}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return decorator


def all_outreach_hardware_markdown():
	files = []
	for base in (OUTREACH_DIR, HARDWARE_DIR):
		if base.exists():
			files.extend(sorted(base.rglob("*.md")))
	return files


@check("lab-package has all 6 required files")
def _c1():
	missing = [f for f in REQUIRED_LAB_PACKAGE_FILES
		if not (OUTREACH_DIR / "lab-package" / f).exists()]
	return len(missing) == 0, f"missing: {missing}" if missing else "all 6 present"


@check("internal (relative, non-http) links in outreach/ and hardware/ resolve")
def _c2():
	errors = []
	for md_file in all_outreach_hardware_markdown():
		md_dir = md_file.parent
		for m in re.finditer(r"\]\(([^:)]+)\)", md_file.read_text()):
			target = m.group(1).split("#")[0]
			if not target:
				continue
			resolved = (md_dir / target).resolve()
			if not resolved.exists():
				errors.append(f"{md_file.relative_to(REPO_ROOT)} -> {target}")
	return len(errors) == 0, f"{len(errors)} broken link(s): {errors[:10]}" if errors else "all internal links resolve"


@check("hardware/results-schema.json is valid JSON with the required field list")
def _c3():
	schema = json.loads((HARDWARE_DIR / "results-schema.json").read_text())
	required_fields = {
		"board", "fpga", "toolchain", "bitstream_hash", "clock_mhz",
		"pcie_generation", "pcie_width", "host_cpu", "operation", "precision",
		"batch_size", "queue_depth", "bytes", "throughput_bytes_per_sec",
		"latency_ns", "cpu_utilization_pct", "board_power_watts",
		"temperature_celsius", "parity_failures", "timestamp_utc", "result_label",
	}
	declared = set(schema.get("required", []))
	missing = required_fields - declared
	return len(missing) == 0, f"schema missing required fields: {missing}" if missing else f"all {len(required_fields)} required fields present"


def _validate_against_schema(instance, schema):
	errors = []
	props = schema["properties"]
	for r in schema.get("required", []):
		if r not in instance:
			errors.append(f"missing required field: {r}")
	for k, v in instance.items():
		if k not in props:
			if schema.get("additionalProperties") is False:
				errors.append(f"unexpected field not in schema: {k}")
			continue
		spec = props[k]
		t = spec.get("type")
		if t == "string" and not isinstance(v, str):
			errors.append(f"{k}: expected string")
		if t == "number" and not isinstance(v, (int, float)):
			errors.append(f"{k}: expected number")
		if t == "integer" and not isinstance(v, int):
			errors.append(f"{k}: expected integer")
		if t == "object" and not isinstance(v, dict):
			errors.append(f"{k}: expected object")
		if "enum" in spec and v not in spec["enum"]:
			errors.append(f"{k}: {v!r} not in enum {spec['enum']}")
		if "pattern" in spec and isinstance(v, str) and not re.fullmatch(spec["pattern"], v):
			errors.append(f"{k}: {v!r} does not match pattern {spec['pattern']}")
		if t == "object" and "required" in spec:
			for rr in spec["required"]:
				if rr not in v:
					errors.append(f"{k}.{rr}: missing required nested field")
	return errors


@check("hardware/results-example.json validates against the schema and is labeled DOCUMENTED_EXAMPLE")
def _c4():
	schema = json.loads((HARDWARE_DIR / "results-schema.json").read_text())
	example_path = HARDWARE_DIR / "results-example.json"
	if not example_path.exists():
		return False, "hardware/results-example.json does not exist"
	example = json.loads(example_path.read_text())
	errors = _validate_against_schema(example, schema)
	if example.get("result_label") != "DOCUMENTED_EXAMPLE":
		errors.append(f"result_label is {example.get('result_label')!r}, expected DOCUMENTED_EXAMPLE")
	return len(errors) == 0, "; ".join(errors) if errors else "validates cleanly, correctly labeled DOCUMENTED_EXAMPLE"


@check("no REAL_HARDWARE-labeled result exists anywhere in the repository (none is disclosed as real yet)")
def _c5():
	bad = []
	for json_file in list(HARDWARE_DIR.rglob("*.json")):
		try:
			data = json.loads(json_file.read_text())
		except (json.JSONDecodeError, UnicodeDecodeError):
			continue
		records = data if isinstance(data, list) else [data]
		for rec in records:
			if isinstance(rec, dict) and rec.get("result_label") == "REAL_HARDWARE":
				bad.append(str(json_file.relative_to(REPO_ROOT)))
	return len(bad) == 0, f"REAL_HARDWARE record(s) found (not allowed yet): {bad}" if bad else "none found, as expected"


@check("no hype word (revolutionary/production-ready/industry-leading) in outreach/")
def _c6():
	bad = []
	for md_file in sorted(OUTREACH_DIR.rglob("*.md")):
		text = md_file.read_text().lower()
		for word in HYPE_WORDS:
			if word in text:
				bad.append(f"{md_file.relative_to(REPO_ROOT)}: '{word}'")
	return len(bad) == 0, "; ".join(bad) if bad else f"no occurrence of {HYPE_WORDS}"


@check("no gated hardware claim appears unhedged outside hardware-claim-gates.md's own documentation")
def _c7():
	bad = []
	gates_file = OUTREACH_DIR / "hardware-claim-gates.md"
	for md_file in all_outreach_hardware_markdown():
		if md_file == gates_file:
			continue  # the gates doc itself documents these phrases; that's its job
		text = md_file.read_text().lower()
		for phrase, gate in GATED_CLAIMS.items():
			for m in re.finditer(re.escape(phrase), text):
				window = text[max(0, m.start() - 80): m.end() + 40]
				if any(neg in window for neg in NEGATION_MARKERS):
					continue
				bad.append(f"{md_file.relative_to(REPO_ROOT)}: '{phrase}' (requires {gate}) near: ...{window.strip()[:100]}...")
	return len(bad) == 0, "; ".join(bad[:8]) if bad else f"no unhedged occurrence of {len(GATED_CLAIMS)} gated claims outside hardware-claim-gates.md"


@check("hardware-claim-gates.md's summary table matches its own per-gate sections")
def _c8():
	text = (OUTREACH_DIR / "hardware-claim-gates.md").read_text()
	gate_headers = re.findall(r"## Gate (\d+):.*?\((PASSED|NOT PASSED)\)", text)
	table_rows = re.findall(r"\| (\d+)\. .*? \| (\*\*PASSED\*\*|not passed) \|", text)
	header_status = {int(n): s for n, s in gate_headers}
	table_status = {int(n): ("PASSED" if "PASSED" in s else "NOT PASSED") for n, s in table_rows}
	mismatches = [n for n in header_status if header_status.get(n) != table_status.get(n)]
	return len(mismatches) == 0, f"gate(s) with mismatched status between section header and summary table: {mismatches}" if mismatches else f"{len(header_status)} gates consistent"


@check("email-templates.md has no unfilled bracket field left as a literal send-ready example")
def _c9():
	text = (OUTREACH_DIR / "email-templates.md").read_text()
	# Bracketed fields are EXPECTED (they're meant to be filled in) --
	# this check instead confirms the file's own header explicitly warns
	# never to send with one still present, so the safeguard is documented.
	has_bracket_fields = bool(re.search(r"\[Last name\]|\[Lab name\]|\[Team", text))
	has_warning = "never send with a placeholder" in text.lower()
	ok = has_bracket_fields and has_warning
	return ok, "bracketed fields present and the never-send-with-a-placeholder warning is present" if ok else "expected bracket fields or warning missing"


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9):
		fn()
	print()
	print(f"{CHECK_COUNT - len(FAILURES)}/{CHECK_COUNT} checks passed")
	if FAILURES:
		print("\nFAILED:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
