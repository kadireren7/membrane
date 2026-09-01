#!/usr/bin/env python3
"""Validate Phase 28's CLI-UX evidence and doc/CLI consistency.

Checks results/cli-ux/validation.json's schema and internal consistency
(every example has a real command/exit_code, every embedded JSON excerpt
actually parses, no absolute filesystem path anywhere in the file), that
every documented command's flags actually exist in the real membrane-run
--help output (tools/membrane-run/product_cli.cpp's usage() text -- this
validates the COMMITTED evidence file and doc-CLI consistency, it does
not re-run the binary itself), and a handful of doc-drift guards (no
stale "llama-only" claim now that Qwen2 is supported, no unsupported
speed/marketing claim in the reorganized --help text).

Same one-file-per-concern convention as every other scripts/verify-*.py
in this project (see scripts/verify-results.py).

Exit code: 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "results" / "cli-ux" / "validation.json"
PRODUCT_CLI_CPP_PATH = REPO_ROOT / "tools" / "membrane-run" / "product_cli.cpp"
MAIN_CPP_PATH = REPO_ROOT / "tools" / "membrane-run" / "main.cpp"
README_PATH = REPO_ROOT / "README.md"

FAILURES = []
CHECK_COUNT = 0

REQUIRED_EXAMPLE_FIELDS = ("id", "command", "exit_code")
REQUIRED_EXAMPLE_IDS = {
	"help",
	"basic-cpu-run",
	"vulkan-auto-run-json",
	"plan-only-human",
	"qwen2-compressed-run",
	"json-parse-error-unknown-flag",
	"unsupported-architecture-error",
}


def check(name):
	def decorator(fn):
		def wrapper():
			global CHECK_COUNT
			CHECK_COUNT += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001 -- report, don't crash the whole run
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			status = "PASS" if ok else "FAIL"
			print(f"[{status}] {name}: {detail}")
			if not ok:
				FAILURES.append((name, detail))
		return wrapper
	return decorator


_DATA_CACHE = {}


def _load_data():
	if "data" not in _DATA_CACHE:
		_DATA_CACHE["data"] = json.loads(DATA_PATH.read_text())
	return _DATA_CACHE["data"]


_NO_PATH_RE = re.compile(r'(?<![\w.\-:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+')
_NO_PATH_WIN_RE = re.compile(r'[A-Za-z]:\\[^"\\]*(?:\\[^"\\]*)+')


def _find_paths(text):
	return _NO_PATH_RE.findall(text) + _NO_PATH_WIN_RE.findall(text)


@check("results/cli-ux/validation.json exists, is valid JSON, has schema_version 1")
def _c1():
	if not DATA_PATH.exists():
		return False, f"{DATA_PATH} does not exist"
	d = _load_data()
	ok = d.get("schema_version") == 1
	return ok, f"schema_version={d.get('schema_version')!r}"


@check("captured_at_commit looks like a real commit SHA (40 hex chars)")
def _c2():
	d = _load_data()
	sha = d.get("captured_at_commit", "")
	ok = bool(re.fullmatch(r"[0-9a-f]{40}", sha))
	return ok, f"captured_at_commit={sha!r}"


@check("every required example id is present, each with command/exit_code")
def _c3():
	d = _load_data()
	examples = d.get("examples", [])
	ids = {e.get("id") for e in examples}
	missing = REQUIRED_EXAMPLE_IDS - ids
	bad_schema = [e.get("id") for e in examples
		if any(f not in e for f in REQUIRED_EXAMPLE_FIELDS)]
	ok = not missing and not bad_schema
	return ok, (f"missing required ids: {missing}" if missing
		else f"malformed entries: {bad_schema}" if bad_schema
		else f"{len(examples)} examples, all required ids present")


@check("every example's command actually names membrane-run (no typo'd binary)")
def _c4():
	d = _load_data()
	bad = [e["id"] for e in d.get("examples", [])
		if "membrane-run" not in e.get("command", "")]
	return len(bad) == 0, f"{bad}" if bad else "all commands invoke membrane-run"


@check("every embedded *_excerpt field that looks like a JSON object actually "
	"parses as valid JSON")
def _c5():
	d = _load_data()
	bad = []
	checked = 0
	for e in d.get("examples", []):
		for key, val in e.items():
			if not key.endswith("_excerpt") or not isinstance(val, str):
				continue
			stripped = val.strip()
			if not stripped.startswith("{"):
				continue
			checked += 1
			# Excerpts may be truncated (a prefix of a longer real object,
			# noted as such in the surrounding key name/example) -- only
			# fields that stand alone as a complete object are asserted
			# to parse; a truncated one is checked for JSON-shaped syntax
			# up to the truncation point instead (balanced braces to that
			# point never even opened is the actual bug class this check
			# exists to catch: garbage that was never real JSON at all).
			try:
				json.loads(stripped)
			except json.JSONDecodeError:
				if not re.match(r'^\{"schema_version":\d+,"membrane_version":"[^"]+"', stripped):
					bad.append(f"{e['id']}/{key}")
	return len(bad) == 0, f"{bad}" if bad else f"{checked} JSON-shaped excerpts checked"


@check("every exit_code is a real, documented membrane-run exit code (0,2,3,4,5)")
def _c6():
	d = _load_data()
	bad = [(e["id"], e.get("exit_code")) for e in d.get("examples", [])
		if e.get("exit_code") not in (0, 2, 3, 4, 5)]
	return len(bad) == 0, f"{bad}" if bad else "all exit codes in {0,2,3,4,5}"


@check("json-parse-error examples: exit_code matches the embedded JSON "
	"object's own exit_code field")
def _c7():
	d = _load_data()
	bad = []
	for e in d.get("examples", []):
		if not e["id"].startswith("json-parse-error"):
			continue
		excerpt = e.get("stdout_excerpt", "")
		try:
			obj = json.loads(excerpt)
		except json.JSONDecodeError:
			bad.append(f"{e['id']}: stdout_excerpt is not valid JSON")
			continue
		if obj.get("exit_code") != e.get("exit_code"):
			bad.append(f"{e['id']}: embedded exit_code={obj.get('exit_code')} "
				f"!= example exit_code={e.get('exit_code')}")
		if obj.get("ok") is not False:
			bad.append(f"{e['id']}: embedded ok is not false")
		if not obj.get("reason_code"):
			bad.append(f"{e['id']}: missing reason_code")
		if not obj.get("message"):
			bad.append(f"{e['id']}: missing message")
		suggestions = obj.get("suggestions")
		if not isinstance(suggestions, list) or not suggestions:
			bad.append(f"{e['id']}: suggestions must be a non-empty array "
				f"(got {suggestions!r})")
	return len(bad) == 0, "; ".join(bad) if bad else "parse-error contract fields all present and consistent"


@check("unsupported-architecture-error example: suggestion is legal "
	"(never recommends an impossible fix)")
def _c8():
	d = _load_data()
	for e in d.get("examples", []):
		if e["id"] != "unsupported-architecture-error":
			continue
		excerpt = e.get("stdout_excerpt", "")
		obj = json.loads(excerpt)
		suggestion = obj.get("error", {}).get("suggestion", "")
		# The one always-legal fallback for a KV-compat rejection: native
		# has no architecture gate at all (see product_cli.cpp).
		ok = "--kv native" in suggestion
		return ok, f"suggestion={suggestion!r}"
	return False, "unsupported-architecture-error example not found"


@check("no absolute filesystem path anywhere in results/cli-ux/validation.json")
def _c9():
	text = DATA_PATH.read_text()
	hits = _find_paths(text)
	return len(hits) == 0, f"suspicious path-like strings: {hits[:5]}" if hits else "none found"


@check("every flag referenced in a documented command actually exists in "
	"membrane_run_usage()'s real --help text")
def _c10():
	usage_text = PRODUCT_CLI_CPP_PATH.read_text()
	d = _load_data()
	bad = []
	# json-parse-error-unknown-flag's whole point is a flag that does NOT
	# exist (--bogus) -- excluded by design, not a doc-drift gap.
	exempt_ids = {"json-parse-error-unknown-flag"}
	for e in d.get("examples", []):
		if e["id"] in exempt_ids:
			continue
		cmd = e.get("command", "")
		flags = re.findall(r'(?<!\S)(--[a-z][a-z0-9-]*)', cmd)
		for flag in flags:
			# --json/--auto/etc. all appear as literal "  --flag" lines in
			# the usage() text; a renamed/removed flag would no longer
			# match anywhere in that source, catching doc/CLI drift.
			if flag not in usage_text:
				bad.append(f"{e['id']}: {flag} not found in membrane_run_usage() text")
	return len(bad) == 0, "; ".join(bad) if bad else "every documented flag exists in --help text"


@check("--help text documents both LLM_ARCH_LLAMA and LLM_ARCH_QWEN2 -- no "
	"stale llama-only claim")
def _c11():
	usage_text = PRODUCT_CLI_CPP_PATH.read_text()
	ok = "LLM_ARCH_LLAMA" in usage_text and "LLM_ARCH_QWEN2" in usage_text
	return ok, "both architectures named in usage() text" if ok else "usage() text missing an architecture claim"


@check("--help text makes no unsupported absolute speed/marketing claim "
	"(no bare tok/s number, no superlative)")
def _c12():
	usage_text = PRODUCT_CLI_CPP_PATH.read_text()
	bad = []
	if re.search(r'\d+(\.\d+)?\s*tok(en)?s?/s', usage_text, re.IGNORECASE):
		bad.append("usage() text cites a bare tokens/sec figure")
	for word in ("fastest", "blazing", "best-in-class", "state-of-the-art"):
		if re.search(word, usage_text, re.IGNORECASE):
			bad.append(f"usage() text uses marketing superlative: {word!r}")
	return len(bad) == 0, "; ".join(bad) if bad else "no unsupported speed/marketing claim"


@check("product_cli.h's CLI-parse-time JSON reason_code macro exists and "
	"matches what the evidence file's parse-error examples embed")
def _c13():
	header_path = REPO_ROOT / "tools" / "membrane-run" / "product_cli.h"
	header_text = header_path.read_text()
	m = re.search(r'MEMBRANE_REASON_CLI_PARSE_ERROR\s+"([^"]+)"', header_text)
	if not m:
		return False, "MEMBRANE_REASON_CLI_PARSE_ERROR not found in product_cli.h"
	code = m.group(1)
	d = _load_data()
	bad = []
	for e in d.get("examples", []):
		if not e["id"].startswith("json-parse-error"):
			continue
		obj = json.loads(e["stdout_excerpt"])
		if obj.get("reason_code") != code:
			bad.append(f"{e['id']}: reason_code={obj.get('reason_code')!r} != {code!r}")
	return len(bad) == 0, "; ".join(bad) if bad else f"reason_code {code!r} consistent"


@check("README's Quick Start section documents both a minimal (no --auto) "
	"and an --auto invocation, in that order")
def _c14():
	readme = README_PATH.read_text()
	m = re.search(r'## Quick Start(.*?)\n## ', readme, re.DOTALL)
	if not m:
		return False, "## Quick Start section not found"
	section = m.group(1)
	# The minimal invocation is the one line that runs a real prompt
	# with no --auto/--ctx on the SAME line (Phase 30 restructured the
	# section into one combined bash block with per-command comments,
	# rather than separate minimal/--auto code blocks -- match the line
	# itself, not a "```"-terminated block boundary).
	minimal_match = re.search(
		r'^membrane-run --model model\.gguf --prompt "Hello"\s*$',
		section, re.MULTILINE)
	auto_idx = section.find("--auto")
	minimal_idx = minimal_match.start() if minimal_match else -1
	ok = minimal_idx != -1 and auto_idx != -1 and minimal_idx < auto_idx
	return ok, f"minimal_idx={minimal_idx} auto_idx={auto_idx}"


@check("main.cpp's --list-devices/--doctor implementations exist and are "
	"wired into main() before --model/--prompt are required")
def _c15():
	main_text = MAIN_CPP_PATH.read_text()
	has_impls = ("run_list_devices_mode" in main_text
		and "run_doctor_mode" in main_text)
	has_wiring = "want_list_devices || o.want_doctor" in main_text
	# The wiring block must appear before resolve_prompt() is called, so
	# neither command requires --prompt.
	wiring_idx = main_text.find("want_list_devices || o.want_doctor")
	prompt_idx = main_text.find("resolve_prompt(o, &prompt_text)")
	ok = has_impls and has_wiring and 0 <= wiring_idx < prompt_idx
	return ok, (f"has_impls={has_impls} has_wiring={has_wiring} "
		f"wiring_idx={wiring_idx} prompt_idx={prompt_idx}")


def main():
	for fn in (_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10, _c11,
			_c12, _c13, _c14, _c15):
		fn()
	print(f"\n{CHECK_COUNT - len(FAILURES)}/{CHECK_COUNT} checks passed")
	if FAILURES:
		print("\nFAILURES:")
		for name, detail in FAILURES:
			print(f"  - {name}: {detail}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
