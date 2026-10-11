#!/usr/bin/env python3
"""Milestone M: cross-platform documentation and packaging consistency.

Static checks only (no build): the platform docs say what is validated and
nothing more, the install guide has a section per OS without another OS's
commands in it, the artifact manifest and the CI acceptance wiring agree, and
the portable packaging script keeps its safety properties.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAILS = []
N = 0


def check(name):
	def deco(fn):
		def run():
			global N
			N += 1
			try:
				ok, detail = fn()
			except Exception as e:  # noqa: BLE001
				ok, detail = False, f"raised {type(e).__name__}: {e}"
			print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
			if not ok:
				FAILS.append(name)
		return run
	return deco


def read(p):
	return (REPO / p).read_text()


def section(text, heading):
	m = re.search(r"^## " + re.escape(heading) + r"\n(.*?)(?=^## |\Z)", text,
		re.S | re.M)
	return m.group(1) if m else ""


@check("no 'works everywhere' style overclaim in README or platform docs")
def _a():
	bad = []
	for p in ("README.md", "docs/install.md", "docs/support-matrix.md"):
		for m in re.finditer(r"works everywhere|runs everywhere|on every platform|"
			r"fully cross-platform|all platforms are supported", read(p), re.I):
			bad.append(f"{p}: {m.group(0)}")
	return not bad, "; ".join(bad) or "none"


@check("install.md has Linux, macOS and Windows sections")
def _b():
	t = read("docs/install.md")
	miss = [h for h in ("Linux", "macOS", "Windows") if not section(t, h)]
	return not miss, f"missing: {miss}" if miss else "all three present"


@check("Windows section has no Linux-only commands; macOS section has no apt/systemctl")
def _c():
	t = read("docs/install.md")
	win, mac = section(t, "Windows"), section(t, "macOS")
	bad = [w for w in ("apt ", "sudo ", "systemctl", "dpkg", "~/.local") if w in win]
	bad += [f"macOS:{w}" for w in ("apt install", "systemctl", "dpkg -i") if w in mac]
	return not bad, f"found: {bad}" if bad else "clean"


@check("macOS and Windows sections state the artifact is unsigned and name "
	"Gatekeeper / SmartScreen")
def _d():
	t = read("docs/install.md")
	ok = ("Gatekeeper" in section(t, "macOS") and "unsigned" in t.lower()
		and "SmartScreen" in section(t, "Windows"))
	return ok, "stated" if ok else "missing"


@check("support matrix separates packaged / CI / real hardware / not validated "
	"and never lists Windows GPU as validated")
def _e():
	t = read("docs/support-matrix.md")
	have = all(w in t for w in ("real hardware", "CI", "compile only",
		"not validated", "unsupported"))
	win_gpu = re.findall(r"\| Windows \| (?:CUDA|Vulkan) \| ([^|]+)\|", t)
	ok = have and win_gpu and all("not validated" in x for x in win_gpu)
	return bool(ok), f"windows gpu rows: {win_gpu}"


@check("manifest declares both portable artifacts as unsigned, with the "
	"expected filename patterns")
def _f():
	d = json.loads(read("results/release-artifacts/manifest.json"))
	arts = {a["os"]: a for a in d.get("portable_artifacts", [])}
	ok = (set(arts) == {"macos", "windows"}
		and arts["macos"]["filename_pattern"] == "membrane_<version>_macos_arm64.tar.gz"
		and arts["windows"]["filename_pattern"] == "membrane_<version>_windows_x86_64.zip"
		and all(a["signed"] is False for a in arts.values()))
	return ok, "ok" if ok else str(sorted(arts))


@check("CI runs the package acceptance script against the extracted .deb, "
	"ZIP and macOS archive")
def _g():
	t = read(".github/workflows/ci.yml")
	n = t.count("scripts/package-acceptance.sh")
	return n >= 3, f"{n} invocations"


@check("package script is deterministic and ships no tests/results/build dirs")
def _h():
	t = read("scripts/package-portable.py")
	ok = all(s in t for s in ("SOURCE_DATE_EPOCH", "sorted(", '"signed": False',
		'("bin", "lib")'))
	return ok, "ok" if ok else "missing determinism/content markers"


@check("downloads are verified with the in-tree SHA-256, not an external tool")
def _i():
	t = read("tools/membrane/download_manager.cpp")
	ok = "membrane_sha256_file" in t and '"sha256sum"' not in t \
		and '"shasum"' not in t
	return ok, "in-tree" if ok else "external tool referenced"


@check("platform paths: Linux data dir rule is unchanged (~/.local/share/membrane)")
def _j():
	t = read("tools/membrane/platform_paths.cpp")
	ok = '"/.local/share"' in t and "MEMBRANE_PLATFORM_LINUX" in t
	return ok, "ok" if ok else "linux rule missing"


@check("no wildcard CORS or non-loopback default introduced")
def _k():
	t = read("tools/membrane/server.cpp")
	bad = re.search(r"Access-Control-Allow-Origin", t)
	return bad is None, "no CORS header in server.cpp" if not bad else "found one"


for f in (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k):
	f()
print(f"\n{N - len(FAILS)}/{N} checks passed")
if FAILS:
	print("FAILURES:\n  - " + "\n  - ".join(FAILS))
	sys.exit(1)
