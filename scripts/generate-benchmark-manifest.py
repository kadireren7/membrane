#!/usr/bin/env python3
"""Generate benchmarks/MANIFEST.json from the git-tracked artifacts under
benchmarks/. Deterministic: given the same file contents, produces
byte-identical output (sorted keys, fixed field order, no timestamps).

Each entry's SHA-256, size, and git-tracked status are computed fresh from
the working tree; every other field (phase, model, workload, label,
generating_command, related_doc, status) comes from the static ARTIFACTS
table below, which must be updated by hand when a new artifact is added --
this is intentional: label/model/phase cannot be reliably inferred from a
CSV's bytes alone, and a wrong auto-inferred label would be worse than a
maintained one.

Usage:
    scripts/generate-benchmark-manifest.py [--check]

--check: regenerate to a temp buffer and diff against the committed
benchmarks/MANIFEST.json; exit 1 on any difference (used by
scripts/prepare-release.sh and CI) instead of overwriting the file.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "MANIFEST.json"

# label: REAL | SIMULATED | EXTRAPOLATED | ORACLE | ASSUMED
# status: complete | partial | superseded
ARTIFACTS = {
	"benchmarks/results/phase1-store-summary.md": dict(
		phase="phase1", model="n/a", workload="lossless block store + RAW/RLE codec benchmark",
		label="REAL", generating_command="membrane-store-bench (Phase 1)",
		related_doc="docs/architecture.md", status="complete"),

	"benchmarks/kv/prompts/code.txt": dict(
		phase="phase2", model="n/a", workload="prompt fixture: code",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase2-kv-analysis.md", status="complete"),
	"benchmarks/kv/prompts/distractor.txt": dict(
		phase="phase6.2", model="n/a", workload="prompt fixture: distractor (recall-shaped)",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/kv/prompts/longcontext.txt": dict(
		phase="phase6.2", model="n/a", workload="prompt fixture: longcontext (recall-shaped)",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/kv/prompts/natural.txt": dict(
		phase="phase2", model="n/a", workload="prompt fixture: natural language",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase2-kv-analysis.md", status="complete"),
	"benchmarks/kv/prompts/recall.txt": dict(
		phase="phase6.2", model="n/a", workload="prompt fixture: recall (recall-shaped)",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/kv/prompts/repeat.txt": dict(
		phase="phase2", model="n/a", workload="prompt fixture: repeated sentence",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase2-kv-analysis.md", status="complete"),
	"benchmarks/kv/prompts/secrets.txt": dict(
		phase="phase6.2", model="n/a", workload="prompt fixture: secrets (recall-shaped)",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/kv/prompts/short.txt": dict(
		phase="phase2", model="n/a", workload="prompt fixture: short",
		label="REAL", generating_command="n/a (authored fixture)",
		related_doc="docs/phase2-kv-analysis.md", status="complete"),

	"benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16.kvtrace": dict(
		phase="phase6.1", model="SmolLM2-135M", workload="real captured KV-cache trace",
		label="REAL", generating_command="membrane-kv-trace-capture --model models/SmolLM2-135M-Instruct-f16.gguf",
		related_doc="docs/phase6-cxl-near-memory.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-360M-Instruct-f16.kvtrace": dict(
		phase="phase6.1", model="SmolLM2-360M", workload="real captured KV-cache trace",
		label="REAL", generating_command="membrane-kv-trace-capture --model models/SmolLM2-360M-Instruct-f16.gguf",
		related_doc="docs/phase6-cxl-near-memory.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16.attntrace": dict(
		phase="phase6.2", model="SmolLM2-135M", workload="real captured attention trace",
		label="REAL", generating_command="membrane-kv-attn-trace-capture --model models/SmolLM2-135M-Instruct-f16.gguf",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-360M-Instruct-f16.attntrace": dict(
		phase="phase6.2", model="SmolLM2-360M", workload="real captured attention trace",
		label="REAL", generating_command="membrane-kv-attn-trace-capture --model models/SmolLM2-360M-Instruct-f16.gguf",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16-long.attntrace": dict(
		phase="phase6.3", model="SmolLM2-135M", workload="real attention trace, extended capture length",
		label="REAL", generating_command="membrane-kv-attn-trace-capture --model models/SmolLM2-135M-Instruct-f16.gguf --n-tokens <long>",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-360M-Instruct-f16-long.attntrace": dict(
		phase="phase6.3", model="SmolLM2-360M", workload="real attention trace, extended capture length",
		label="REAL", generating_command="membrane-kv-attn-trace-capture --model models/SmolLM2-360M-Instruct-f16.gguf --n-tokens <long>",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16-long.attntrace.manifest.json": dict(
		phase="phase6.3", model="SmolLM2-135M", workload="trace-hash sidecar for the long capture above",
		label="REAL", generating_command="membrane-kv-attn-trace-capture (auto-emitted sidecar)",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),
	"benchmarks/cxl-sim/traces/SmolLM2-360M-Instruct-f16-long.attntrace.manifest.json": dict(
		phase="phase6.3", model="SmolLM2-360M", workload="trace-hash sidecar for the long capture above",
		label="REAL", generating_command="membrane-kv-attn-trace-capture (auto-emitted sidecar)",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),

	"benchmarks/cxl-sim/attn-quality-report.csv": dict(
		phase="phase6.2", model="SmolLM2-135M + SmolLM2-360M", workload="attention-prediction quality vs. ground truth",
		label="REAL", generating_command="membrane-kv-attn-quality",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/cxl-sim/sweep-report.csv": dict(
		phase="phase6.1", model="SmolLM2-135M", workload="near-memory/CXL appliance sweep (concurrency x context x capacity)",
		label="SIMULATED", generating_command="membrane-cxl-sim --trace benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16.kvtrace",
		related_doc="docs/phase6-cxl-near-memory.md", status="complete"),
	"benchmarks/cxl-sim/workingset-sweep.csv": dict(
		phase="phase6.2", model="SmolLM2-135M + SmolLM2-360M", workload="attention working-set size sweep",
		label="SIMULATED", generating_command="membrane-kv-workingset-sim --trace-135m ... --trace-360m ...",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/cxl-sim/workingset-context-sweep.csv": dict(
		phase="phase6.2", model="SmolLM2-135M + SmolLM2-360M", workload="working-set size vs. context length sweep",
		label="SIMULATED", generating_command="membrane-kv-workingset-sim (context sweep mode)",
		related_doc="docs/phase6-attention-working-set.md", status="complete"),
	"benchmarks/cxl-sim/exact-sweep.csv": dict(
		phase="phase6.3", model="SmolLM2-135M", workload="exact sparse KV retrieval sweep, 4K-context capacity-bound scenario",
		label="SIMULATED", generating_command="membrane-kv-exact-sim (Phase 6.3 configuration)",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),
	"benchmarks/cxl-sim/exact-sweep-layer-head-detail.csv": dict(
		phase="phase6.3", model="SmolLM2-135M", workload="per-layer/per-head predictor accuracy detail",
		label="REAL", generating_command="membrane-kv-exact-sim (layer/head detail mode)",
		related_doc="docs/phase6-exact-sparse-retrieval.md", status="complete"),
	"benchmarks/cxl-sim/unified-sweep.csv": dict(
		phase="phase6.4/6.5", model="SmolLM2-135M + SmolLM2-360M", workload="unified 128K-context x 512-concurrency sweep, 462/462 scenarios",
		label="SIMULATED", generating_command="membrane-kv-exact-sim --backend streaming (see docs/reproduction.md Level 3)",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
	"benchmarks/cxl-sim/unified-sweep-hardware-sensitivity.csv": dict(
		phase="phase6.4", model="SmolLM2-135M", workload="CXL link / quant-pipeline hardware sensitivity, 10 points",
		label="SIMULATED", generating_command="membrane-cxl-sim (hardware-sensitivity mode)",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
	"benchmarks/cxl-sim/unified-sweep-layer-head-detail.csv": dict(
		phase="phase6.4/6.5", model="SmolLM2-135M + SmolLM2-360M", workload="per-layer/per-head predictor accuracy, unified sweep",
		label="REAL", generating_command="membrane-kv-exact-sim (layer/head detail mode)",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
	"benchmarks/cxl-sim/unified-tail-samples.csv": dict(
		phase="phase6.4/6.5", model="SmolLM2-135M + SmolLM2-360M", workload="tail-latency raw samples, 900 per model",
		label="SIMULATED", generating_command="membrane-kv-exact-sim --tail-recovery-only (see docs/phase6-out-of-core-simulator.md)",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
	"benchmarks/cxl-sim/unified-tail-recovery-135m.csv": dict(
		phase="phase6.5", model="SmolLM2-135M", workload="tail-sample recovery sweep (45 scenarios, replaces data lost to a Phase 6.4 CSV-truncation bug)",
		label="SIMULATED", generating_command="membrane-kv-exact-sim --tail-recovery-only --tail-recovery-model SmolLM2-135M",
		related_doc="docs/phase6-out-of-core-simulator.md", status="complete"),
	"benchmarks/cxl-sim/unified-tail-recovery-360m.csv": dict(
		phase="phase6.5", model="SmolLM2-360M", workload="tail-sample recovery sweep (45 scenarios, 34 pre-seeded from the main run to avoid recompute)",
		label="SIMULATED", generating_command="membrane-kv-exact-sim --tail-recovery-only --tail-recovery-model SmolLM2-360M",
		related_doc="docs/phase6-out-of-core-simulator.md", status="complete"),

	"benchmarks/cxl-sim/quality-reverify/phase6.4-135m.jsonl": dict(
		phase="phase6.4", model="SmolLM2-135M", workload="quality re-verification snapshot",
		label="REAL", generating_command="membrane-kv-quality (Phase 6.4 re-verify pass)",
		related_doc="docs/phase6-unified-stress.md", status="superseded"),
	"benchmarks/cxl-sim/quality-reverify/phase6.4-360m.jsonl": dict(
		phase="phase6.4", model="SmolLM2-360M", workload="quality re-verification snapshot",
		label="REAL", generating_command="membrane-kv-quality (Phase 6.4 re-verify pass)",
		related_doc="docs/phase6-unified-stress.md", status="superseded"),
	"benchmarks/cxl-sim/quality-reverify/quality-reverify-135m.jsonl": dict(
		phase="phase6.4", model="SmolLM2-135M", workload="quality re-verification, current",
		label="REAL", generating_command="membrane-kv-quality",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
	"benchmarks/cxl-sim/quality-reverify/quality-reverify-360m.jsonl": dict(
		phase="phase6.4", model="SmolLM2-360M", workload="quality re-verification, current",
		label="REAL", generating_command="membrane-kv-quality",
		related_doc="docs/phase6-unified-stress.md", status="complete"),
}


def sha256_of(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def git_tracked_files(repo_root: Path) -> set:
	out = subprocess.run(
		["git", "-C", str(repo_root), "ls-files", "benchmarks/"],
		capture_output=True, text=True, check=True,
	).stdout
	return set(out.splitlines())


def build_manifest() -> dict:
	tracked = git_tracked_files(REPO_ROOT)
	entries = []
	missing = []
	for rel_path, meta in ARTIFACTS.items():
		if rel_path not in tracked:
			missing.append(rel_path)
			continue
		abs_path = REPO_ROOT / rel_path
		entry = {
			"path": rel_path,
			"phase": meta["phase"],
			"model": meta["model"],
			"workload": meta["workload"],
			"label": meta["label"],
			"sha256": sha256_of(abs_path),
			"size_bytes": abs_path.stat().st_size,
			"generating_command": meta["generating_command"],
			"related_doc": meta["related_doc"],
			"status": meta["status"],
		}
		entries.append(entry)
	if missing:
		print("generate-benchmark-manifest.py: ARTIFACTS table names paths "
			"not tracked by git (removed or renamed?): " + ", ".join(missing),
			file=sys.stderr)
		sys.exit(1)
	entries.sort(key=lambda e: e["path"])
	return {
		"schema_version": 1,
		"labels": ["REAL", "SIMULATED", "EXTRAPOLATED", "ORACLE", "ASSUMED"],
		"statuses": ["complete", "partial", "superseded"],
		"artifact_count": len(entries),
		"artifacts": entries,
	}


def main() -> int:
	check_only = "--check" in sys.argv
	manifest = build_manifest()
	text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
	if check_only:
		current = MANIFEST_PATH.read_text() if MANIFEST_PATH.exists() else ""
		if current != text:
			print("generate-benchmark-manifest.py --check: "
				"benchmarks/MANIFEST.json is out of date; "
				"run scripts/generate-benchmark-manifest.py to regenerate it.",
				file=sys.stderr)
			return 1
		print("generate-benchmark-manifest.py --check: up to date.")
		return 0
	MANIFEST_PATH.write_text(text)
	print(f"wrote {MANIFEST_PATH} ({manifest['artifact_count']} artifacts)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
