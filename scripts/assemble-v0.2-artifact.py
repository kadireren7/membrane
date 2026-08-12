#!/usr/bin/env python3
"""Assembles the v0.2 compressed-KV-storage memory/quality artifact from
per-context membrane-run --json outputs (see scripts/benchmark-v0.2.sh).

Deliberately sanitizes everything that goes into the committed artifact:
- model_label/prompt_fixture are already basenames (membrane-run itself
  never emits an absolute path or prompt text in --json output).
- host_label is a fixed, generic string -- never the real hostname.
- No raw logits/KV, no absolute paths, anywhere in the output.
"""
import argparse
import json
import subprocess
from pathlib import Path


def git_sha() -> str:
	try:
		return subprocess.check_output(
			["git", "rev-parse", "HEAD"], text=True
		).strip()
	except Exception:
		return "unknown"


def load(path: Path) -> dict:
	return json.loads(path.read_text())


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--workdir", required=True)
	ap.add_argument("--contexts", required=True)
	ap.add_argument("--model", required=True)
	ap.add_argument("--prompt", required=True)
	ap.add_argument("--gen-tokens", required=True, type=int)
	ap.add_argument("--output", required=True)
	args = ap.parse_args()

	workdir = Path(args.workdir)
	contexts = [int(c) for c in args.contexts.split()]

	model_label = None
	prompt_fixture = None
	membrane_version = None
	rows = []
	for ctx in contexts:
		native = load(workdir / f"native_{ctx}.json")
		q8 = load(workdir / f"q8_{ctx}.json")
		compare = load(workdir / f"compare_{ctx}.json")

		model_label = model_label or native["model_label"]
		prompt_fixture = prompt_fixture or Path(args.prompt).name
		for label, run in (("native", native), ("q8", q8), ("compare", compare)):
			v = run["membrane_version"]
			if membrane_version is None:
				membrane_version = v
			elif v != membrane_version:
				raise SystemExit(
					f"membrane_version mismatch at ctx={ctx} ({label}): "
					f"expected {membrane_version!r}, got {v!r} -- inputs "
					"were not all produced by the same membrane-run build"
				)

		native_bytes = compare["storage"]["native_kv_allocated_bytes"]
		q8_bytes = compare["storage"]["q8_kv_allocated_bytes"]
		native_rss = native["memory"]["rss_after_context_kb"]
		q8_rss = q8["memory"]["rss_after_context_kb"]
		reduction_pct = 100.0 * (native_rss - q8_rss) / native_rss

		rows.append({
			"ctx": ctx,
			"native_kv_allocated_bytes": native_bytes,
			"q8_kv_allocated_bytes": q8_bytes,
			"kv_bytes_ratio": round(native_bytes / q8_bytes, 6),
			"native_rss_after_context_kb": native_rss,
			"q8_rss_after_context_kb": q8_rss,
			"rss_reduction_pct": round(reduction_pct, 4),
			"quality": {
				"token_identity": compare["quality"]["token_identity"],
				"first_divergence": compare["quality"]["first_divergence"],
				"logit_rel_l2": compare["quality"]["logit_rel_l2"],
				"top1_preservation": compare["quality"]["top1_preservation"],
				"delta_nll": compare["quality"]["delta_nll"],
			},
			"performance": {
				"native_generation_tok_per_s": native["performance"]["generation_tok_per_s"],
				"q8_generation_tok_per_s": q8["performance"]["generation_tok_per_s"],
			},
		})

	artifact = {
		"schema_version": 1,
		"membrane_version": membrane_version,
		"git_sha": git_sha(),
		"model_label": model_label,
		"prompt_fixture": prompt_fixture,
		"gen_tokens": args.gen_tokens,
		"host_label": "dev-linux-x86_64-single-host",
		"methodology": (
			"native and q8 KV storage measured via two separate, single-"
			"context membrane-run invocations each (no cross-process RSS "
			"contamination -- see docs/live-runtime.md's Phase 7 "
			"methodology-fix note); quality via membrane-run --compare-kv "
			"(native free-running reference vs. q8 teacher-forced on the "
			"same reference tokens, aligned per-step logit/NLL "
			"comparison). CPU-only, single host, LLM_ARCH_LLAMA only, "
			"greedy decoding."
		),
		"contexts": rows,
	}
	out_path = Path(args.output)
	out_path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
