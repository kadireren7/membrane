#!/bin/bash
# Reproduces the committed results/v0.2/smollm2-q8-memory.json artifact:
# native vs. q8 KV storage, real process memory + quality, at contexts
# 512/1024/2048/4096/8192, using membrane-run itself (not a diagnostic
# tool) as the measurement instrument.
#
# Usage:
#   scripts/benchmark-v0.2.sh MODEL_PATH [OUTPUT_JSON]
#
# MODEL_PATH must already exist locally -- this script never downloads
# a model. OUTPUT_JSON defaults to a scratch path; it is NEVER the
# committed results/v0.2/smollm2-q8-memory.json unless you pass that
# path explicitly, so a normal run cannot accidentally overwrite the
# canonical artifact.
#
# Environment:
#   MEMBRANE_RUN_BIN   path to the membrane-run binary
#                      (default: ./build-llama/tools/membrane-run/membrane-run)
#   MEMBRANE_PROMPT    prompt file (default: benchmarks/kv/prompts/short.txt)
#   MEMBRANE_GEN_TOKENS  tokens generated per run (default: 32)
#   MEMBRANE_CONTEXTS  space-separated context sizes
#                      (default: "512 1024 2048 4096 8192")
set -euo pipefail

MODEL="${1:?Usage: $0 MODEL_PATH [OUTPUT_JSON]}"
OUT="${2:-/tmp/membrane-v0.2-benchmark.json}"
BIN="${MEMBRANE_RUN_BIN:-./build-llama/tools/membrane-run/membrane-run}"
PROMPT="${MEMBRANE_PROMPT:-benchmarks/kv/prompts/short.txt}"
GEN_TOKENS="${MEMBRANE_GEN_TOKENS:-32}"
CONTEXTS="${MEMBRANE_CONTEXTS:-512 1024 2048 4096 8192}"

if [ ! -x "$BIN" ]; then
	echo "error: membrane-run binary not found/executable at $BIN" >&2
	echo "build it first: cmake --build build-llama --target membrane-run" >&2
	exit 1
fi
if [ ! -f "$MODEL" ]; then
	echo "error: model file not found: $MODEL" >&2
	exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "[" > "$WORKDIR/rows.jsonl.tmp"
first=1
for ctx in $CONTEXTS; do
	echo "=== ctx=$ctx ===" >&2
	"$BIN" --model "$MODEL" --prompt-file "$PROMPT" --ctx "$ctx" \
		--kv native --gen-tokens "$GEN_TOKENS" --json \
		> "$WORKDIR/native_${ctx}.json" 2>"$WORKDIR/native_${ctx}.err"
	"$BIN" --model "$MODEL" --prompt-file "$PROMPT" --ctx "$ctx" \
		--kv q8 --gen-tokens "$GEN_TOKENS" --json \
		> "$WORKDIR/q8_${ctx}.json" 2>"$WORKDIR/q8_${ctx}.err"
	"$BIN" --model "$MODEL" --prompt-file "$PROMPT" --ctx "$ctx" \
		--compare-kv --gen-tokens "$GEN_TOKENS" --json \
		> "$WORKDIR/compare_${ctx}.json" 2>"$WORKDIR/compare_${ctx}.err"
done

python3 "$(dirname "$0")/assemble-v0.2-artifact.py" \
	--workdir "$WORKDIR" --contexts "$CONTEXTS" --model "$MODEL" \
	--prompt "$PROMPT" --gen-tokens "$GEN_TOKENS" --output "$OUT"

echo "wrote $OUT" >&2
if [ "${3:-}" = "--verify" ] || [ "${MEMBRANE_VERIFY:-0}" = "1" ]; then
	python3 "$(dirname "$0")/verify-results.py" --artifact "$OUT"
fi
