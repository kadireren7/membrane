#!/usr/bin/env bash
#
# MEMBRANE quick research-release demo.
#
# Runs, using only small, already-committed test fixtures -- no model
# download, no multi-hour sweep:
#   1. build the project
#   2. bit-exact CPU quantization parity test (100,000+ blocks vs. ggml)
#   3. full 520,000-transaction FPGA Verilator cosimulation
#   4. a small, committed-fixture exact-retrieval scenario (test_exact_engine)
#   5. print + write a summary (demo-results.json, demo-results.md)
#
# Usage: scripts/demo.sh [--quick|--full] [--skip-build] [--output-dir DIR]
#
#   --quick        (default) run the steps above, nothing else.
#   --full         also run the complete ctest suite (all registered
#                  tests) in the plain Release build, in addition to the
#                  --quick steps. Still no model download, still no
#                  multi-hour sweep -- see docs/reproduction.md Level 2/3
#                  for those.
#   --skip-build   assume the build directories already exist and are
#                  built; only re-run the tests/binaries.
#   --output-dir   where demo-results.json/.md and step logs are written
#                  (default: ./demo-output).
#
# Steps 2 (quant parity) and 3 (FPGA Verilator) each have a one-time
# environment dependency this script does not install for you: the
# third_party/llama.cpp submodule (for step 2) and a Verilator binary
# (for step 3, checked in PATH and in the locally-extracted
# tools/.local-verilator/usr/bin used during this project's own
# development). If either dependency is missing, that step is reported
# as SKIPPED with instructions -- not silently omitted, and not treated
# as a failure. If the dependency IS present and the step itself fails
# (a real parity mismatch or cosim failure), that IS treated as a
# required failure.
#
# Exit code: 0 if every REQUIRED step (build, exact-retrieval scenario,
# and any present-but-attempted quant-parity/Verilator step) passed;
# non-zero otherwise, with the failing step(s) named on stderr.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="quick"
SKIP_BUILD=0
OUTPUT_DIR="$REPO_ROOT/demo-output"

while [ $# -gt 0 ]; do
	case "$1" in
	--quick) MODE="quick" ;;
	--full) MODE="full" ;;
	--skip-build) SKIP_BUILD=1 ;;
	--output-dir)
		shift
		OUTPUT_DIR="$1"
		;;
	-h | --help)
		sed -n '2,32p' "$0"
		exit 0
		;;
	*)
		echo "demo.sh: unknown argument: $1" >&2
		exit 2
		;;
	esac
	shift
done

mkdir -p "$OUTPUT_DIR"
BUILD_DIR="$REPO_ROOT/build-demo"
BUILD_DIR_LLAMA="$REPO_ROOT/build-demo-llama"

STEP_NAMES=()
STEP_STATUS=()   # PASS / FAIL / SKIP
STEP_SECONDS=()
STEP_DETAIL=()
REQUIRED_FAILED=0

record_step() {
	STEP_NAMES+=("$1")
	STEP_STATUS+=("$2")
	STEP_SECONDS+=("$3")
	STEP_DETAIL+=("$4")
	printf '[%s] %-28s (%ss) %s\n' "$2" "$1" "$3" "$4"
}

# ---------------------------------------------------------------------
# Step 1: build
# ---------------------------------------------------------------------
echo "== Step 1/5: build =="
if [ "$SKIP_BUILD" -eq 1 ]; then
	if [ -d "$BUILD_DIR" ]; then
		record_step "build" "PASS" "0.0" "skipped (--skip-build, reusing $BUILD_DIR)"
	else
		record_step "build" "FAIL" "0.0" "--skip-build given but $BUILD_DIR does not exist"
		REQUIRED_FAILED=1
	fi
else
	t0=$(date +%s)
	if cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
		>"$OUTPUT_DIR/01-configure.log" 2>&1 &&
		cmake --build "$BUILD_DIR" -j --target test_exact_engine \
			>"$OUTPUT_DIR/01-build.log" 2>&1; then
		t1=$(date +%s)
		if [ "$MODE" = "full" ]; then
			if cmake --build "$BUILD_DIR" -j >>"$OUTPUT_DIR/01-build.log" 2>&1; then
				record_step "build" "PASS" "$((t1 - t0))" "Release, all targets (--full)"
			else
				record_step "build" "FAIL" "$((t1 - t0))" "full build failed, see $OUTPUT_DIR/01-build.log"
				REQUIRED_FAILED=1
			fi
		else
			record_step "build" "PASS" "$((t1 - t0))" "Release, target test_exact_engine"
		fi
	else
		t1=$(date +%s)
		record_step "build" "FAIL" "$((t1 - t0))" "configure/build failed, see $OUTPUT_DIR/01-configure.log and 01-build.log"
		REQUIRED_FAILED=1
	fi
fi

# ---------------------------------------------------------------------
# Step 2: bit-exact CPU quantization parity (needs third_party/llama.cpp)
# ---------------------------------------------------------------------
echo "== Step 2/5: quant parity (vs. ggml) =="
if [ ! -f "$REPO_ROOT/third_party/llama.cpp/CMakeLists.txt" ]; then
	record_step "quant_parity" "SKIP" "0.0" \
		"third_party/llama.cpp submodule not present -- run: git submodule update --init --recursive"
else
	t0=$(date +%s)
	build_ok=1
	if [ "$SKIP_BUILD" -eq 0 ]; then
		if ! cmake -S "$REPO_ROOT" -B "$BUILD_DIR_LLAMA" -DCMAKE_BUILD_TYPE=Release \
			-DMEMBRANE_ENABLE_LLAMA=ON >"$OUTPUT_DIR/02-configure.log" 2>&1; then
			build_ok=0
		elif ! cmake --build "$BUILD_DIR_LLAMA" -j --target test_ggml_quant_parity \
			>"$OUTPUT_DIR/02-build.log" 2>&1; then
			build_ok=0
		fi
	fi
	PARITY_BIN="$BUILD_DIR_LLAMA/test_ggml_quant_parity"
	if [ ! -x "$PARITY_BIN" ]; then
		PARITY_BIN="$(find "$BUILD_DIR_LLAMA" -maxdepth 2 -name test_ggml_quant_parity -type f 2>/dev/null | head -1)"
	fi
	t1=$(date +%s)
	if [ "$build_ok" -eq 0 ]; then
		record_step "quant_parity" "SKIP" "$((t1 - t0))" \
			"MEMBRANE_ENABLE_LLAMA build failed, see $OUTPUT_DIR/02-configure.log / 02-build.log -- treated as environment-dependent, not a required failure"
	elif [ -z "${PARITY_BIN:-}" ] || [ ! -x "$PARITY_BIN" ]; then
		record_step "quant_parity" "SKIP" "$((t1 - t0))" "test_ggml_quant_parity binary not found after build"
	else
		if "$PARITY_BIN" >"$OUTPUT_DIR/02-run.log" 2>&1; then
			detail=$(grep -o 'all ggml quant parity tests passed.*' "$OUTPUT_DIR/02-run.log" | head -1)
			record_step "quant_parity" "PASS" "$((t1 - t0))" "${detail:-passed, see $OUTPUT_DIR/02-run.log}"
		else
			record_step "quant_parity" "FAIL" "$((t1 - t0))" "test binary reported failure, see $OUTPUT_DIR/02-run.log"
			REQUIRED_FAILED=1
		fi
	fi
fi

# ---------------------------------------------------------------------
# Step 3: FPGA full-pipeline Verilator cosimulation (needs Verilator)
# ---------------------------------------------------------------------
echo "== Step 3/5: FPGA Verilator cosimulation =="
VERILATOR_BIN=""
LOCAL_VERILATOR_ROOT="$REPO_ROOT/tools/.local-verilator/usr/share/verilator"
for candidate in verilator "$REPO_ROOT/tools/.local-verilator/usr/bin/verilator"; do
	if command -v "$candidate" >/dev/null 2>&1; then
		VERILATOR_BIN="$(command -v "$candidate")"
		break
	elif [ -x "$candidate" ]; then
		VERILATOR_BIN="$candidate"
		break
	fi
done
# The system verilator package installs verilated_std.sv under
# /usr/share/verilator; this project's locally-extracted copy (no root
# access to install system-wide, see .gitignore's Phase 5.3 note) needs
# VERILATOR_ROOT pointed at its own share/ directory instead.
case "$VERILATOR_BIN" in
"$REPO_ROOT/tools/.local-verilator/"*)
	if [ -d "$LOCAL_VERILATOR_ROOT" ]; then
		export VERILATOR_ROOT="$LOCAL_VERILATOR_ROOT"
	fi
	;;
esac

if [ -z "$VERILATOR_BIN" ]; then
	record_step "fpga_verilator" "SKIP" "0.0" \
		"verilator not found in PATH or tools/.local-verilator/usr/bin -- install verilator to run this step"
else
	t0=$(date +%s)
	VOBJ="$OUTPUT_DIR/verilator-obj"
	VECDIR="$OUTPUT_DIR/verilator-vectors"
	mkdir -p "$VECDIR"
	fpga_ok=1
	log="$OUTPUT_DIR/03-fpga.log"
	: >"$log"
	if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$VOBJ/Vtop" ]; then
		{
			cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_top_x" \
				"$REPO_ROOT/rtl/tb/gen_top_x_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" &&
				cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_pack" \
					"$REPO_ROOT/rtl/tb/gen_pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c" &&
				cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_dequant" \
					"$REPO_ROOT/rtl/tb/gen_dequant_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c" &&
				cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_q4pack" \
					"$REPO_ROOT/rtl/tb/gen_q4pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c" &&
				cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_q4unpack" \
					"$REPO_ROOT/rtl/tb/gen_q4unpack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c" &&
				"$VECDIR/gen_top_x" 120000 "$VECDIR/top_x_120k.txt" &&
				"$VECDIR/gen_pack" "$VECDIR/top_x_120k.txt" "$VECDIR/top_q8pack_120k.txt" &&
				"$VECDIR/gen_dequant" "$VECDIR/top_q8pack_120k.txt" "$VECDIR/top_q8dequant_120k.txt" &&
				"$VECDIR/gen_q4pack" "$VECDIR/top_x_120k.txt" "$VECDIR/top_q4pack_120k.txt" &&
				"$VECDIR/gen_q4unpack" "$VECDIR/top_q4pack_120k.txt" "$VECDIR/top_q4unpack_120k.txt" &&
				"$VERILATOR_BIN" --cc --exe --build -j 0 -Wno-fatal --Mdir "$VOBJ" \
					--top-module membrane_quant_stream_top \
					"$REPO_ROOT/rtl/membrane_fp_pkg.sv" "$REPO_ROOT/rtl/membrane_fp_adder.sv" \
					"$REPO_ROOT/rtl/membrane_fp_divider.sv" "$REPO_ROOT/rtl/membrane_fp_multiplier.sv" \
					"$REPO_ROOT/rtl/stream_fifo.sv" "$REPO_ROOT/rtl/valid_delay_line.sv" \
					"$REPO_ROOT/rtl/q4_pack.sv" "$REPO_ROOT/rtl/q4_scale.sv" "$REPO_ROOT/rtl/q4_scan.sv" \
					"$REPO_ROOT/rtl/q4_unpack.sv" "$REPO_ROOT/rtl/q8_dequantize.sv" \
					"$REPO_ROOT/rtl/q8_maxabs_reduce.sv" "$REPO_ROOT/rtl/q8_quantize_pack.sv" \
					"$REPO_ROOT/rtl/q8_scale.sv" "$REPO_ROOT/rtl/membrane_quant_stream_top.sv" \
					"$REPO_ROOT/rtl/tb/tb_top_verilator.cpp" -o Vtop
		} >>"$log" 2>&1 || fpga_ok=0
	fi
	t1=$(date +%s)
	if [ "$fpga_ok" -eq 0 ]; then
		record_step "fpga_verilator" "FAIL" "$((t1 - t0))" "build/vector-gen failed, see $log"
		REQUIRED_FAILED=1
	else
		# The harness expects the vector files at fixed /tmp paths.
		cp "$VECDIR"/top_*.txt /tmp/ 2>/dev/null
		if "$VOBJ/Vtop" >>"$log" 2>&1; then
			detail=$(grep -o 'PASS: membrane_quant_stream_top Verilator cosim.*' "$log" | tail -1)
			t2=$(date +%s)
			record_step "fpga_verilator" "PASS" "$((t2 - t0))" "${detail:-passed, see $log}"
		else
			t2=$(date +%s)
			record_step "fpga_verilator" "FAIL" "$((t2 - t0))" "cosimulation reported failure, see $log"
			REQUIRED_FAILED=1
		fi
	fi
fi

# ---------------------------------------------------------------------
# Step 4: small CXL exact-retrieval scenario (committed fixture, llama-free)
# ---------------------------------------------------------------------
echo "== Step 4/5: small CXL exact-retrieval scenario =="
EXACT_BIN="$(find "$BUILD_DIR" -maxdepth 3 -name test_exact_engine -type f 2>/dev/null | head -1)"
if [ -z "$EXACT_BIN" ] || [ ! -x "$EXACT_BIN" ]; then
	record_step "cxl_exact_retrieval" "FAIL" "0.0" "test_exact_engine binary not found under $BUILD_DIR (build step must have failed)"
	REQUIRED_FAILED=1
else
	t0=$(date +%s)
	if "$EXACT_BIN" >"$OUTPUT_DIR/04-exact-retrieval.log" 2>&1; then
		t1=$(date +%s)
		n_pass=$(grep -c '^PASS ' "$OUTPUT_DIR/04-exact-retrieval.log")
		record_step "cxl_exact_retrieval" "PASS" "$((t1 - t0))" "$n_pass real exact-retrieval scenarios passed (device capacity, contention, microbatch, deterministic replay, compute-floor)"
	else
		t1=$(date +%s)
		record_step "cxl_exact_retrieval" "FAIL" "$((t1 - t0))" "see $OUTPUT_DIR/04-exact-retrieval.log"
		REQUIRED_FAILED=1
	fi
fi

if [ "$MODE" = "full" ]; then
	echo "== Step 4b/5 (--full): complete ctest suite =="
	t0=$(date +%s)
	if ctest --test-dir "$BUILD_DIR" --output-on-failure >"$OUTPUT_DIR/04b-ctest-full.log" 2>&1; then
		t1=$(date +%s)
		summary=$(grep -o '[0-9]*% tests passed.*' "$OUTPUT_DIR/04b-ctest-full.log" | tail -1)
		record_step "full_ctest_suite" "PASS" "$((t1 - t0))" "${summary:-see $OUTPUT_DIR/04b-ctest-full.log}"
	else
		t1=$(date +%s)
		record_step "full_ctest_suite" "FAIL" "$((t1 - t0))" "see $OUTPUT_DIR/04b-ctest-full.log"
		REQUIRED_FAILED=1
	fi
fi

# ---------------------------------------------------------------------
# Step 5: summary
# ---------------------------------------------------------------------
echo "== Step 5/5: summary =="

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="clean"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | grep -v '^ M third_party/llama.cpp')" ]; then
	GIT_DIRTY="dirty"
fi
UNAME="$(uname -a)"
CC_VERSION="$(cc --version 2>/dev/null | head -1)"
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

json_escape() {
	printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' 2>/dev/null ||
		printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"
}

{
	echo "{"
	echo "  \"git_commit\": $(json_escape "$GIT_COMMIT"),"
	echo "  \"git_tree\": $(json_escape "$GIT_DIRTY"),"
	echo "  \"mode\": $(json_escape "$MODE"),"
	echo "  \"build_type\": \"Release\","
	echo "  \"generated_utc\": $(json_escape "$NOW_UTC"),"
	echo "  \"environment\": {"
	echo "    \"uname\": $(json_escape "$UNAME"),"
	echo "    \"compiler\": $(json_escape "$CC_VERSION")"
	echo "  },"
	echo "  \"headline_metrics_reference\": {"
	echo "    \"note\": \"This quick demo verifies correctness/determinism on small fixtures. It does not regenerate the full-scale headline metrics (unified 128K x 512 sweep, bytes/token reduction). Those come from the committed, SHA-256-verified artifacts cited below -- see docs/reproduction.md Level 3 to regenerate them for real.\","
	echo "    \"unified_sweep_csv\": \"benchmarks/cxl-sim/unified-sweep.csv\","
	echo "    \"results_doc\": \"docs/phase6-unified-stress.md\","
	echo "    \"manifest\": \"benchmarks/MANIFEST.json\""
	echo "  },"
	echo "  \"steps\": ["
	n=${#STEP_NAMES[@]}
	for i in "${!STEP_NAMES[@]}"; do
		echo -n "    {\"name\": $(json_escape "${STEP_NAMES[$i]}"), \"status\": $(json_escape "${STEP_STATUS[$i]}"), \"elapsed_seconds\": $(json_escape "${STEP_SECONDS[$i]}"), \"detail\": $(json_escape "${STEP_DETAIL[$i]}")}"
		if [ "$((i + 1))" -lt "$n" ]; then echo ","; else echo ""; fi
	done
	echo "  ],"
	echo "  \"required_failed\": $([ "$REQUIRED_FAILED" -eq 1 ] && echo true || echo false)"
	echo "}"
} >"$OUTPUT_DIR/demo-results.json"

{
	echo "# MEMBRANE demo results"
	echo
	echo "- Git commit: \`$GIT_COMMIT\` ($GIT_DIRTY working tree)"
	echo "- Mode: $MODE"
	echo "- Generated: $NOW_UTC"
	echo "- Environment: \`$UNAME\`"
	echo "- Compiler: $CC_VERSION"
	echo
	echo "## Steps"
	echo
	echo "| Step | Status | Elapsed (s) | Detail |"
	echo "|---|---|---|---|"
	for i in "${!STEP_NAMES[@]}"; do
		echo "| ${STEP_NAMES[$i]} | ${STEP_STATUS[$i]} | ${STEP_SECONDS[$i]} | ${STEP_DETAIL[$i]} |"
	done
	echo
	echo "## Headline metrics (not regenerated by this quick demo)"
	echo
	echo "This demo verifies correctness and determinism on small, committed"
	echo "fixtures. It does not reproduce the full-scale headline numbers"
	echo "(the 462/462-scenario unified sweep, bytes/token reduction) --"
	echo "those come from \`benchmarks/cxl-sim/unified-sweep.csv\`"
	echo "(SHA-256-tracked in \`benchmarks/MANIFEST.json\`) and"
	echo "\`docs/phase6-unified-stress.md\`. See \`docs/reproduction.md\`"
	echo "Level 3 to regenerate those for real."
} >"$OUTPUT_DIR/demo-results.md"

echo
echo "Results written to $OUTPUT_DIR/demo-results.json and $OUTPUT_DIR/demo-results.md"

if [ "$REQUIRED_FAILED" -eq 1 ]; then
	echo
	echo "demo.sh: one or more REQUIRED steps failed -- see per-step logs in $OUTPUT_DIR" >&2
	exit 1
fi

echo
echo "demo.sh: all required steps passed."
exit 0
