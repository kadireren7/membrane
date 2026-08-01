#!/usr/bin/env bash
#
# Reproduction script for EXP-FPGA-DIV-002 (Q8_0 divider-pair baseline +
# feasibility study, experiments/EXP-FPGA-DIV-002/). Phase A only: no
# production RTL is modified or synthesized as an alternative here -- this
# reproduces the CHARACTERIZATION and DIFFERENTIAL FEASIBILITY measurements
# (experiments/EXP-FPGA-DIV-002/baseline.md), same "no new divider variant
# written" scope as the experiment itself. Modeled on
# scripts/verify-q4-radix4-divider.sh's own structure/conventions
# (EXP-FPGA-DIV-001), not a copy -- this experiment measures a candidate
# space rather than gating a promoted production integration.
#
# Usage: scripts/run-exp-q8-divider-002.sh [--quick|--full] [--resume]
#            [--output-dir <dir>]
#
#   --quick   (default) CI-sized: a focused feasibility subset (~100K
#             cases, seconds), a Q8-focused smoke slice of the full
#             520,000-transaction datapath test (still runs the whole
#             thing -- it is cheap, ~13-30s, no reason to shrink it, same
#             reasoning verify-q4-radix4-divider.sh's own comment gives),
#             and a synthesis smoke test (elaboration only, no
#             synth_ecp5).
#   --full    Research-sized: reproduces this experiment's own full scope
#             (2,000,000+ random cases + 50,000 runtime-distribution
#             cases for the feasibility differential) plus the full
#             generic+ECP5 synthesis matrix (standalone
#             membrane_fp_divider, q8_scale, and a best-effort,
#             time-bounded full-top-level attempt -- see this script's
#             own TOP_SYNTH_TIMEOUT_S, matching baseline.md section 4's
#             own disclosed UNAVAILABLE result for that last one).
#
# No production RTL file is modified by this script. No new divider
# variant is written, synthesized, or promoted -- this only re-drives the
# existing rtl/membrane_fp_divider.sv / rtl/membrane_fp_multiplier.sv RTL
# (via rtl/tb/tb_q8_scale_feasibility.cpp) and re-synthesizes the existing
# rtl/membrane_fp_divider.sv / rtl/q8_scale.sv / rtl/membrane_quant_stream_top.sv
# RTL, unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="quick"
RESUME=0
BUILD_DIR="$REPO_ROOT/build-exp-q8-divider-002"

while [ $# -gt 0 ]; do
	case "$1" in
	--quick) MODE="quick" ;;
	--full) MODE="full" ;;
	--resume) RESUME=1 ;;
	--output-dir)
		shift
		BUILD_DIR="$1"
		;;
	-h | --help)
		sed -n '2,40p' "$0"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 1
		;;
	esac
	shift
done

YOSYS="$REPO_ROOT/tools/.local-yosys/usr/bin/yosys"
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
case "$VERILATOR_BIN" in
"$REPO_ROOT/tools/.local-verilator/"*)
	[ -d "$LOCAL_VERILATOR_ROOT" ] && export VERILATOR_ROOT="$LOCAL_VERILATOR_ROOT"
	;;
esac
if [ -z "$YOSYS" ] || [ ! -x "$YOSYS" ]; then
	echo "error: yosys not found at $YOSYS" >&2
	exit 1
fi
if [ -z "$VERILATOR_BIN" ]; then
	echo "error: verilator not found in PATH or tools/.local-verilator/usr/bin" >&2
	exit 1
fi

mkdir -p "$BUILD_DIR" "$BUILD_DIR/synth" "$BUILD_DIR/vectors"

stage() { echo; echo "== $(date '+%H:%M:%S') [stage] $* =="; }

need_build() {
	local artifact="$1"
	if [ "$RESUME" -eq 1 ] && [ -e "$artifact" ]; then
		stage "resume: '$artifact' already exists, skipping rebuild"
		return 1
	fi
	return 0
}

VINC="$REPO_ROOT/tools/.local-verilator/usr/share/verilator/include"
FAILS=0

# =============================================================================
# 1. Differential feasibility tool (rtl/tb/tb_q8_scale_feasibility.cpp):
#    re-drives the real membrane_fp_divider/membrane_fp_multiplier RTL to
#    measure candidate B (reciprocal reconstruction) and C
#    (constant-reciprocal multiply) against the production d/id.
# =============================================================================
stage "build: q8_scale divider-pair feasibility differential"
DIV_OBJ="$BUILD_DIR/div-obj"
MUL_OBJ="$BUILD_DIR/mul-obj"
FEAS_BIN="$BUILD_DIR/tb_q8_scale_feasibility"
if need_build "$FEAS_BIN"; then
	rm -rf "$DIV_OBJ" "$MUL_OBJ"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$DIV_OBJ" --top-module membrane_fp_divider \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/membrane_fp_divider.sv"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$MUL_OBJ" --top-module membrane_fp_multiplier \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/membrane_fp_multiplier.sv"
	make -C "$DIV_OBJ" -f Vmembrane_fp_divider.mk Vmembrane_fp_divider__ALL.a -j2
	make -C "$MUL_OBJ" -f Vmembrane_fp_multiplier.mk Vmembrane_fp_multiplier__ALL.a -j2
	g++ -O2 -std=c++17 -I "$VINC" -I "$VINC/vltstd" -I "$DIV_OBJ" -I "$MUL_OBJ" -I "$REPO_ROOT/include" \
		"$REPO_ROOT/rtl/tb/tb_q8_scale_feasibility.cpp" "$REPO_ROOT/src/codecs/f16convert.c" \
		"$VINC/verilated.cpp" "$VINC/verilated_threads.cpp" \
		"$DIV_OBJ/Vmembrane_fp_divider__ALL.a" "$MUL_OBJ/Vmembrane_fp_multiplier__ALL.a" \
		-pthread -lpthread -latomic -o "$FEAS_BIN"
fi

if [ "$MODE" = "full" ]; then
	FEAS_RANDOM=2000000; FEAS_RUNTIME=50000
else
	FEAS_RANDOM=100000; FEAS_RUNTIME=2000
fi
stage "differential feasibility ($MODE: random=$FEAS_RANDOM runtime_dist=$FEAS_RUNTIME + edge cases). Progress heartbeats every 5s on long runs."
"$FEAS_BIN" "$FEAS_RANDOM" "$FEAS_RUNTIME" | tee "$BUILD_DIR/feasibility-differential.log"
grep -q "^DONE:" "$BUILD_DIR/feasibility-differential.log" || { echo "FAIL: feasibility differential did not complete its planned case count"; FAILS=$((FAILS + 1)); }

# =============================================================================
# 2. Production full-datapath test (existing rtl/tb/tb_top_verilator.cpp,
#    unchanged RTL) -- includes the focused Q8 encode (120,000) and Q8
#    decode (120,000) stages this experiment treats as its own "focused
#    Q8 encode/decode" test, per its own baseline.md section 3.
# =============================================================================
stage "build: production full-datapath test (membrane_quant_stream_top)"
TOP_OBJ="$BUILD_DIR/top-obj"
if need_build "$TOP_OBJ/Vtop_production"; then
	rm -rf "$TOP_OBJ"
	"$VERILATOR_BIN" --cc --exe --build -j 2 -Wno-fatal --Mdir "$TOP_OBJ" \
		--top-module membrane_quant_stream_top \
		"$REPO_ROOT/rtl/membrane_fp_pkg.sv" "$REPO_ROOT/rtl/membrane_fp_adder.sv" \
		"$REPO_ROOT/rtl/membrane_fp_divider.sv" "$REPO_ROOT/rtl/membrane_fp_multiplier.sv" \
		"$REPO_ROOT/rtl/stream_fifo.sv" "$REPO_ROOT/rtl/valid_delay_line.sv" \
		"$REPO_ROOT/rtl/q4_pack.sv" "$REPO_ROOT/rtl/q4_scale.sv" "$REPO_ROOT/rtl/q4_scan.sv" \
		"$REPO_ROOT/rtl/q4_unpack.sv" "$REPO_ROOT/rtl/q8_dequantize.sv" \
		"$REPO_ROOT/rtl/q8_maxabs_reduce.sv" "$REPO_ROOT/rtl/q8_quantize_pack.sv" \
		"$REPO_ROOT/rtl/q8_scale.sv" "$REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv" \
		"$REPO_ROOT/rtl/membrane_fp_divider_radix4.sv" "$REPO_ROOT/rtl/membrane_quant_stream_top.sv" \
		"$REPO_ROOT/rtl/tb/tb_top_verilator.cpp" -o Vtop_production
fi

VECDIR="$BUILD_DIR/vectors"
if need_build "$VECDIR/gen_top_x"; then
	cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_top_x" "$REPO_ROOT/rtl/tb/gen_top_x_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_pack" "$REPO_ROOT/rtl/tb/gen_pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_dequant" "$REPO_ROOT/rtl/tb/gen_dequant_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_q4pack" "$REPO_ROOT/rtl/tb/gen_q4pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$VECDIR/gen_q4unpack" "$REPO_ROOT/rtl/tb/gen_q4unpack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
fi
if need_build "$BUILD_DIR/vectors/top_x_120k.txt"; then
	"$VECDIR/gen_top_x" 120000 "$BUILD_DIR/vectors/top_x_120k.txt"
	"$VECDIR/gen_pack" "$BUILD_DIR/vectors/top_x_120k.txt" "$BUILD_DIR/vectors/top_q8pack_120k.txt"
	"$VECDIR/gen_dequant" "$BUILD_DIR/vectors/top_q8pack_120k.txt" "$BUILD_DIR/vectors/top_q8dequant_120k.txt"
	"$VECDIR/gen_q4pack" "$BUILD_DIR/vectors/top_x_120k.txt" "$BUILD_DIR/vectors/top_q4pack_120k.txt"
	"$VECDIR/gen_q4unpack" "$BUILD_DIR/vectors/top_q4pack_120k.txt" "$BUILD_DIR/vectors/top_q4unpack_120k.txt"
fi

# tb_top_verilator.cpp reads its golden vectors from fixed /tmp paths
# (its own long-standing convention, unchanged here) -- point it there.
cp "$BUILD_DIR/vectors/top_x_120k.txt" /tmp/top_x_120k.txt
cp "$BUILD_DIR/vectors/top_q8pack_120k.txt" /tmp/top_q8pack_120k.txt
cp "$BUILD_DIR/vectors/top_q8dequant_120k.txt" /tmp/top_q8dequant_120k.txt
cp "$BUILD_DIR/vectors/top_q4pack_120k.txt" /tmp/top_q4pack_120k.txt
cp "$BUILD_DIR/vectors/top_q4unpack_120k.txt" /tmp/top_q4unpack_120k.txt

stage "production full-datapath test: 520,000 transactions (incl. focused Q8 encode/decode, 120,000 each)"
"$TOP_OBJ/Vtop_production" | tee "$BUILD_DIR/top-datapath.log"
grep -q "^PASS:" "$BUILD_DIR/top-datapath.log" || { echo "FAIL: production full-datapath test"; FAILS=$((FAILS + 1)); }

# =============================================================================
# 3. Synthesis matrix.
# =============================================================================
# Full-top synth_ecp5 is a best-effort, time-bounded attempt -- this
# experiment's own baseline.md section 4 already disclosed it as
# UNAVAILABLE (killed after 25 minutes on this project's own
# memory-constrained dev machine). A timeout, not a hard failure: an
# UNAVAILABLE result here is expected and does not fail this script.
TOP_SYNTH_TIMEOUT_S=1500

if [ "$MODE" = "full" ]; then
	stage "synthesis matrix (generic + ECP5): standalone membrane_fp_divider, q8_scale, best-effort full top"

	cat >"$BUILD_DIR/synth/standalone-generic.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv
hierarchy -check -top membrane_fp_divider
proc; opt
synth -top membrane_fp_divider
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/standalone-generic.ys" | tee "$BUILD_DIR/synth/standalone-generic.log"

	cat >"$BUILD_DIR/synth/standalone-ecp5.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv
hierarchy -check -top membrane_fp_divider
synth_ecp5 -top membrane_fp_divider
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/standalone-ecp5.ys" | tee "$BUILD_DIR/synth/standalone-ecp5.log"

	cat >"$BUILD_DIR/synth/q8scale-generic.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/q8_scale.sv
hierarchy -check -top q8_scale
proc; opt
synth -top q8_scale
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/q8scale-generic.ys" | tee "$BUILD_DIR/synth/q8scale-generic.log"

	cat >"$BUILD_DIR/synth/q8scale-ecp5.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/q8_scale.sv
hierarchy -check -top q8_scale
synth_ecp5 -top q8_scale
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/q8scale-ecp5.ys" | tee "$BUILD_DIR/synth/q8scale-ecp5.log"

	# Performance/regression gate: standalone divider and q8_scale ECP5
	# cell counts. Range derived from this experiment's OWN measured
	# baseline (experiments/EXP-FPGA-DIV-002/results/synthesis.csv:
	# standalone 73629, q8_scale 123742), reproduced fresh on this branch,
	# with a wide tolerance (yosys/ABC ordering is not perfectly
	# deterministic run-to-run -- same caveat verify-q4-radix4-divider.sh
	# already documents for its own gates).
	check_cell_range() {
		local label="$1" log="$2" lo="$3" hi="$4"
		local n
		n=$(grep -A1 "Number of cells:" "$log" | head -1 | awk '{print $NF}')
		if [ -z "$n" ]; then
			echo "FAIL: could not read cell count for $label from $log"
			FAILS=$((FAILS + 1))
			return
		fi
		if [ "$n" -lt "$lo" ] || [ "$n" -gt "$hi" ]; then
			echo "FAIL: $label ECP5 cells=$n outside expected [$lo,$hi]"
			FAILS=$((FAILS + 1))
		else
			echo "PASS: $label ECP5 cells=$n within [$lo,$hi]"
		fi
	}
	check_cell_range "standalone membrane_fp_divider" "$BUILD_DIR/synth/standalone-ecp5.log" 65000 82000
	check_cell_range "q8_scale" "$BUILD_DIR/synth/q8scale-ecp5.log" 110000 137000

	stage "full top-level synth_ecp5: best-effort, bounded at ${TOP_SYNTH_TIMEOUT_S}s (UNAVAILABLE-on-timeout is expected, see baseline.md section 4)"
	cat >"$BUILD_DIR/synth/top-ecp5.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/membrane_fp_adder.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/membrane_fp_multiplier.sv $REPO_ROOT/rtl/stream_fifo.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/q4_pack.sv $REPO_ROOT/rtl/q4_scale.sv $REPO_ROOT/rtl/q4_scan.sv $REPO_ROOT/rtl/q4_unpack.sv $REPO_ROOT/rtl/q8_dequantize.sv $REPO_ROOT/rtl/q8_maxabs_reduce.sv $REPO_ROOT/rtl/q8_quantize_pack.sv $REPO_ROOT/rtl/q8_scale.sv $REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv $REPO_ROOT/rtl/membrane_quant_stream_top.sv
hierarchy -check -top membrane_quant_stream_top
synth_ecp5 -top membrane_quant_stream_top
stat
EOF
	if timeout "$TOP_SYNTH_TIMEOUT_S" "$YOSYS" -s "$BUILD_DIR/synth/top-ecp5.ys" >"$BUILD_DIR/synth/top-ecp5.log" 2>&1; then
		echo "top-level synth_ecp5 completed (better than this experiment's own baseline run -- see the fresh log for real numbers)"
	else
		rc=$?
		if [ "$rc" -eq 124 ]; then
			echo "UNAVAILABLE: top-level synth_ecp5 timed out after ${TOP_SYNTH_TIMEOUT_S}s (expected, matches baseline.md section 4 -- not a script failure)"
		else
			echo "FAIL: top-level synth_ecp5 exited with unexpected error code $rc"
			FAILS=$((FAILS + 1))
		fi
	fi
else
	stage "synthesis smoke test (elaboration only, no full synth_ecp5 -- see --full)"
	cat >"$BUILD_DIR/synth/elab-divider.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv
hierarchy -check -top membrane_fp_divider
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/elab-divider.ys"
	cat >"$BUILD_DIR/synth/elab-q8scale.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/q8_scale.sv
hierarchy -check -top q8_scale
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/elab-q8scale.ys"
	cat >"$BUILD_DIR/synth/elab-top.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/membrane_fp_adder.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/membrane_fp_multiplier.sv $REPO_ROOT/rtl/stream_fifo.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/q4_pack.sv $REPO_ROOT/rtl/q4_scale.sv $REPO_ROOT/rtl/q4_scan.sv $REPO_ROOT/rtl/q4_unpack.sv $REPO_ROOT/rtl/q8_dequantize.sv $REPO_ROOT/rtl/q8_maxabs_reduce.sv $REPO_ROOT/rtl/q8_quantize_pack.sv $REPO_ROOT/rtl/q8_scale.sv $REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv $REPO_ROOT/rtl/membrane_quant_stream_top.sv
hierarchy -check -top membrane_quant_stream_top
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/elab-top.ys"
fi

# =============================================================================
# 4. Local CI-equivalent verification (--full only -- --quick assumes the
#    caller already has these covered by normal CI; matches this
#    experiment's own resource-budget scoping).
# =============================================================================
if [ "$MODE" = "full" ]; then
	stage "local verification: Debug/Release/ASan+UBSan/TSan ctest, ggml quant parity, verify-*.py"
	for cfg in build-debug build build-asan build-tsan; do
		if [ -d "$REPO_ROOT/$cfg" ]; then
			cmake --build "$REPO_ROOT/$cfg" -j "$(nproc)"
			if [ "$cfg" = "build-tsan" ]; then
				setarch "$(uname -m)" -R ctest --test-dir "$REPO_ROOT/$cfg" --output-on-failure || FAILS=$((FAILS + 1))
			else
				ctest --test-dir "$REPO_ROOT/$cfg" --output-on-failure || FAILS=$((FAILS + 1))
			fi
		else
			echo "note: $cfg not configured, skipping (run cmake -S . -B $cfg first for full coverage)"
		fi
	done
	python3 "$REPO_ROOT/scripts/verify-results.py" || FAILS=$((FAILS + 1))
	python3 "$REPO_ROOT/scripts/verify-outreach.py" || FAILS=$((FAILS + 1))
	python3 "$REPO_ROOT/paper/scripts/verify-paper.py" || FAILS=$((FAILS + 1))
fi

echo
if [ "$FAILS" -eq 0 ]; then
	echo "=== ALL CHECKS PASSED ($MODE mode) ==="
	exit 0
else
	echo "=== $FAILS CHECK(S) FAILED ($MODE mode) ==="
	exit 1
fi
