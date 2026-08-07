#!/usr/bin/env bash
#
# Production verification + performance regression gate for the Q4_0
# radix-4 divider integration (rtl/membrane_fp_scale_neg_pow2.sv,
# rtl/membrane_fp_divider_radix4.sv, rtl/q4_scale.sv,
# rtl/membrane_quant_stream_top.sv). Promoted from EXP-FPGA-DIV-001; the
# full research record (including promotion-audit.md/promotion-plan.md,
# what was and wasn't brought into production) now lives at
# https://github.com/kadireren7/membrane-research/tree/main/experiments/EXP-FPGA-DIV-001
# -- not duplicated in this repository, see docs/repository-boundary.md.
#
# Usage: scripts/verify-q4-radix4-divider.sh [--quick|--full] [--skip-build]
#            [--resume] [--output-dir <dir>]
#
#   --quick   (default) CI-sized: bounded differential case counts, the
#             full 520,000-transaction production datapath test (cheap
#             enough to always run in full, see this script's own
#             comment at its invocation), synthesis smoke (elaboration
#             only, no synth_ecp5).
#   --full    Nightly/research-sized: reproduces EXP-FPGA-DIV-001 Phase
#             B1/B4's own full differential scope (2,204,128 / 4,456,685
#             cases respectively, same fixed seeds -- see
#             rtl/tb/tb_membrane_fp_scale_neg_pow2.cpp/
#             tb_membrane_fp_divider_radix4.cpp for the seed values) plus
#             the full generic+ECP5 synthesis matrix with performance-gate
#             threshold checks. This is the "explicit research mode" for
#             the 4.4M-case scope task item 7 asks to keep out of ordinary
#             CI -- run it locally or in a dedicated nightly job, not on
#             every push.
#
# Threshold policy (task item 8): thresholds are derived from this
# project's own synthesis-tool-measured EXP-FPGA-DIV-001 Phase B4 results
# (b4-synthesis.csv, b4-differential.json -- see the membrane-research URL
# above) -- Yosys/ABC-synthesized ECP5 technology-mapped cell counts for
# the standalone divider and q4_scale integration, not vendor LUT counts,
# reproduced fresh on this branch (not copied blindly) with a generous
# tolerance -- not brittle exact-cell equality, since yosys is not
# perfectly deterministic run-to-run/version-to-version (ABC's own
# internal ordering can shift cell counts by a handful of cells even for
# byte-identical RTL -- observed directly while preparing this script,
# see that same research record's promotion-comparison.md). No real
# hardware Fmax claim is made or gated on here -- there is no
# place-and-route tool in this environment (unchanged disclosure from
# every EXP-FPGA-DIV-001 phase).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="quick"
SKIP_BUILD=0
RESUME=0
BUILD_DIR="$REPO_ROOT/build-q4-radix4-divider"

while [ $# -gt 0 ]; do
	case "$1" in
	--quick) MODE="quick" ;;
	--full) MODE="full" ;;
	--skip-build) SKIP_BUILD=1 ;;
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
	[ "$SKIP_BUILD" -eq 1 ] && return 1
	if [ "$RESUME" -eq 1 ] && [ -e "$artifact" ]; then
		stage "resume: '$artifact' already exists, skipping rebuild"
		return 1
	fi
	return 0
}

VINC="$REPO_ROOT/tools/.local-verilator/usr/share/verilator/include"
FAILS=0

# =============================================================================
# 1. Focused B1 differential test: membrane_fp_scale_neg_pow2 vs.
#    membrane_fp_divider.
# =============================================================================
stage "build: B1 differential test (membrane_fp_scale_neg_pow2)"
B1_BASE_OBJ="$BUILD_DIR/b1-baseline-obj"
B1_CAND_OBJ="$BUILD_DIR/b1-candidate-obj"
B1_BIN="$BUILD_DIR/tb_membrane_fp_scale_neg_pow2"
if need_build "$B1_BIN"; then
	rm -rf "$B1_BASE_OBJ" "$B1_CAND_OBJ"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$B1_BASE_OBJ" --top-module membrane_fp_divider \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/membrane_fp_divider.sv"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$B1_CAND_OBJ" --top-module membrane_fp_scale_neg_pow2 \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv"
	make -C "$B1_BASE_OBJ" -f Vmembrane_fp_divider.mk Vmembrane_fp_divider__ALL.a -j2
	make -C "$B1_CAND_OBJ" -f Vmembrane_fp_scale_neg_pow2.mk Vmembrane_fp_scale_neg_pow2__ALL.a -j2
	g++ -O2 -std=c++17 -I "$VINC" -I "$VINC/vltstd" -I "$B1_BASE_OBJ" -I "$B1_CAND_OBJ" \
		"$REPO_ROOT/rtl/tb/tb_membrane_fp_scale_neg_pow2.cpp" \
		"$VINC/verilated.cpp" "$VINC/verilated_threads.cpp" \
		"$B1_BASE_OBJ/Vmembrane_fp_divider__ALL.a" "$B1_CAND_OBJ/Vmembrane_fp_scale_neg_pow2__ALL.a" \
		-pthread -lpthread -latomic -o "$B1_BIN"
fi

B1_COUNT=20000
[ "$MODE" = "full" ] && B1_COUNT=2200000
stage "B1 differential test ($MODE: $B1_COUNT random cases + boundary sweep + specials)"
"$B1_BIN" "$B1_COUNT" | tee "$BUILD_DIR/b1-differential.log"
grep -q "^PASS:" "$BUILD_DIR/b1-differential.log" || { echo "FAIL: B1 differential test"; FAILS=$((FAILS + 1)); }

# =============================================================================
# 2. Focused B4 differential test: membrane_fp_divider_radix4 vs.
#    membrane_fp_divider.
# =============================================================================
stage "build: B4 differential test (membrane_fp_divider_radix4)"
B4_CAND_OBJ="$BUILD_DIR/b4-candidate-obj"
B4_BIN="$BUILD_DIR/tb_membrane_fp_divider_radix4"
if need_build "$B4_BIN"; then
	rm -rf "$B4_CAND_OBJ"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$B4_CAND_OBJ" --top-module membrane_fp_divider_radix4 \
		"$REPO_ROOT/rtl/membrane_fp_divider_radix4.sv"
	make -C "$B4_CAND_OBJ" -f Vmembrane_fp_divider_radix4.mk Vmembrane_fp_divider_radix4__ALL.a -j2
	g++ -O2 -std=c++17 -I "$VINC" -I "$VINC/vltstd" -I "$B1_BASE_OBJ" -I "$B4_CAND_OBJ" -I "$REPO_ROOT/include" \
		"$REPO_ROOT/rtl/tb/tb_membrane_fp_divider_radix4.cpp" "$REPO_ROOT/src/codecs/f16convert.c" \
		"$VINC/verilated.cpp" "$VINC/verilated_threads.cpp" \
		"$B1_BASE_OBJ/Vmembrane_fp_divider__ALL.a" "$B4_CAND_OBJ/Vmembrane_fp_divider_radix4__ALL.a" \
		-pthread -lpthread -latomic -o "$B4_BIN"
fi

if [ "$MODE" = "full" ]; then
	B4_RANDOM=4200000; B4_GENERAL=200000; B4_Q4DIST=50000
else
	B4_RANDOM=20000; B4_GENERAL=2000; B4_Q4DIST=500
fi
stage "B4 differential test ($MODE: random=$B4_RANDOM general=$B4_GENERAL q4dist=$B4_Q4DIST)"
"$B4_BIN" "$B4_RANDOM" "$B4_GENERAL" "$B4_Q4DIST" | tee "$BUILD_DIR/b4-differential.log"
grep -q "^PASS:" "$BUILD_DIR/b4-differential.log" || { echo "FAIL: B4 differential test"; FAILS=$((FAILS + 1)); }

# Performance gate 1: divider latency. Derived from this project's own
# measured EXP-FPGA-DIV-001 Phase B4 result (general-path mean ~15 cycles,
# max 15 with no backpressure) with a generous tolerance -- catches a
# gross regression (e.g. an accidental fall-back to one-bit-per-cycle
# iteration, ~29 cycles) without being brittle to +/-1 cycle noise.
B4_MEAN_NOBP=$(grep "^latency cycles:" "$BUILD_DIR/b4-differential.log" | sed -n 's/.*no-bp: min=[0-9]*\ mean=\([0-9.]*\).*/\1/p')
if [ -n "$B4_MEAN_NOBP" ]; then
	B4_LATENCY_OK=$(awk -v m="$B4_MEAN_NOBP" 'BEGIN { print (m >= 10.0 && m <= 20.0) ? 1 : 0 }')
	if [ "$B4_LATENCY_OK" != "1" ]; then
		echo "FAIL: B4 no-backpressure mean latency $B4_MEAN_NOBP cycles outside the expected [10,20] range (measured EXP-FPGA-DIV-001 Phase B4: ~14.9-15.2)"
		FAILS=$((FAILS + 1))
	else
		echo "PASS: B4 no-backpressure mean latency $B4_MEAN_NOBP cycles within [10,20]"
	fi
fi

# =============================================================================
# 3. Production full-datapath test (existing rtl/tb/tb_top_verilator.cpp,
#    unchanged, run against the new production RTL).
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
if need_build "/tmp/top_x_120k.txt"; then
	"$VECDIR/gen_top_x" 120000 /tmp/top_x_120k.txt
	"$VECDIR/gen_pack" /tmp/top_x_120k.txt /tmp/top_q8pack_120k.txt
	"$VECDIR/gen_dequant" /tmp/top_q8pack_120k.txt /tmp/top_q8dequant_120k.txt
	"$VECDIR/gen_q4pack" /tmp/top_x_120k.txt /tmp/top_q4pack_120k.txt
	"$VECDIR/gen_q4unpack" /tmp/top_q4pack_120k.txt /tmp/top_q4unpack_120k.txt
fi

# This is the SAME 520,000-transaction test regardless of --quick/--full --
# it is cheap (~13s on this project's own 5.6GiB dev machine) even with
# Q4_0 encode's now-variable, much longer latency, so there is no reason
# to shrink it for CI -- unlike the two component differential tests
# above, whose --full scope (4.4M+ cases) is genuinely nightly-sized.
stage "production full-datapath test: 520,000 transactions"
"$TOP_OBJ/Vtop_production" | tee "$BUILD_DIR/top-datapath.log"
grep -q "^PASS:" "$BUILD_DIR/top-datapath.log" || { echo "FAIL: production full-datapath test"; FAILS=$((FAILS + 1)); }

# =============================================================================
# 4. Synthesis + area/performance gate.
# =============================================================================
if [ "$MODE" = "full" ]; then
	stage "synthesis matrix (generic + ECP5, standalone radix-4 divider + q4_scale integration)"

	cat >"$BUILD_DIR/synth/standalone-generic.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv
hierarchy -check -top membrane_fp_divider_radix4
proc; opt
synth -top membrane_fp_divider_radix4
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/standalone-generic.ys" | tee "$BUILD_DIR/synth/standalone-generic.log"

	cat >"$BUILD_DIR/synth/standalone-ecp5.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv
hierarchy -check -top membrane_fp_divider_radix4
synth_ecp5 -top membrane_fp_divider_radix4
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/standalone-ecp5.ys" | tee "$BUILD_DIR/synth/standalone-ecp5.log"

	cat >"$BUILD_DIR/synth/q4scale-ecp5.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv $REPO_ROOT/rtl/q4_scale.sv
hierarchy -check -top q4_scale
synth_ecp5 -top q4_scale
stat
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/q4scale-ecp5.ys" | tee "$BUILD_DIR/synth/q4scale-ecp5.log"

	# Performance gate 2: ECP5 cell count, both at the standalone-unit and
	# q4_scale-integration granularity. Range derived from this project's
	# own measured EXP-FPGA-DIV-001 Phase B4 numbers
	# (results/b4-synthesis.csv: standalone 1509, q4_scale 2836),
	# reproduced fresh on this branch (1509, 2833-2836 across repeated
	# runs -- a few cells of run-to-run ABC noise, not a real change) with
	# a wide +/-30% tolerance: enough to catch a real regression (e.g. the
	# radix-4 iteration accidentally duplicated, or a fallback to the
	# ~73,629-cell baseline combinational divider) without failing on
	# ordinary tool-version/ABC-ordering drift.
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
	check_cell_range "standalone membrane_fp_divider_radix4" "$BUILD_DIR/synth/standalone-ecp5.log" 1000 2000
	check_cell_range "q4_scale integration" "$BUILD_DIR/synth/q4scale-ecp5.log" 2000 3700
else
	stage "synthesis smoke test (elaboration only, no full synth_ecp5 -- see --full)"
	cat >"$BUILD_DIR/synth/elab-radix4.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv
hierarchy -check -top membrane_fp_divider_radix4
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/elab-radix4.ys"
	cat >"$BUILD_DIR/synth/elab-top.ys" <<EOF
read_verilog -sv $REPO_ROOT/rtl/membrane_fp_pkg.sv $REPO_ROOT/rtl/membrane_fp_adder.sv $REPO_ROOT/rtl/membrane_fp_divider.sv $REPO_ROOT/rtl/membrane_fp_multiplier.sv $REPO_ROOT/rtl/stream_fifo.sv $REPO_ROOT/rtl/valid_delay_line.sv $REPO_ROOT/rtl/q4_pack.sv $REPO_ROOT/rtl/q4_scale.sv $REPO_ROOT/rtl/q4_scan.sv $REPO_ROOT/rtl/q4_unpack.sv $REPO_ROOT/rtl/q8_dequantize.sv $REPO_ROOT/rtl/q8_maxabs_reduce.sv $REPO_ROOT/rtl/q8_quantize_pack.sv $REPO_ROOT/rtl/q8_scale.sv $REPO_ROOT/rtl/membrane_fp_scale_neg_pow2.sv $REPO_ROOT/rtl/membrane_fp_divider_radix4.sv $REPO_ROOT/rtl/membrane_quant_stream_top.sv
hierarchy -check -top membrane_quant_stream_top
EOF
	"$YOSYS" -s "$BUILD_DIR/synth/elab-top.ys"
fi

echo
if [ "$FAILS" -eq 0 ]; then
	echo "=== ALL CHECKS PASSED ($MODE mode) ==="
	exit 0
else
	echo "=== $FAILS CHECK(S) FAILED ($MODE mode) ==="
	exit 1
fi
