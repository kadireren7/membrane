#!/usr/bin/env bash
#
# EXP-FPGA-DIV-001 reproduction script.
#
# Usage: scripts/run-exp-fp-divider-001.sh --phase <phase> [--quick|--full] [--skip-build]
#
#   --phase b1     Phase B1: fp32_scale_neg_pow2 vs. membrane_fp_divider
#                  for the Q4 mx/-8.0f operation (only phase implemented
#                  so far -- see experiments/EXP-FPGA-DIV-001/).
#   --quick        (default) a fast smoke pass: a small differential
#                  run, a small full-datapath integration run (same
#                  binary, same checks, fewer transactions -- see
#                  rtl/experimental/fp_div/tb_top_verilator_variant.cpp's
#                  own header comment for why this is "the same
#                  testbench", not a separately written lighter test),
#                  and a synthesis ELABORATION-ONLY smoke check
#                  (hierarchy -check, no full synth_ecp5 -- that takes
#                  minutes per module, not appropriate for --quick).
#   --full         the exact scope experiments/EXP-FPGA-DIV-001/phase-b1.md
#                  reports: >=2,000,000 differential cases, the full
#                  520,000-transaction datapath test for BOTH the
#                  baseline and B1 variants, and the complete
#                  generic+ECP5 synthesis matrix (4 runs: standalone
#                  unit x2 variants, q4_scale-level x2 variants).
#   --skip-build   assume binaries already built under $BUILD_DIR; only
#                  re-run them.
#
# No step is ever skipped or turned into a soft failure -- any real
# test failure here is a script failure (`set -e`), not a warning.
# Long-running steps (the full differential test, the full synthesis
# matrix) print their own progress: the differential/datapath binaries
# have a built-in 60-second heartbeat, and this script prints a
# start/elapsed/end banner around every yosys invocation, since yosys
# itself does not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PHASE=""
MODE="quick"
SKIP_BUILD=0
BUILD_DIR="$REPO_ROOT/build-exp-fpga-div-001"

while [ $# -gt 0 ]; do
	case "$1" in
	--phase)
		shift
		PHASE="$1"
		;;
	--quick) MODE="quick" ;;
	--full) MODE="full" ;;
	--skip-build) SKIP_BUILD=1 ;;
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

if [ "$PHASE" != "b1" ]; then
	echo "error: --phase b1 is the only implemented phase (got: '${PHASE}')" >&2
	exit 1
fi

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

mkdir -p "$BUILD_DIR"

stage()
{
	echo
	echo "== $(date '+%H:%M:%S') [stage] $* =="
}

yosys_run()
{
	local label="$1"
	local script="$2"
	local t0 t1

	stage "yosys: $label"
	t0=$(date +%s)
	"$YOSYS" -s "$script"
	t1=$(date +%s)
	echo "[stage] yosys: $label done in $((t1 - t0))s"
}

# ---------------------------------------------------------------------
# Build: differential test (fp32_scale_neg_pow2 vs. membrane_fp_divider)
# ---------------------------------------------------------------------
DIFF_BASELINE_OBJ="$BUILD_DIR/diff-baseline-obj"
DIFF_CANDIDATE_OBJ="$BUILD_DIR/diff-candidate-obj"
DIFF_BIN="$BUILD_DIR/tb_fp32_scale_neg_pow2"

build_differential()
{
	stage "build: differential test (membrane_fp_divider vs. fp32_scale_neg_pow2)"
	rm -rf "$DIFF_BASELINE_OBJ" "$DIFF_CANDIDATE_OBJ"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$DIFF_BASELINE_OBJ" \
		--top-module membrane_fp_divider \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/membrane_fp_divider.sv"
	"$VERILATOR_BIN" --cc -Wno-fatal --Mdir "$DIFF_CANDIDATE_OBJ" \
		--top-module fp32_scale_neg_pow2 \
		"$REPO_ROOT/rtl/valid_delay_line.sv" "$REPO_ROOT/rtl/experimental/fp_div/fp32_scale_neg_pow2.sv"
	make -C "$DIFF_BASELINE_OBJ" -f Vmembrane_fp_divider.mk Vmembrane_fp_divider__ALL.a -j2
	make -C "$DIFF_CANDIDATE_OBJ" -f Vfp32_scale_neg_pow2.mk Vfp32_scale_neg_pow2__ALL.a -j2

	local vinc="$REPO_ROOT/tools/.local-verilator/usr/share/verilator/include"
	g++ -O2 -std=c++17 \
		-I "$vinc" -I "$vinc/vltstd" \
		-I "$DIFF_BASELINE_OBJ" -I "$DIFF_CANDIDATE_OBJ" \
		"$REPO_ROOT/rtl/tb/tb_fp32_scale_neg_pow2.cpp" \
		"$vinc/verilated.cpp" "$vinc/verilated_threads.cpp" \
		"$DIFF_BASELINE_OBJ/Vmembrane_fp_divider__ALL.a" \
		"$DIFF_CANDIDATE_OBJ/Vfp32_scale_neg_pow2__ALL.a" \
		-pthread -lpthread -latomic \
		-o "$DIFF_BIN"
}

# ---------------------------------------------------------------------
# Build: full-datapath variant test (baseline + B1 top-level)
# ---------------------------------------------------------------------
TOP_BASELINE_OBJ="$BUILD_DIR/top-baseline-obj"
TOP_B1_OBJ="$BUILD_DIR/top-b1-obj"

build_datapath()
{
	stage "build: full-datapath test (baseline membrane_quant_stream_top)"
	rm -rf "$TOP_BASELINE_OBJ"
	"$VERILATOR_BIN" --cc --exe --build -j 2 -Wno-fatal --Mdir "$TOP_BASELINE_OBJ" \
		--top-module membrane_quant_stream_top \
		"$REPO_ROOT/rtl/membrane_fp_pkg.sv" "$REPO_ROOT/rtl/membrane_fp_adder.sv" \
		"$REPO_ROOT/rtl/membrane_fp_divider.sv" "$REPO_ROOT/rtl/membrane_fp_multiplier.sv" \
		"$REPO_ROOT/rtl/stream_fifo.sv" "$REPO_ROOT/rtl/valid_delay_line.sv" \
		"$REPO_ROOT/rtl/q4_pack.sv" "$REPO_ROOT/rtl/q4_scale.sv" "$REPO_ROOT/rtl/q4_scan.sv" \
		"$REPO_ROOT/rtl/q4_unpack.sv" "$REPO_ROOT/rtl/q8_dequantize.sv" \
		"$REPO_ROOT/rtl/q8_maxabs_reduce.sv" "$REPO_ROOT/rtl/q8_quantize_pack.sv" \
		"$REPO_ROOT/rtl/q8_scale.sv" "$REPO_ROOT/rtl/membrane_quant_stream_top.sv" \
		"$REPO_ROOT/rtl/experimental/fp_div/tb_top_verilator_variant.cpp" \
		-o Vtop_baseline

	stage "build: full-datapath test (B1 membrane_quant_stream_top_b1)"
	rm -rf "$TOP_B1_OBJ"
	"$VERILATOR_BIN" --cc --exe --build -j 2 -Wno-fatal --Mdir "$TOP_B1_OBJ" \
		--top-module membrane_quant_stream_top_b1 \
		"$REPO_ROOT/rtl/membrane_fp_pkg.sv" "$REPO_ROOT/rtl/membrane_fp_adder.sv" \
		"$REPO_ROOT/rtl/membrane_fp_divider.sv" "$REPO_ROOT/rtl/membrane_fp_multiplier.sv" \
		"$REPO_ROOT/rtl/stream_fifo.sv" "$REPO_ROOT/rtl/valid_delay_line.sv" \
		"$REPO_ROOT/rtl/q4_pack.sv" "$REPO_ROOT/rtl/q4_scan.sv" \
		"$REPO_ROOT/rtl/q4_unpack.sv" "$REPO_ROOT/rtl/q8_dequantize.sv" \
		"$REPO_ROOT/rtl/q8_maxabs_reduce.sv" "$REPO_ROOT/rtl/q8_quantize_pack.sv" \
		"$REPO_ROOT/rtl/q8_scale.sv" \
		"$REPO_ROOT/rtl/experimental/fp_div/fp32_scale_neg_pow2.sv" \
		"$REPO_ROOT/rtl/experimental/fp_div/q4_scale_b1.sv" \
		"$REPO_ROOT/rtl/experimental/fp_div/membrane_quant_stream_top_b1.sv" \
		"$REPO_ROOT/rtl/experimental/fp_div/tb_top_verilator_variant.cpp" \
		-CFLAGS "-DMEMBRANE_B1_VARIANT" \
		-o Vtop_b1
}

gen_datapath_vectors()
{
	stage "generate golden vectors for full-datapath test"
	local vecdir="$BUILD_DIR/vectors"
	mkdir -p "$vecdir"
	cc -O2 -I "$REPO_ROOT/include" -o "$vecdir/gen_top_x" "$REPO_ROOT/rtl/tb/gen_top_x_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$vecdir/gen_pack" "$REPO_ROOT/rtl/tb/gen_pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$vecdir/gen_dequant" "$REPO_ROOT/rtl/tb/gen_dequant_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$vecdir/gen_q4pack" "$REPO_ROOT/rtl/tb/gen_q4pack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	cc -O2 -I "$REPO_ROOT/include" -o "$vecdir/gen_q4unpack" "$REPO_ROOT/rtl/tb/gen_q4unpack_vectors.c" "$REPO_ROOT/src/codecs/f16convert.c" "$REPO_ROOT/src/quant/quant_simd.c"
	"$vecdir/gen_top_x" 120000 /tmp/top_x_120k.txt
	"$vecdir/gen_pack" /tmp/top_x_120k.txt /tmp/top_q8pack_120k.txt
	"$vecdir/gen_dequant" /tmp/top_q8pack_120k.txt /tmp/top_q8dequant_120k.txt
	"$vecdir/gen_q4pack" /tmp/top_x_120k.txt /tmp/top_q4pack_120k.txt
	"$vecdir/gen_q4unpack" /tmp/top_q4pack_120k.txt /tmp/top_q4unpack_120k.txt
}

if [ "$SKIP_BUILD" -eq 0 ]; then
	build_differential
	build_datapath
fi
if [ ! -f /tmp/top_x_120k.txt ] || [ "$SKIP_BUILD" -eq 0 ]; then
	gen_datapath_vectors
fi

# ---------------------------------------------------------------------
# Differential test
# ---------------------------------------------------------------------
if [ "$MODE" = "quick" ]; then
	DIFF_RANDOM_COUNT=20000
else
	DIFF_RANDOM_COUNT=2200000
fi
stage "differential test ($MODE: $DIFF_RANDOM_COUNT random cases + boundary sweep + specials)"
"$DIFF_BIN" "$DIFF_RANDOM_COUNT"

# ---------------------------------------------------------------------
# Full-datapath integration test (both variants, same testbench binary
# family, transaction count depends on --quick/--full)
# ---------------------------------------------------------------------
if [ "$MODE" = "quick" ]; then
	N_PER_MODE=2000
	N_MIX=500
else
	N_PER_MODE=120000
	N_MIX=40000
fi
stage "full-datapath test, baseline ($MODE: N_PER_MODE=$N_PER_MODE N_MIX=$N_MIX)"
"$TOP_BASELINE_OBJ/Vtop_baseline" "$N_PER_MODE" "$N_MIX"
stage "full-datapath test, B1 variant ($MODE: N_PER_MODE=$N_PER_MODE N_MIX=$N_MIX)"
"$TOP_B1_OBJ/Vtop_b1" "$N_PER_MODE" "$N_MIX"

# ---------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------
SYNTH_DIR="$BUILD_DIR/synth"
mkdir -p "$SYNTH_DIR"

if [ "$MODE" = "quick" ]; then
	stage "synthesis smoke test (elaboration only, no full synth_ecp5 -- see --full)"
	cat >"$SYNTH_DIR/elab-standalone.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/membrane_fp_divider.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv
hierarchy -check -top membrane_fp_divider
EOF
	yosys_run "elaborate membrane_fp_divider" "$SYNTH_DIR/elab-standalone.ys"
	cat >"$SYNTH_DIR/elab-candidate.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv
hierarchy -check -top fp32_scale_neg_pow2
EOF
	yosys_run "elaborate fp32_scale_neg_pow2" "$SYNTH_DIR/elab-candidate.ys"
	cat >"$SYNTH_DIR/elab-q4scale-b1.ys" <<EOF
read_verilog -sv rtl/membrane_fp_pkg.sv rtl/membrane_fp_divider.sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv rtl/experimental/fp_div/q4_scale_b1.sv
hierarchy -check -top q4_scale_b1
EOF
	yosys_run "elaborate q4_scale_b1" "$SYNTH_DIR/elab-q4scale-b1.ys"
else
	stage "full synthesis matrix (generic + ECP5, standalone unit + q4_scale, both variants -- several minutes, real synth_ecp5 memory cost, run sequentially not in parallel)"

	cat >"$SYNTH_DIR/standalone-baseline-generic.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/membrane_fp_divider.sv
hierarchy -check -top membrane_fp_divider
proc; opt
synth -top membrane_fp_divider
stat
EOF
	yosys_run "standalone baseline (membrane_fp_divider), generic" "$SYNTH_DIR/standalone-baseline-generic.ys"

	cat >"$SYNTH_DIR/standalone-baseline-ecp5.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/membrane_fp_divider.sv
hierarchy -check -top membrane_fp_divider
synth_ecp5 -top membrane_fp_divider
stat
EOF
	yosys_run "standalone baseline (membrane_fp_divider), ECP5" "$SYNTH_DIR/standalone-baseline-ecp5.ys"

	cat >"$SYNTH_DIR/standalone-b1-generic.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv
hierarchy -check -top fp32_scale_neg_pow2
proc; opt
synth -top fp32_scale_neg_pow2
stat
EOF
	yosys_run "standalone B1 (fp32_scale_neg_pow2), generic" "$SYNTH_DIR/standalone-b1-generic.ys"

	cat >"$SYNTH_DIR/standalone-b1-ecp5.ys" <<EOF
read_verilog -sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv
hierarchy -check -top fp32_scale_neg_pow2
synth_ecp5 -top fp32_scale_neg_pow2
stat
EOF
	yosys_run "standalone B1 (fp32_scale_neg_pow2), ECP5" "$SYNTH_DIR/standalone-b1-ecp5.ys"

	cat >"$SYNTH_DIR/q4scale-baseline-generic.ys" <<EOF
read_verilog -sv rtl/membrane_fp_pkg.sv rtl/membrane_fp_divider.sv rtl/valid_delay_line.sv rtl/q4_scale.sv
hierarchy -check -top q4_scale
proc; opt
synth -top q4_scale
stat
EOF
	yosys_run "q4_scale baseline, generic" "$SYNTH_DIR/q4scale-baseline-generic.ys"

	cat >"$SYNTH_DIR/q4scale-baseline-ecp5.ys" <<EOF
read_verilog -sv rtl/membrane_fp_pkg.sv rtl/membrane_fp_divider.sv rtl/valid_delay_line.sv rtl/q4_scale.sv
hierarchy -check -top q4_scale
synth_ecp5 -top q4_scale
stat
EOF
	yosys_run "q4_scale baseline, ECP5" "$SYNTH_DIR/q4scale-baseline-ecp5.ys"

	cat >"$SYNTH_DIR/q4scale-b1-generic.ys" <<EOF
read_verilog -sv rtl/membrane_fp_pkg.sv rtl/membrane_fp_divider.sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv rtl/experimental/fp_div/q4_scale_b1.sv
hierarchy -check -top q4_scale_b1
proc; opt
synth -top q4_scale_b1
stat
EOF
	yosys_run "q4_scale_b1, generic" "$SYNTH_DIR/q4scale-b1-generic.ys"

	cat >"$SYNTH_DIR/q4scale-b1-ecp5.ys" <<EOF
read_verilog -sv rtl/membrane_fp_pkg.sv rtl/membrane_fp_divider.sv rtl/valid_delay_line.sv rtl/experimental/fp_div/fp32_scale_neg_pow2.sv rtl/experimental/fp_div/q4_scale_b1.sv
hierarchy -check -top q4_scale_b1
synth_ecp5 -top q4_scale_b1
stat
EOF
	yosys_run "q4_scale_b1, ECP5" "$SYNTH_DIR/q4scale-b1-ecp5.ys"
fi

stage "done ($MODE)"
echo "See experiments/EXP-FPGA-DIV-001/phase-b1.md and results/b1-comparison.md for the committed numbers this script reproduces."
