#ifndef MEMBRANE_QUANT_BACKEND_H
# define MEMBRANE_QUANT_BACKEND_H

# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 5.4: CPU/FPGA/AUTO backend selection for quantization work.
 *
 * Honest scope note: MEMBRANE's own quant_simd.c (the CPU path here)
 * IS what a real live llama.cpp/ggml inference run uses today, via the
 * kv_type_override mechanism in tools/membrane-kv-runtime -- but that
 * happens INSIDE ggml's own internal tensor kernels at KV-cache-append
 * time, which this phase does not patch to call out to an external
 * FPGA backend (that would mean patching ggml/llama.cpp's internal
 * quantize dispatch, a much larger and riskier change than this phase's
 * remaining time budget allows -- disclosed, not attempted). What IS
 * real and backend-selectable here: any BATCH quantize/dequantize work
 * issued directly against this API (offline preprocessing, the
 * membrane-fpga-runtime tool's own benchmarking, and the composed
 * quantization-wall-time comparison docs/phase5-pcie-hardware-loop.md
 * uses for its inference-experiment numbers) genuinely runs on whichever
 * backend is selected.
 *
 * membrane_choose_quant_backend implements the AUTO decision using this
 * phase's own measured numbers (docs/phase5-pcie-hardware-loop.md
 * section on break-even analysis): a single emulated FPGA pipeline
 * (bottlenecked by membrane_dma_bridge.sv's 32-bit payload port, see
 * that file's header comment) beats single-threaded CPU scalar
 * quantize/dequantize for batches of a handful of blocks or more, but a
 * well-threaded (>=4 core) CPU path consistently wins on raw
 * throughput over this specific FPGA configuration -- so AUTO prefers
 * FPGA only when cores are scarce AND the batch is large enough to
 * amortize the DMA round-trip's fixed cost AND the FPGA's own queue
 * isn't already saturated.
 */
typedef enum e_membrane_quant_backend
{
	MEMBRANE_QUANT_BACKEND_CPU = 0,
	MEMBRANE_QUANT_BACKEND_FPGA = 1,
	MEMBRANE_QUANT_BACKEND_AUTO = 2
}	membrane_quant_backend_t;

/*
 * Batch size (in 32-element blocks) at or above which a single emulated
 * FPGA pipeline's sustained throughput starts winning over
 * single-threaded CPU scalar quantize/dequantize, per this phase's own
 * measured round-trip-latency-vs-sustained-throughput crossover
 * (docs/phase5-pcie-hardware-loop.md's break-even section) -- the
 * least favorable (highest) of the four operations' crossover points,
 * so a batch at or above this threshold is a safe FPGA choice for any
 * of Q8/Q4 encode/decode when only comparing against single-threaded
 * CPU.
 */
# define MEMBRANE_QUANT_FPGA_BREAK_EVEN_BLOCKS	4u

/*
 * Resolves MEMBRANE_QUANT_BACKEND_AUTO to a concrete choice; CPU/FPGA
 * pass through unchanged. Never returns AUTO.
 *
 *   batch_blocks: size of the batch about to be quantized/dequantized.
 *   fpga_queue_used / fpga_queue_depth: current FPGA submission queue
 *     occupancy (0/0 if no FPGA backend is available at all -- in that
 *     case AUTO always resolves to CPU, see the "used >= depth"
 *     comparison below, which is trivially true for 0/0... actually
 *     pass fpga_queue_depth=0 explicitly to mean "no FPGA available" as
 *     a distinct, clearer signal, checked first).
 *   cpu_cores_available: free (not otherwise busy) CPU cores the
 *     caller could use for a CPU-backend quantize call right now.
 */
membrane_quant_backend_t	membrane_choose_quant_backend(
			membrane_quant_backend_t requested,
			uint32_t batch_blocks, uint32_t fpga_queue_used,
			uint32_t fpga_queue_depth, uint32_t cpu_cores_available);

# ifdef __cplusplus
}
# endif

#endif
