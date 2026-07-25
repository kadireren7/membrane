#ifndef MEMBRANE_QUANT_SIMD_H
# define MEMBRANE_QUANT_SIMD_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 5.1: a portable (no llama.cpp dependency, unlike
 * membrane_ggml_quant/src/quant/ggml_quant.c), MEMBRANE-owned
 * scalar+SIMD Q8_0/Q4_0 quantize/dequantize engine, bit-exact with
 * ggml's own math (see docs/phase5-quant-engine.md for exactly which
 * ggml formula each backend replicates and why). The Phase 4.4 module
 * remains the correctness ORACLE this engine is tested against, and the
 * safe fallback if this engine is unavailable or its self-check fails.
 *
 * Block format matches ggml exactly: MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES
 * (34: 2-byte F16 scale + 32 signed int8) /
 * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES (18: 2-byte F16 scale + 16 bytes of
 * packed 4-bit pairs) -- numerically identical to
 * MEMBRANE_GGML_Q{8,4}_0_BLOCK_BYTES in the (llama-gated)
 * membrane/ggml_quant.h, redefined here under a different name so this
 * header stays free of any llama.cpp dependency.
 */

# define MEMBRANE_QSIMD_BLOCK_ELEMS		32u
# define MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES	34u
# define MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES	18u

typedef enum e_membrane_simd_backend
{
	MEMBRANE_SIMD_SCALAR = 0,
	MEMBRANE_SIMD_SSE41,
	MEMBRANE_SIMD_AVX2,
	MEMBRANE_SIMD_COUNT
}	membrane_simd_backend_t;

/*
 * Detects the best backend this CPU actually supports (checked once via
 * cpuid at first call, cached in a function-local static -- not a
 * mutable global, and safe under concurrent first-calls since the
 * detected result is idempotent regardless of which thread computes it
 * first). MEMBRANE_SIMD_SCALAR is always "supported".
 */
membrane_simd_backend_t	membrane_simd_best_backend(void);
const char					*membrane_simd_backend_name(
								membrane_simd_backend_t b);

/*
 * Quantizes/dequantizes `n` F16 elements (n a multiple of 32) using the
 * given backend explicitly -- MEMBRANE_ERR_UNIMPLEMENTED if `backend`
 * isn't supported by this CPU (never silently runs an unsupported
 * instruction). Bit-exact across every supported backend for the same
 * input, and bit-exact with membrane_ggml_quant (Phase 4.4) -- verified
 * in tests/unit/test_quant_simd_parity.c.
 */
membrane_status_t	membrane_simd_q8_0_quantize(membrane_simd_backend_t
						backend, const uint16_t *x_f16, size_t n,
						uint8_t *out);
membrane_status_t	membrane_simd_q8_0_dequantize(membrane_simd_backend_t
						backend, const uint8_t *in, size_t n,
						uint16_t *out_f16);
membrane_status_t	membrane_simd_q4_0_quantize(membrane_simd_backend_t
						backend, const uint16_t *x_f16, size_t n,
						uint8_t *out);
membrane_status_t	membrane_simd_q4_0_dequantize(membrane_simd_backend_t
						backend, const uint8_t *in, size_t n,
						uint16_t *out_f16);

/*
 * Batch API (item 5): quantizes/dequantizes `n_blocks` independent rows
 * in one call. `elems_per_row` must be a multiple of 32. `x_f16`/
 * `out_f16` are `n_blocks * elems_per_row` elements, contiguous, one row
 * after another; `packed` is `n_blocks * elems_per_row/32 *
 * MEMBRANE_GGML_Q{8,4}_0_BLOCK_BYTES` bytes, also contiguous. No
 * allocation happens inside these calls -- `scratch`/`scratch_bytes` is
 * a caller-provided buffer used for the (small, O(elems_per_row))
 * per-row F32 intermediate; membrane_simd_batch_scratch_bytes() reports
 * the minimum size. Passing a too-small scratch buffer returns
 * MEMBRANE_ERR_BUFFER_TOO_SMALL and writes nothing.
 *
 * Uses `backend` for every row if it picks a backend safely (this call
 * does not internally parallelize -- see membrane/quant_pool.h for the
 * worker-pool wrapper that calls these per-chunk across threads).
 */
size_t	membrane_simd_batch_scratch_bytes(size_t elems_per_row);

membrane_status_t	membrane_simd_q8_0_quantize_batch(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						void *scratch, size_t scratch_bytes);
membrane_status_t	membrane_simd_q8_0_dequantize_batch(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						void *scratch, size_t scratch_bytes);
membrane_status_t	membrane_simd_q4_0_quantize_batch(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						void *scratch, size_t scratch_bytes);
membrane_status_t	membrane_simd_q4_0_dequantize_batch(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						void *scratch, size_t scratch_bytes);

/*
 * Parallel worker-pool batch API (item 7): identical contract and
 * bit-exact output to the single-threaded batch functions above --
 * rows are fully independent (no shared mutable state, no cross-row
 * dependency), so splitting them across `n_threads_requested` workers
 * never changes the result, only the wall time. `n_threads_requested`
 * <= 0 uses membrane_simd_default_threads() (physical core count).
 * Below MEMBRANE_QSIMD_MIN_ROWS_PER_THREAD rows per worker, the actual
 * thread count used is reduced (down to 1) automatically, since thread
 * launch overhead would dominate a workload that small -- the number
 * of threads ACTUALLY used is returned via *out_threads_used (may be
 * passed NULL).
 */
# define MEMBRANE_QSIMD_MIN_ROWS_PER_THREAD	64u

unsigned int	membrane_simd_default_threads(void);

membrane_status_t	membrane_simd_q8_0_quantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						int n_threads_requested, int *out_threads_used);
membrane_status_t	membrane_simd_q8_0_dequantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						int n_threads_requested, int *out_threads_used);
membrane_status_t	membrane_simd_q4_0_quantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						int n_threads_requested, int *out_threads_used);
membrane_status_t	membrane_simd_q4_0_dequantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						int n_threads_requested, int *out_threads_used);

# ifdef __cplusplus
}
# endif

#endif
