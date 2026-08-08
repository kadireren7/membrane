#ifndef MEMBRANE_WORKLOAD_GEN_H
# define MEMBRANE_WORKLOAD_GEN_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Shared deterministic synthetic KV-like F16 block generator, used by
 * both membrane-demo (tools/membrane-demo) and membrane-bench
 * (tools/membrane-bench). Every kind here is SYNTHETIC: a deterministic
 * generator calibrated against the real Q4_0 quantize/dequantize engine
 * to exercise a range of real quantization outcomes (from easily
 * Q4-compressible to Q8-requiring), not a captured or modeled real
 * KV-cache trace from an actual LLM. Any human- or machine-readable
 * output that names a workload must label it synthetic.
 *
 * Each block is `bias + noise_amp * (0.5*smooth + 0.4*noise)`: `bias`
 * sweeps [0, 8) across the block index via a Weyl low-discrepancy
 * sequence phase-shifted by `seed` (so a large, easily-quantized DC
 * offset is common for some blocks and near-absent for others); `noise`
 * is a per-element xorshift32 stream seeded from (seed, blk_idx);
 * `smooth` is a per-element low-frequency sine term. `noise_amp` is
 * what actually separates the four kinds -- calibrated (see
 * workload_gen.c) so each produces a distinct, non-fabricated Q4/Q8
 * split under the maintained engine, not a hand-picked ratio.
 */
typedef enum e_membrane_workload_kind
{
	MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT = 0,
	MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE,
	MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE,
	MEMBRANE_WORKLOAD_SYNTHETIC_MIXED,
	MEMBRANE_WORKLOAD_COUNT
}	membrane_workload_kind_t;

/*
 * Stable, hyphenated CLI/JSON name, e.g. "synthetic-low-variance" --
 * every name this build knows starts with "synthetic-", which is also
 * exactly the "workload_kind" every result reports (see
 * MEMBRANE_WORKLOAD_KIND_LABEL below): there is no non-synthetic
 * workload in this build.
 */
const char	*membrane_workload_kind_name(membrane_workload_kind_t k);

/* Constant category label every result's "workload_kind" field carries
 * -- kept as a named constant (not inlined at each call site) so a
 * future truthful non-synthetic workload addition has one place to
 * change this, not several. */
# define MEMBRANE_WORKLOAD_KIND_LABEL	"synthetic"

/* Returns 1 and sets *out on a match (case-sensitive, exact), 0
 * otherwise. */
int			membrane_workload_kind_from_name(const char *name,
				membrane_workload_kind_t *out);

/*
 * Fills `out` (elems F16 values, elems a positive multiple of 32 --
 * the caller guarantees this, not re-checked here) for block `blk_idx`
 * of a deterministic workload of this kind/seed. Pure function of
 * (kind, seed, blk_idx, elems): calling it twice with the same
 * arguments produces bit-identical output.
 */
void	membrane_workload_generate_block(membrane_workload_kind_t kind,
			uint32_t seed, uint32_t blk_idx, uint16_t *out, uint32_t elems);

/* Overflow-checked a*b: returns 0 (and leaves *out untouched) if the
 * product would overflow uint64_t, 1 and sets *out otherwise. */
int		membrane_checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out);

# ifdef __cplusplus
}
# endif

#endif
