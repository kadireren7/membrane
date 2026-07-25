#ifndef MEMBRANE_POLICY_H
# define MEMBRANE_POLICY_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"
# include "membrane/hash.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * A precision-map policy: the output of an offline optimizer (Phase
 * 3.4-3.6), loaded at inference time to drive a real per-layer KV cache
 * (Phase 4.1). The numeric values are the storage bit-width, so an
 * out-of-range byte on disk is trivially rejected as unsupported.
 */
typedef enum e_membrane_precision
{
	MEMBRANE_PRECISION_FP16 = 16,
	MEMBRANE_PRECISION_Q8 = 8,
	MEMBRANE_PRECISION_Q4 = 4
}	membrane_precision_t;

typedef struct s_membrane_policy	membrane_policy_t;

/*
 * The identity of the model/runtime actually in use, to check a loaded
 * policy against before trusting it. model_sha256 is the raw digest
 * (see membrane_sha256_file); llama_cpp_commit is the 40-hex-char commit
 * the running binary was built against.
 */
typedef struct s_membrane_policy_context
{
	uint8_t		model_sha256[MEMBRANE_SHA256_DIGEST_BYTES];
	const char	*llama_cpp_commit;
	uint32_t	layer_count;
}	membrane_policy_context_t;

/*
 * Plain field bundle for membrane_policy_save -- not a handle, just the
 * values to write. k_prec/v_prec must each have layer_count entries.
 */
typedef struct s_membrane_policy_build
{
	uint8_t						model_sha256[MEMBRANE_SHA256_DIGEST_BYTES];
	const char					*llama_cpp_commit;
	uint32_t					layer_count;
	const membrane_precision_t	*k_prec;
	const membrane_precision_t	*v_prec;
	const char					*model_name;
	const char					*tier_name;
	double						cosine_min;
	double						top1_min;
	double						top5_min;
	double						cosine_margin;
	double						top1_margin;
	double						top5_margin;
	uint32_t					search_budget;
	uint32_t					evals_used;
	uint64_t					created_unix_time;
}	membrane_policy_build_t;

/*
 * Writes a policy file to `path` from `b`. Returns
 * MEMBRANE_ERR_INVALID_ARG for a NULL k_prec/v_prec, zero layer_count, an
 * out-of-range precision value, or a model_name/tier_name/commit string
 * too long to fit the on-disk fields; MEMBRANE_ERR_IO on a write failure.
 */
membrane_status_t	membrane_policy_save(const char *path,
						const membrane_policy_build_t *b);

/*
 * Parses the policy file at `path`: magic, format version, and the
 * trailing CRC32 checksum (covering everything before it) are all
 * checked before any field is trusted. Returns MEMBRANE_ERR_IO if the
 * file cannot be read, MEMBRANE_ERR_CORRUPT_DATA for a bad magic/version,
 * truncated file, checksum mismatch, or an out-of-range precision byte,
 * MEMBRANE_ERR_ALLOC_FAILED on allocation failure. On any error *out is
 * left NULL. Never touches global state -- the returned handle owns its
 * own heap allocation.
 */
membrane_status_t	membrane_policy_load(const char *path,
						membrane_policy_t **out);

/*
 * Checks a successfully-loaded policy against the model/runtime it is
 * about to be applied to. Returns MEMBRANE_OK if the policy's recorded
 * model hash, llama.cpp commit, and layer count all match `ctx`.
 * Otherwise returns MEMBRANE_ERR_MISMATCH and, if reason_buf is non-NULL,
 * writes a human-readable explanation of which field(s) mismatched
 * (truncated to reason_cap - 1 bytes, always NUL-terminated).
 */
membrane_status_t	membrane_policy_validate(const membrane_policy_t *policy,
						const membrane_policy_context_t *ctx,
						char *reason_buf, size_t reason_cap);

/*
 * Looks up the precision this policy assigns to one layer's K or V
 * cache. Returns MEMBRANE_ERR_INVALID_ARG if policy/out is NULL or
 * `layer` is out of range for this policy's layer count.
 */
membrane_status_t	membrane_policy_query(const membrane_policy_t *policy,
						uint32_t layer, int is_v,
						membrane_precision_t *out);

uint32_t			membrane_policy_layer_count(
						const membrane_policy_t *policy);

/* Frees a policy returned by membrane_policy_load. NULL is a no-op. */
void				membrane_policy_destroy(membrane_policy_t *policy);

# ifdef __cplusplus
}
# endif

#endif
