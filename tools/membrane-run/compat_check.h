#ifndef MEMBRANE_RUN_COMPAT_CHECK_H
# define MEMBRANE_RUN_COMPAT_CHECK_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Llama-free (no ggml/llama header) pre-flight check for --kv q8,
 * Section 9: fail clearly BEFORE creating a context, rather than
 * however llama.cpp's own validation happens to fail deep inside
 * llama_init_from_model. Pure function over plain integers/strings so
 * it's testable (test_compat_check.c) without a real model.
 *
 * Scope, matching this project's own documented limits
 * (docs/compatibility.md): compressed KV (q8/q5/adaptive) is verified
 * for LLM_ARCH_LLAMA and (Phase 26) LLM_ARCH_QWEN2 only -- an explicit
 * allowlist in compat_check.c, not a broad architecture-family
 * heuristic; every other architecture string still fails closed here
 * regardless of backend. Flash-attention *availability* itself is a
 * runtime-resolved property of the backend (see llama-context.cpp's
 * resolve_fused_ops) that cannot be soundly determined without
 * actually constructing a context -- this check does not claim to
 * predict it; it only rejects the inputs that are knowable in advance
 * to make context construction fail (architecture, head-dimension/
 * block-size divisibility, degenerate shapes).
 */

# define MEMBRANE_Q8_0_BLOCK_SIZE	32u
/* Phase 10C: Q5_1's own ggml block size, checked explicitly rather
 * than reused from MEMBRANE_Q8_0_BLOCK_SIZE above -- both happen to
 * be 32 in the pinned ggml commit, but that is a fact about ggml's
 * block formats, not something this header should assume stays true;
 * see membrane_check_kv_compat()'s kv_mode-specific block-size
 * selection. */
# define MEMBRANE_Q5_1_BLOCK_SIZE	32u

typedef struct s_membrane_compat_result
{
	int		ok;
	char	reason[256];	/* human-readable, only meaningful if !ok */
}	membrane_compat_result_t;

/*
 * arch_name: model's "general.architecture" GGUF metadata value (e.g.
 * "llama"), or NULL/empty if unknown -- treated as unsupported (fail
 * closed, not fail open).
 * n_embd/n_head/n_head_kv: real model hparams (llama_model_n_*).
 * ctx_size: the resolved (nonzero) context size the caller intends to
 * use.
 * kv_mode: MEMBRANE_KV_STORE_NATIVE (always ok, no checks),
 * MEMBRANE_KV_STORE_Q8, or MEMBRANE_KV_STORE_Q5 (see
 * kv_store_telemetry.h for the constants; not included here to keep
 * this header dependency-free -- pass the raw 0/1/2 value).
 *
 * Returns 1 (out->ok = 1) if supported, 0 otherwise with out->reason
 * filled in. Never crashes on NULL/zero/negative inputs.
 */
int	membrane_check_kv_compat(const char *arch_name, int32_t n_embd,
		int32_t n_head, int32_t n_head_kv, uint32_t ctx_size, int kv_mode,
		membrane_compat_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
