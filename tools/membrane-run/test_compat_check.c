#include <string.h>

#include "compat_check.h"
#include "test_helpers.h"

#define KV_NATIVE	0
#define KV_Q8		1
#define KV_Q5		2

static void	test_native_always_ok(void)
{
	membrane_compat_result_t	r;

	/* Even wildly bad inputs are fine for native -- no checks apply. */
	TEST_ASSERT(membrane_check_kv_compat("weird-arch", -1, 0, -5, 0,
			KV_NATIVE, &r) == 1,
		"native mode is always compatible, no checks run");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
}

static void	test_q8_llama_smollm2_shape_ok(void)
{
	membrane_compat_result_t	r;

	/* SmolLM2-135M's real hparams: n_embd=576, n_head=9 (head dim 64,
	 * divisible by 32), n_head_kv=3. */
	TEST_ASSERT(membrane_check_kv_compat("llama", 576, 9, 3, 2048, KV_Q8,
			&r) == 1, "SmolLM2-135M's real shape passes q8 compat");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
}

static void	test_q8_wrong_architecture_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("gemma3", 576, 9, 3, 2048, KV_Q8,
			&r) == 0, "non-llama architecture rejected for q8");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(strstr(r.reason, "gemma3") != NULL,
		"the offending architecture name appears in the reason");
}

/* Phase 26: qwen2 was rejected here through Phase 25 (see the historical
 * MC-17/MC-18/MC-19 UNSUPPORTED rows this test used to back, now
 * SUPPORTED -- docs/compatibility.json). Source-level investigation
 * this phase proved LLM_ARCH_QWEN2 and LLM_ARCH_LLAMA both route
 * through the exact same generic, non-SWA llama_kv_cache construction
 * path in llama.cpp's create_memory() (Qwen2 never sets swa_type), and
 * a real CPU+Vulkan q8/q5/adaptive/placement experiment against the
 * local Qwen2.5-1.5B-Instruct fixture confirmed it end to end (see
 * results/compat-expansion/validation.json). Shapes below are that
 * exact model's real hparams (verified via a throwaway
 * membrane_gpu_estimate_model() dump against the committed GGUF: n_embd
 * 1536, n_head 12, n_head_kv 2, head_dim 128 -- 128 is evenly divisible
 * by both Q8_0's and Q5_1's 32-element block size), not arbitrary
 * stand-in values. */
static void	test_q8_qwen2_real_shape_accepted(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("qwen2", 1536, 12, 2, 512,
			KV_Q8, &r) == 1,
		"Qwen2.5-1.5B's real shape passes q8 compat (Phase 26)");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
}

static void	test_q5_qwen2_real_shape_accepted(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("qwen2", 1536, 12, 2, 512,
			KV_Q5, &r) == 1,
		"Qwen2.5-1.5B's real shape passes q5 compat (Phase 26)");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
}

/* The shape guard still runs for qwen2 too -- allowlisting the
 * architecture does not bypass the head-dimension/block-alignment
 * check below it. Same synthetic bad-shape example
 * test_q8_head_dim_not_block_aligned_rejected() uses for llama (head
 * dim 80, not divisible by 32), just under the qwen2 architecture
 * string this time. */
static void	test_q8_qwen2_bad_shape_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("qwen2", 640, 8, 8, 2048,
			KV_Q8, &r) == 0,
		"qwen2 with head dim 80 (not divisible by 32) still rejected");
	TEST_ASSERT(strstr(r.reason, "80") != NULL,
		"the actual head dimension appears in the reason");
}

/* Phase 26's allowlist is a two-entry list, not a broadened family
 * heuristic -- a real, still-unsupported architecture (gemma3, no
 * MEMBRANE evidence for compressed KV at all) must still fail closed
 * exactly as it did before this phase, matching
 * test_q8_wrong_architecture_rejected() above but stated explicitly so
 * this file itself proves the allowlist didn't silently widen further
 * than intended. */
static void	test_q8_gemma3_still_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("gemma3", 1536, 12, 2, 512,
			KV_Q8, &r) == 0,
		"gemma3 (not on the Phase 26 allowlist) still rejected for q8");
	TEST_ASSERT(strstr(r.reason, "LLM_ARCH_QWEN2") != NULL,
		"the updated reason message names both allowlisted architectures");
}

static void	test_q8_unknown_architecture_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat(NULL, 576, 9, 3, 2048, KV_Q8,
			&r) == 0, "NULL architecture name fails closed, not open");
	TEST_ASSERT(membrane_check_kv_compat("", 576, 9, 3, 2048, KV_Q8,
			&r) == 0, "empty architecture name fails closed too");
}

/* This is the exact example from the Phase 8 task's own spec: "head
 * dimension 80 is not divisible by Q8_0 block size 32." -- n_embd=640,
 * n_head=8 gives head dim 80. */
static void	test_q8_head_dim_not_block_aligned_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 640, 8, 8, 2048, KV_Q8,
			&r) == 0, "head dim 80 (not divisible by 32) rejected");
	TEST_ASSERT(strstr(r.reason, "80") != NULL,
		"the actual head dimension appears in the reason");
	TEST_ASSERT(strstr(r.reason, "32") != NULL,
		"the Q8_0 block size appears in the reason");
}

static void	test_q8_head_dim_block_aligned_accepted(void)
{
	membrane_compat_result_t	r;

	/* n_embd=512, n_head=8 -> head dim 64, divisible by 32. */
	TEST_ASSERT(membrane_check_kv_compat("llama", 512, 8, 8, 2048, KV_Q8,
			&r) == 1, "head dim 64 (divisible by 32) accepted");
}

static void	test_q8_degenerate_shapes_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 0, 0, 0, 2048, KV_Q8,
			&r) == 0, "zero n_embd/n_head rejected, not a crash");
	TEST_ASSERT(membrane_check_kv_compat("llama", 576, 9, 0, 2048, KV_Q8,
			&r) == 0, "zero n_head_kv rejected");
	TEST_ASSERT(membrane_check_kv_compat("llama", -576, 9, 3, 2048, KV_Q8,
			&r) == 0, "negative n_embd rejected, not a crash");
}

static void	test_q8_uneven_embd_head_split_rejected(void)
{
	membrane_compat_result_t	r;

	/* n_embd=577 (prime-ish), n_head=9 -> 577/9 is not exact. */
	TEST_ASSERT(membrane_check_kv_compat("llama", 577, 9, 3, 2048, KV_Q8,
			&r) == 0,
		"n_embd not evenly divisible by n_head rejected rather than "
		"silently truncated");
}

static void	test_q8_zero_ctx_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 576, 9, 3, 0, KV_Q8,
			&r) == 0, "ctx_size 0 rejected");
}

/* Phase 10C Section 18: q5 (Q5_1) compatibility, checked independently
 * of q8 -- same shapes as the q8 tests above, run through KV_Q5. */
static void	test_q5_llama_smollm2_shape_ok(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 576, 9, 3, 2048, KV_Q5,
			&r) == 1, "SmolLM2-135M's real shape passes q5 compat");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
}

static void	test_q5_wrong_architecture_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("gemma3", 576, 9, 3, 2048, KV_Q5,
			&r) == 0, "non-llama architecture rejected for q5");
	TEST_ASSERT(strstr(r.reason, "Q5_1") != NULL,
		"the q5-specific reason names Q5_1 explicitly (not just \"q5\")");
}

/* This is the exact example from the Phase 8 task's own spec (same as
 * the q8 test): head dim 80 (n_embd=640, n_head=8) is not divisible by
 * 32 -- q5's own unsupported-head-dimension case. */
static void	test_q5_head_dim_not_block_aligned_rejected(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 640, 8, 8, 2048, KV_Q5,
			&r) == 0, "head dim 80 (not divisible by 32) rejected for q5");
	TEST_ASSERT(strstr(r.reason, "80") != NULL,
		"the actual head dimension appears in the reason");
	TEST_ASSERT(strstr(r.reason, "Q5_1") != NULL,
		"the reason names Q5_1");
	TEST_ASSERT(strstr(r.reason, "Q8_0") == NULL,
		"a q5 check's reason never mentions Q8_0 (no accidental aliasing)");
}

static void	test_q5_head_dim_block_aligned_accepted(void)
{
	membrane_compat_result_t	r;

	TEST_ASSERT(membrane_check_kv_compat("llama", 512, 8, 8, 2048, KV_Q5,
			&r) == 1, "head dim 64 (divisible by 32) accepted for q5");
}

static void	test_null_out_is_safe(void)
{
	TEST_ASSERT(membrane_check_kv_compat("llama", 576, 9, 3, 2048, KV_Q8,
			NULL) == 0, "NULL out pointer never crashes, just fails");
}

int	main(void)
{
	test_native_always_ok();
	test_q8_llama_smollm2_shape_ok();
	test_q8_wrong_architecture_rejected();
	test_q8_qwen2_real_shape_accepted();
	test_q5_qwen2_real_shape_accepted();
	test_q8_qwen2_bad_shape_rejected();
	test_q8_gemma3_still_rejected();
	test_q8_unknown_architecture_rejected();
	test_q8_head_dim_not_block_aligned_rejected();
	test_q8_head_dim_block_aligned_accepted();
	test_q8_degenerate_shapes_rejected();
	test_q8_uneven_embd_head_split_rejected();
	test_q8_zero_ctx_rejected();
	test_q5_llama_smollm2_shape_ok();
	test_q5_wrong_architecture_rejected();
	test_q5_head_dim_not_block_aligned_rejected();
	test_q5_head_dim_block_aligned_accepted();
	test_null_out_is_safe();
	printf("test_compat_check: all tests passed\n");
	return (0);
}
