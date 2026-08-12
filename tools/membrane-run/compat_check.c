#include "compat_check.h"

#include <stdio.h>
#include <string.h>

int	membrane_check_kv_compat(const char *arch_name, int32_t n_embd,
		int32_t n_head, int32_t n_head_kv, uint32_t ctx_size, int kv_mode,
		membrane_compat_result_t *out)
{
	int32_t	n_embd_head;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->ok = 1;
	if (kv_mode == 0)	/* MEMBRANE_KV_STORE_NATIVE: no checks apply */
		return (1);
	if (ctx_size < 1)
	{
		snprintf(out->reason, sizeof(out->reason),
			"context size must be at least 1");
		return (out->ok = 0, 0);
	}
	if (arch_name == NULL || arch_name[0] == '\0'
		|| strcmp(arch_name, "llama") != 0)
	{
		snprintf(out->reason, sizeof(out->reason),
			"model architecture '%s' is not validated for Q8 KV "
			"storage (only LLM_ARCH_LLAMA is supported)",
			(arch_name != NULL && arch_name[0] != '\0')
				? arch_name : "(unknown)");
		return (out->ok = 0, 0);
	}
	if (n_head <= 0 || n_embd <= 0)
	{
		snprintf(out->reason, sizeof(out->reason),
			"model reports a non-positive n_embd (%d) or n_head (%d)",
			n_embd, n_head);
		return (out->ok = 0, 0);
	}
	if (n_head_kv <= 0)
	{
		snprintf(out->reason, sizeof(out->reason),
			"model reports a non-positive n_head_kv (%d)", n_head_kv);
		return (out->ok = 0, 0);
	}
	if (n_embd % n_head != 0)
	{
		snprintf(out->reason, sizeof(out->reason),
			"n_embd (%d) is not evenly divisible by n_head (%d) -- "
			"per-head dimension cannot be derived reliably for this "
			"model", n_embd, n_head);
		return (out->ok = 0, 0);
	}
	n_embd_head = n_embd / n_head;
	if (n_embd_head % (int32_t)MEMBRANE_Q8_0_BLOCK_SIZE != 0)
	{
		snprintf(out->reason, sizeof(out->reason),
			"head dimension %d is not divisible by Q8_0 block size %u",
			n_embd_head, MEMBRANE_Q8_0_BLOCK_SIZE);
		return (out->ok = 0, 0);
	}
	return (1);
}
