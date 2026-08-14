#include "adaptive_kv_policy.h"

#include <string.h>

static void	set_result(membrane_adaptive_kv_result_t *out, int ok,
				int mode, const membrane_adaptive_kv_candidate_t *winner,
				const char *reason)
{
	out->ok = ok;
	out->selected_mode = ok ? mode : 0;
	out->selected_layers = (ok && winner != NULL) ? winner->selected_layers
		: 0;
	out->selected_kv_bytes = (ok && winner != NULL) ? winner->kv_bytes : 0;
	strncpy(out->reason, reason, sizeof(out->reason) - 1);
	out->reason[sizeof(out->reason) - 1] = '\0';
}

static int	resolve_cpu(const membrane_adaptive_kv_candidate_t *q8,
				const membrane_adaptive_kv_candidate_t *q5,
				membrane_adaptive_kv_result_t *out)
{
	if (q8->valid)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q8, q8,
			MEMBRANE_ADAPTIVE_REASON_CPU_ADAPTIVE_Q8_DEFAULT);
		return (1);
	}
	if (q5->valid)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q5, q5,
			MEMBRANE_ADAPTIVE_REASON_CPU_MEMORY_PRESSURE_Q5);
		return (1);
	}
	set_result(out, 0, 0, NULL, MEMBRANE_ADAPTIVE_REASON_NO_COMPRESSED_MODE_FITS);
	return (0);
}

/* Neither candidate reached full residency (or is invalid): compare
 * the safe partial-offload plans. Q8 wins ties -- "do not sacrifice
 * quality for zero practical memory benefit" (Section 7). */
static int	resolve_gpu_partial(const membrane_adaptive_kv_candidate_t *q8,
				const membrane_adaptive_kv_candidate_t *q5,
				membrane_adaptive_kv_result_t *out)
{
	if (q8->valid && q5->valid)
	{
		if (q5->selected_layers > q8->selected_layers)
			set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q5, q5,
				MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_MEMORY_GUARD);
		else
			set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q8, q8,
				MEMBRANE_ADAPTIVE_REASON_Q8_FITS);
		return (1);
	}
	if (q5->valid)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q5, q5,
			MEMBRANE_ADAPTIVE_REASON_Q5_ONLY_COMPRESSED_MODE_THAT_FITS);
		return (1);
	}
	if (q8->valid)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q8, q8,
			MEMBRANE_ADAPTIVE_REASON_Q8_FITS);
		return (1);
	}
	set_result(out, 0, 0, NULL, MEMBRANE_ADAPTIVE_REASON_NO_COMPRESSED_MODE_FITS);
	return (0);
}

static int	resolve_gpu(const membrane_adaptive_kv_candidate_t *q8,
				const membrane_adaptive_kv_candidate_t *q5,
				membrane_adaptive_kv_result_t *out)
{
	if (q8->valid && q8->full_residency)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q8, q8,
			MEMBRANE_ADAPTIVE_REASON_Q8_FULL_RESIDENCY);
		return (1);
	}
	if (q5->valid && q5->full_residency)
	{
		set_result(out, 1, MEMBRANE_ADAPTIVE_KV_MODE_Q5, q5,
			MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_FULL_RESIDENCY);
		return (1);
	}
	return (resolve_gpu_partial(q8, q5, out));
}

int	membrane_adaptive_kv_resolve(
		const membrane_adaptive_kv_candidate_t *q8,
		const membrane_adaptive_kv_candidate_t *q5, int is_gpu_backend,
		membrane_adaptive_kv_result_t *out)
{
	if (out == NULL || q8 == NULL || q5 == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	if (is_gpu_backend)
		return (resolve_gpu(q8, q5, out));
	return (resolve_cpu(q8, q5, out));
}
