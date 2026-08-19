#include "kv_residency_policy.h"

#include <stdio.h>
#include <string.h>

/* Identical formula to gpu_policy.c's static reserve_for() -- reused
 * (not shared via a common header) because it is a two-line, already-
 * documented policy choice (MEMBRANE_GPU_RESERVE_FIXED_BYTES/_PCT live
 * in gpu_policy.h) and this module deliberately has zero #include
 * dependency on gpu_policy.h, staying independently testable. */
static uint64_t	reserve_for(uint64_t device_total_bytes)
{
	uint64_t	pct_reserve;

	pct_reserve = device_total_bytes / 100 * 15;
	return (pct_reserve > (uint64_t)512 * 1024 * 1024
		? pct_reserve : (uint64_t)512 * 1024 * 1024);
}

static uint64_t	margin_for(uint64_t candidate_gpu_kv_bytes)
{
	uint64_t	pct_margin;

	pct_margin = candidate_gpu_kv_bytes / 100
		* MEMBRANE_KV_RESIDENCY_MARGIN_PCT;
	return (pct_margin > MEMBRANE_KV_RESIDENCY_MARGIN_FIXED_BYTES
		? pct_margin : MEMBRANE_KV_RESIDENCY_MARGIN_FIXED_BYTES);
}

const char	*membrane_kv_placement_mode_name(int placement_mode)
{
	if (placement_mode == MEMBRANE_KV_PLACEMENT_GPU)
		return ("gpu");
	if (placement_mode == MEMBRANE_KV_PLACEMENT_CPU)
		return ("cpu");
	if (placement_mode == MEMBRANE_KV_PLACEMENT_AUTO)
		return ("auto");
	return ("default");
}

static void	fill_uniform(membrane_kv_residency_result_t *out,
				int32_t n_layer, uint8_t value)
{
	int32_t	i;

	i = 0;
	while (i < n_layer)
	{
		out->layer_on_gpu[i] = value;
		i++;
	}
}

/* Ascending layer index, GPU-first (Section 9): the simplest stable
 * deterministic order, not a sensitivity ranking. Fills exactly
 * `gpu_layers` entries with 1, the rest with 0. */
static void	fill_split(membrane_kv_residency_result_t *out,
				int32_t n_layer, int32_t gpu_layers)
{
	int32_t	i;

	i = 0;
	while (i < n_layer)
	{
		out->layer_on_gpu[i] = (i < gpu_layers) ? 1 : 0;
		i++;
	}
}

int	membrane_kv_residency_resolve(int placement_mode, int32_t n_layer,
		uint64_t kv_bytes_per_layer, uint64_t device_free_bytes,
		uint64_t device_total_bytes, uint64_t weight_bytes_selected,
		uint64_t compute_buffer_estimate_bytes,
		membrane_kv_residency_result_t *out)
{
	uint64_t	budget_bytes;
	uint64_t	after_weights;
	uint64_t	margin;
	int32_t		max_fit;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	if (n_layer <= 0 || (uint32_t)n_layer > MEMBRANE_KV_RESIDENCY_MAX_LAYERS)
	{
		snprintf(out->reason, sizeof(out->reason),
			"%s: layer count %d out of range [1, %u]",
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG, n_layer,
			MEMBRANE_KV_RESIDENCY_MAX_LAYERS);
		return (0);
	}
	if (placement_mode != MEMBRANE_KV_PLACEMENT_DEFAULT
		&& placement_mode != MEMBRANE_KV_PLACEMENT_GPU
		&& placement_mode != MEMBRANE_KV_PLACEMENT_CPU
		&& placement_mode != MEMBRANE_KV_PLACEMENT_AUTO)
	{
		snprintf(out->reason, sizeof(out->reason),
			"%s: unknown placement_mode %d",
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG, placement_mode);
		return (0);
	}
	out->n_layer = n_layer;
	out->kv_bytes_per_layer = kv_bytes_per_layer;
	out->total_kv_bytes = kv_bytes_per_layer * (uint64_t)n_layer;
	out->safety_reserve_bytes = reserve_for(device_total_bytes);
	budget_bytes = device_free_bytes > out->safety_reserve_bytes
		? device_free_bytes - out->safety_reserve_bytes : 0;
	after_weights = budget_bytes > weight_bytes_selected
		? budget_bytes - weight_bytes_selected : 0;
	after_weights = after_weights > compute_buffer_estimate_bytes
		? after_weights - compute_buffer_estimate_bytes : 0;
	if (placement_mode == MEMBRANE_KV_PLACEMENT_DEFAULT)
	{
		/* No plan computed at all: caller must leave kv_dev_override
		 * NULL and never consult layer_on_gpu/gpu_kv_layers below --
		 * this branch exists so callers that DO ask for a default-mode
		 * result get an honest, clearly-labeled empty answer rather
		 * than an uninitialized one. */
		out->ok = 1;
		snprintf(out->reason, sizeof(out->reason), "%s",
			MEMBRANE_KV_PLACEMENT_REASON_DEFAULT_UNCHANGED);
		return (1);
	}
	margin = margin_for(out->total_kv_bytes);
	out->runtime_margin_bytes = margin;
	out->budget_for_kv_bytes = after_weights > margin
		? after_weights - margin : 0;
	if (placement_mode == MEMBRANE_KV_PLACEMENT_CPU)
	{
		fill_uniform(out, n_layer, 0);
		out->gpu_kv_layers = 0;
		out->cpu_kv_layers = n_layer;
		out->gpu_kv_bytes = 0;
		out->cpu_kv_bytes = out->total_kv_bytes;
		out->ok = 1;
		snprintf(out->reason, sizeof(out->reason), "%s",
			MEMBRANE_KV_PLACEMENT_REASON_CPU_FULL);
		return (1);
	}
	if (placement_mode == MEMBRANE_KV_PLACEMENT_GPU)
	{
		if (out->total_kv_bytes > out->budget_for_kv_bytes)
		{
			snprintf(out->reason, sizeof(out->reason),
				"%s: all-GPU KV needs %llu bytes, only %llu available "
				"(reserve %llu, weights %llu, margin %llu of %llu "
				"free) -- try --kv-placement auto/cpu or a smaller --ctx",
				MEMBRANE_KV_PLACEMENT_REASON_BUDGET_INSUFFICIENT,
				(unsigned long long)out->total_kv_bytes,
				(unsigned long long)out->budget_for_kv_bytes,
				(unsigned long long)out->safety_reserve_bytes,
				(unsigned long long)weight_bytes_selected,
				(unsigned long long)margin,
				(unsigned long long)device_free_bytes);
			out->ok = 0;
			return (0);
		}
		fill_uniform(out, n_layer, 1);
		out->gpu_kv_layers = n_layer;
		out->cpu_kv_layers = 0;
		out->gpu_kv_bytes = out->total_kv_bytes;
		out->cpu_kv_bytes = 0;
		out->ok = 1;
		snprintf(out->reason, sizeof(out->reason), "%s",
			MEMBRANE_KV_PLACEMENT_REASON_GPU_FULL);
		return (1);
	}
	/* AUTO: maximize GPU-resident KV layers subject to budget --
	 * Section 8: chosen for conservative compatibility with prior
	 * (KV-follows-weights, effectively all-GPU-when-weights-are-GPU)
	 * behavior, NOT because Phase 12G found GPU KV faster (it did not
	 * -- see docs/kv-residency.md). Never fails: 0 GPU-resident layers
	 * (all-CPU) is always a valid AUTO outcome, since CPU KV is always
	 * available wherever this code runs at all. */
	if (kv_bytes_per_layer == 0)
		max_fit = n_layer;
	else
	{
		/* Compare the uint64_t quotient against n_layer BEFORE
		 * narrowing to int32_t -- a budget/kv_bytes_per_layer ratio
		 * exceeding INT32_MAX would otherwise hit an
		 * implementation-defined (possibly negative) narrowing
		 * conversion, which could wrongly select zero GPU layers
		 * despite ample budget. n_layer is always in [1,
		 * MEMBRANE_KV_RESIDENCY_MAX_LAYERS], so this cast is safe. */
		if (out->budget_for_kv_bytes / kv_bytes_per_layer
			> (uint64_t)n_layer)
			max_fit = n_layer;
		else
			max_fit = (int32_t)(out->budget_for_kv_bytes
					/ kv_bytes_per_layer);
	}
	fill_split(out, n_layer, max_fit);
	out->gpu_kv_layers = max_fit;
	out->cpu_kv_layers = n_layer - max_fit;
	out->gpu_kv_bytes = (uint64_t)max_fit * kv_bytes_per_layer;
	out->cpu_kv_bytes = out->total_kv_bytes - out->gpu_kv_bytes;
	out->ok = 1;
	snprintf(out->reason, sizeof(out->reason), "%s",
		max_fit == n_layer ? MEMBRANE_KV_PLACEMENT_REASON_GPU_FULL
		: max_fit == 0 ? MEMBRANE_KV_PLACEMENT_REASON_CPU_FULL
		: MEMBRANE_KV_PLACEMENT_REASON_AUTO_SPLIT);
	return (1);
}
