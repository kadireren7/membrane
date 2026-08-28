#include "gpu_policy.h"

#include <stdio.h>
#include <string.h>

static uint64_t	reserve_for(uint64_t device_total_bytes)
{
	uint64_t	pct_reserve;

	pct_reserve = device_total_bytes / 100 * MEMBRANE_GPU_RESERVE_PCT;
	return (pct_reserve > MEMBRANE_GPU_RESERVE_FIXED_BYTES
		? pct_reserve : MEMBRANE_GPU_RESERVE_FIXED_BYTES);
}

uint64_t	membrane_gpu_reserve_bytes(uint64_t device_total_bytes)
{
	return (reserve_for(device_total_bytes));
}

int	membrane_gpu_policy_resolve(int32_t requested_layers,
		int32_t n_layer_all, uint64_t device_free_bytes,
		uint64_t device_total_bytes, uint64_t bytes_per_layer_estimate,
		uint64_t kv_bytes_estimate, membrane_gpu_policy_result_t *out)
{
	uint64_t	budget_for_weights;
	uint64_t	needed;
	int32_t		max_fit;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->estimated_kv_bytes = kv_bytes_estimate;
	out->safety_reserve_bytes = reserve_for(device_total_bytes);
	out->budget_bytes = device_free_bytes > out->safety_reserve_bytes
		? device_free_bytes - out->safety_reserve_bytes : 0;
	budget_for_weights = out->budget_bytes > kv_bytes_estimate
		? out->budget_bytes - kv_bytes_estimate : 0;
	if (n_layer_all <= 0)
	{
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_GPU_POLICY_REASON_INVALID_CONFIG);
		snprintf(out->reason, sizeof(out->reason),
			"%s: model reports a non-positive layer count (%d)",
			MEMBRANE_GPU_POLICY_REASON_INVALID_CONFIG, n_layer_all);
		return (out->ok = 0, 0);
	}
	if (requested_layers == 0)
	{
		out->ok = 1;
		out->selected_layers = 0;
		out->estimated_model_bytes = 0;
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED);
		snprintf(out->reason, sizeof(out->reason), "%s",
			MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED);
		return (1);
	}
	if (requested_layers < MEMBRANE_GPU_LAYERS_AUTO)
	{
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_GPU_POLICY_REASON_INVALID_CONFIG);
		snprintf(out->reason, sizeof(out->reason),
			"%s: unsupported requested_layers value (%d) -- must be 0, "
			"%d (all), %d (auto), or a positive layer count",
			MEMBRANE_GPU_POLICY_REASON_INVALID_CONFIG, requested_layers,
			MEMBRANE_GPU_LAYERS_ALL, MEMBRANE_GPU_LAYERS_AUTO);
		return (out->ok = 0, 0);
	}
	if (requested_layers == MEMBRANE_GPU_LAYERS_AUTO)
	{
		if (bytes_per_layer_estimate == 0)
		{
			snprintf(out->reason_code, sizeof(out->reason_code), "%s",
				MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE);
			snprintf(out->reason, sizeof(out->reason),
				"%s: per-layer model size could not be estimated -- "
				"cannot resolve --gpu-layers auto safely (refusing to "
				"guess)", MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE);
			return (out->ok = 0, 0);
		}
		max_fit = (int32_t)(budget_for_weights / bytes_per_layer_estimate);
		if (max_fit > n_layer_all)
			max_fit = n_layer_all;
		if (max_fit <= 0)
		{
			snprintf(out->reason_code, sizeof(out->reason_code), "%s",
				MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT);
			snprintf(out->reason, sizeof(out->reason),
				"%s: no layers fit safely: estimated %llu bytes/layer, "
				"budget after KV (%llu) and reserve (%llu) is only "
				"%llu bytes of %llu free",
				MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT,
				(unsigned long long)bytes_per_layer_estimate,
				(unsigned long long)kv_bytes_estimate,
				(unsigned long long)out->safety_reserve_bytes,
				(unsigned long long)budget_for_weights,
				(unsigned long long)device_free_bytes);
			return (out->ok = 0, 0);
		}
		out->ok = 1;
		out->selected_layers = max_fit;
		out->estimated_model_bytes = (uint64_t)max_fit
			* bytes_per_layer_estimate;
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			max_fit == n_layer_all ? MEMBRANE_GPU_POLICY_REASON_GPU_FULL_FIT
				: MEMBRANE_GPU_POLICY_REASON_GPU_LAYERS_CLAMPED);
		snprintf(out->reason, sizeof(out->reason), "%s", out->reason_code);
		return (1);
	}
	/* MEMBRANE_GPU_LAYERS_ALL or an explicit N: both are a REQUEST for
	 * an exact layer count (n_layer_all for "all"), checked against
	 * the budget and rejected outright if it doesn't fit -- unlike
	 * auto, this never silently picks a smaller count than asked. */
	max_fit = requested_layers == MEMBRANE_GPU_LAYERS_ALL
		? n_layer_all : requested_layers;
	if (max_fit > n_layer_all)
		max_fit = n_layer_all;
	needed = bytes_per_layer_estimate == 0 ? 0
		: (uint64_t)max_fit * bytes_per_layer_estimate;
	if (bytes_per_layer_estimate != 0 && needed > budget_for_weights)
	{
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT);
		snprintf(out->reason, sizeof(out->reason),
			"%s: insufficient estimated device memory: %d layers need "
			"~%llu bytes, only %llu available after KV (%llu) and "
			"reserve (%llu) out of %llu free -- pass --gpu-layers auto "
			"or a smaller N, or a smaller --ctx",
			MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT, max_fit,
			(unsigned long long)needed,
			(unsigned long long)budget_for_weights,
			(unsigned long long)kv_bytes_estimate,
			(unsigned long long)out->safety_reserve_bytes,
			(unsigned long long)device_free_bytes);
		return (out->ok = 0, 0);
	}
	out->ok = 1;
	out->selected_layers = max_fit;
	out->estimated_model_bytes = needed;
	snprintf(out->reason_code, sizeof(out->reason_code), "%s",
		max_fit == requested_layers || requested_layers
			== MEMBRANE_GPU_LAYERS_ALL
		? MEMBRANE_GPU_POLICY_REASON_GPU_FULL_FIT
		: MEMBRANE_GPU_POLICY_REASON_GPU_LAYERS_CLAMPED);
	snprintf(out->reason, sizeof(out->reason), "%s", out->reason_code);
	return (1);
}
