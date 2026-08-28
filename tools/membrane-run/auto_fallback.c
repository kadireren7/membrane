/* clock_gettime()/CLOCK_MONOTONIC need this feature-test macro under
 * strict -std=c11 (this project's own CMAKE_C_STANDARD, no GNU
 * extensions) -- same fix tools/membrane-demo/demo_core.c already
 * uses for the identical reason. Must be defined before any header is
 * included. */
#define _POSIX_C_SOURCE 200809L

#include "auto_fallback.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

/* Phase 24: this module stays llama-free (clock_gettime()/CLOCK_MONOTONIC
 * are libc, not ggml/llama) -- same "no product-specific dependency"
 * contract every other pure module in this project already keeps. */
static double	seconds_since_ts(const struct timespec *t0)
{
	struct timespec	t1;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	return ((double)(t1.tv_sec - t0->tv_sec)
		+ (double)(t1.tv_nsec - t0->tv_nsec) / 1e9);
}

const char	*membrane_apply_failure_class_name(int failure_class)
{
	if (failure_class == MEMBRANE_APPLY_OK)
		return ("APPLY_OK");
	if (failure_class == MEMBRANE_APPLY_MODEL_LOAD_FAILED)
		return ("MODEL_LOAD_FAILED");
	if (failure_class == MEMBRANE_APPLY_CONTEXT_CREATE_FAILED)
		return ("CONTEXT_CREATE_FAILED");
	if (failure_class == MEMBRANE_APPLY_BACKEND_ALLOCATION_FAILED)
		return ("BACKEND_ALLOCATION_FAILED");
	if (failure_class == MEMBRANE_APPLY_GPU_OOM_CONFIRMED)
		return ("GPU_OOM_CONFIRMED");
	if (failure_class == MEMBRANE_APPLY_HOST_OOM_CONFIRMED)
		return ("HOST_OOM_CONFIRMED");
	if (failure_class == MEMBRANE_APPLY_DEVICE_LOST)
		return ("DEVICE_LOST");
	if (failure_class == MEMBRANE_APPLY_COMPAT_REJECTED)
		return ("COMPAT_REJECTED");
	return ("UNKNOWN_APPLY_FAILURE");
}

int	membrane_apply_failure_is_retryable(int failure_class)
{
	return (failure_class == MEMBRANE_APPLY_MODEL_LOAD_FAILED
		|| failure_class == MEMBRANE_APPLY_CONTEXT_CREATE_FAILED
		|| failure_class == MEMBRANE_APPLY_BACKEND_ALLOCATION_FAILED
		|| failure_class == MEMBRANE_APPLY_GPU_OOM_CONFIRMED
		|| failure_class == MEMBRANE_APPLY_HOST_OOM_CONFIRMED
		|| failure_class == MEMBRANE_APPLY_DEVICE_LOST);
}

/* Section 4: [selected_index] first, then every other ELIGIBLE
 * candidate index in the array's own ascending order -- never an
 * ineligible one (Phase 20 already rejected those; attempting one here
 * would mean instantiating a plan Phase 20 itself already determined
 * doesn't fit/isn't compatible), never a duplicate. */
static int	build_order(const membrane_joint_candidate_t *candidates,
				int candidate_count, int selected_index, int *order)
{
	int	i;
	int	n = 0;

	if (selected_index >= 0 && selected_index < candidate_count
		&& candidates[selected_index].eligible)
		order[n++] = selected_index;
	i = 0;
	while (i < candidate_count)
	{
		if (i != selected_index && candidates[i].eligible)
			order[n++] = i;
		i++;
	}
	return (n);
}

/* Section 11: recomputes a GPU candidate's fit against a freshly
 * queried free-bytes figure, using the SAME "weight + kv <= free -
 * reserve" formula joint_planner.c's own build_candidate() already
 * used at planning time -- reserve_bytes itself is not re-derived here
 * (gpu_policy.h's reserve is a pure function of device_total_bytes
 * alone, which cannot change mid-run; only the free-bytes side is
 * time-varying). A CPU-only candidate (gpu_layers <= 0) always still
 * fits regardless of GPU memory -- there is nothing to refit. */
static int	still_fits(const membrane_joint_candidate_t *c,
				uint64_t reserve_bytes, uint64_t refreshed_free_bytes)
{
	uint64_t	budget;
	uint64_t	need;

	if (c->gpu_layers <= 0)
		return (1);
	budget = refreshed_free_bytes > reserve_bytes
		? refreshed_free_bytes - reserve_bytes : 0;
	need = c->estimated_weight_gpu_bytes + c->estimated_kv_gpu_bytes;
	return (need <= budget);
}

int	membrane_fallback_run(const membrane_joint_candidate_t *candidates,
		int candidate_count, int selected_index,
		uint64_t device_total_bytes, uint64_t reserve_bytes,
		membrane_fallback_apply_fn apply_fn, void *apply_ctx,
		membrane_fallback_refresh_fn refresh_fn, void *refresh_ctx,
		membrane_fallback_trace_t *out)
{
	int							order[MEMBRANE_JOINT_MAX_CANDIDATES];
	int							n_order;
	int							apply_attempts;
	int							i;
	membrane_fallback_attempt_t	*entry;
	membrane_fallback_apply_result_t	ar;

	(void)device_total_bytes;
	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->initial_candidate_index = selected_index;
	out->final_candidate_index = -1;
	if (candidates == NULL || apply_fn == NULL || candidate_count <= 0
		|| candidate_count > MEMBRANE_JOINT_MAX_CANDIDATES
		|| selected_index < 0 || selected_index >= candidate_count
		|| !candidates[selected_index].eligible)
	{
		snprintf(out->final_status, sizeof(out->final_status), "exhausted");
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_FALLBACK_REASON_EXHAUSTED);
		return (0);
	}
	n_order = build_order(candidates, candidate_count, selected_index, order);
	apply_attempts = 0;
	i = 0;
	while (i < n_order && apply_attempts < MEMBRANE_MAX_AUTO_ATTEMPTS)
	{
		const membrane_joint_candidate_t	*c = &candidates[order[i]];

		entry = &out->entries[out->n_entries];
		memset(entry, 0, sizeof(*entry));
		entry->candidate_index = order[i];
		entry->gpu_layers = c->gpu_layers;
		entry->kv_precision = c->kv_precision;
		entry->kv_placement = c->kv_placement;
		entry->fit_after_refresh = 1;
		if (refresh_fn != NULL)
		{
			uint64_t	free_bytes;

			if (refresh_fn(refresh_ctx, &free_bytes))
			{
				entry->refreshed = 1;
				entry->refreshed_available_gpu_bytes = free_bytes;
				entry->fit_after_refresh = still_fits(c, reserve_bytes,
						free_bytes);
			}
		}
		out->n_entries++;
		if (!entry->fit_after_refresh)
		{
			i++;
			continue ;
		}
		entry->apply_started = 1;
		apply_attempts++;
		memset(&ar, 0, sizeof(ar));
		{
			struct timespec	apply_t0;

			clock_gettime(CLOCK_MONOTONIC, &apply_t0);
			entry->apply_ok = apply_fn(c, order[i], apply_ctx, &ar);
			entry->apply_wall_ms = seconds_since_ts(&apply_t0) * 1000.0;
		}
		entry->failure_class = ar.failure_class;
		snprintf(entry->detail, sizeof(entry->detail), "%s", ar.detail);
		entry->cleanup_complete = ar.cleanup_complete;
		entry->model_reload_required = ar.model_reload_required;
		out->attempt_count = apply_attempts;
		/* Section 14: "attempted" reflects whether the outcome ever
		 * departed from the originally-selected candidate -- a memory-
		 * refit SKIP (Section 11/22) is exactly as much a real fallback
		 * event as an apply failure is (the eventual candidate is not
		 * the one originally planned), so this is n_entries > 1 (has
		 * the controller moved past evaluating just candidate #0 in
		 * iteration order for ANY reason), not apply_attempts > 1 (which
		 * would wrongly under-report a skip-then-succeed run as
		 * "attempted": false). */
		out->attempted = out->n_entries > 1;
		if (entry->apply_ok)
		{
			out->final_candidate_index = order[i];
			snprintf(out->final_status, sizeof(out->final_status), "success");
			snprintf(out->reason_code, sizeof(out->reason_code), "%s",
				out->attempted ? MEMBRANE_FALLBACK_REASON_FALLBACK_SUCCEEDED
					: MEMBRANE_FALLBACK_REASON_PRIMARY_SUCCEEDED);
			return (1);
		}
		if (!entry->cleanup_complete)
		{
			snprintf(out->final_status, sizeof(out->final_status),
				"cleanup_blocked");
			snprintf(out->reason_code, sizeof(out->reason_code), "%s",
				MEMBRANE_FALLBACK_REASON_CLEANUP_FAILED);
			return (0);
		}
		if (!membrane_apply_failure_is_retryable(entry->failure_class))
		{
			/* CodeRabbit review (PR #31): this is always an IMMEDIATE
			 * stop on a non-retryable failure class (Section 3: no
			 * second attempt), never a "ran out of retryable
			 * candidates" exhaustion -- regardless of how many earlier
			 * entries were memory-refit SKIPS (out->attempted can
			 * already be true from those). NO_RETRYABLE_FAILURE
			 * unconditionally reports the real reason this specific
			 * stop happened; MEMBRANE_FALLBACK_REASON_EXHAUSTED is
			 * reserved for this function's own end-of-loop fallthrough
			 * below, where every eligible candidate really was tried. */
			snprintf(out->final_status, sizeof(out->final_status),
				"exhausted");
			snprintf(out->reason_code, sizeof(out->reason_code), "%s",
				MEMBRANE_FALLBACK_REASON_NO_RETRYABLE_FAILURE);
			return (0);
		}
		i++;
	}
	out->attempt_count = apply_attempts;
	out->attempted = out->n_entries > 1;
	snprintf(out->final_status, sizeof(out->final_status), "exhausted");
	snprintf(out->reason_code, sizeof(out->reason_code), "%s",
		MEMBRANE_FALLBACK_REASON_EXHAUSTED);
	return (0);
}
