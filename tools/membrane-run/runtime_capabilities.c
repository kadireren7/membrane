#include <string.h>

#include "runtime_capabilities.h"

/* See runtime_capabilities.h's own top comment for the full contract. */

const char	*membrane_capability_state_name(membrane_capability_state_t s)
{
	if (s == MEMBRANE_CAPABILITY_UNSUPPORTED)
		return ("unsupported");
	if (s == MEMBRANE_CAPABILITY_PARTIAL)
		return ("partial");
	if (s == MEMBRANE_CAPABILITY_SUPPORTED)
		return ("supported");
	return ("unknown");
}

const char	*membrane_capability_provenance_name(
				membrane_capability_provenance_t p)
{
	if (p == MEMBRANE_CAPABILITY_PROVENANCE_COMPILED_IN)
		return ("compiled_in");
	if (p == MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT)
		return ("static_contract");
	if (p == MEMBRANE_CAPABILITY_PROVENANCE_RUNTIME_PROBE)
		return ("runtime_probe");
	if (p == MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE)
		return ("api_probe");
	if (p == MEMBRANE_CAPABILITY_PROVENANCE_VERSION_PROBE)
		return ("version_probe");
	return ("unknown");
}

const char	*membrane_runtime_type_name(membrane_runtime_type_t t)
{
	if (t == MEMBRANE_RUNTIME_TYPE_EXTERNAL)
		return ("external");
	return ("native");
}

const char	*membrane_runtime_execution_mode_name(
				membrane_runtime_execution_mode_t m)
{
	if (m == MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL)
		return ("local_external");
	if (m == MEMBRANE_RUNTIME_EXEC_REMOTE_EXTERNAL)
		return ("remote_external");
	return ("embedded_native");
}

const char	*membrane_runtime_availability_name(
				membrane_runtime_availability_t a)
{
	if (a == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
		return ("available");
	if (a == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE)
		return ("unavailable");
	return ("unknown");
}

const char	*membrane_runtime_health_name(membrane_runtime_health_t h)
{
	if (h == MEMBRANE_RUNTIME_HEALTH_UNREACHABLE)
		return ("unreachable");
	if (h == MEMBRANE_RUNTIME_HEALTH_HEALTHY)
		return ("healthy");
	if (h == MEMBRANE_RUNTIME_HEALTH_INCOMPATIBLE)
		return ("incompatible");
	if (h == MEMBRANE_RUNTIME_HEALTH_UNKNOWN)
		return ("unknown");
	return ("not_probed");
}

const char	*membrane_negotiation_result_name(
				membrane_negotiation_result_t r)
{
	if (r == MEMBRANE_NEGOTIATION_FULLY_SUPPORTED)
		return ("fully_supported");
	if (r == MEMBRANE_NEGOTIATION_PARTIAL)
		return ("partially_supported");
	return ("unsupported");
}

static void	copy_str(char *dst, size_t dst_size, const char *src)
{
	size_t	len;

	if (dst_size == 0)
		return ;
	if (src == NULL)
		src = "";
	len = strlen(src);
	if (len >= dst_size)
		len = dst_size - 1;
	memcpy(dst, src, len);
	dst[len] = '\0';
}

/*
 * Part 4: membrane-native's own capability matrix. Every field below
 * was set by reading the actual current code (see this project's own
 * H1 audit, docs/runtime-abstraction.md's capability matrix table for
 * the file:line citation behind each value) -- never a guess and never
 * a project aspiration. Deliberately a plain function, not a table
 * dispatch: there is exactly one native runtime, so there is nothing
 * to select between yet.
 */
static void	fill_native_capabilities(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));

	/* MODEL / LIFECYCLE. Enumeration: registry_core.h + model_catalog.h
	 * + `membrane model list/search/info`, server.cpp's /v1/models.
	 * Switch: server.cpp's real POST /membrane/v1/models/activate
	 * (handle_activate_model). Load/unload: PARTIAL -- loading a model
	 * (via `membrane use`/the activate endpoint) is real and explicit;
	 * there is no matching explicit "unload NAME" verb anywhere in this
	 * codebase, only automatic, non-pinned LRU eviction when resident
	 * slots are full (server.cpp's own `evictable` status field).
	 * Metadata: `membrane model inspect/info` (model_cmd.cpp). */
	c->model_enumeration = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_load_unload = MEMBRANE_CAPABILITY_PARTIAL;
	c->model_switch = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_metadata = MEMBRANE_CAPABILITY_SUPPORTED;

	/* PLANNING / CONTROL. Context/GPU-layers/quant/KV-precision/KV-
	 * placement: all real, wired-through controls (decode_loop.cpp's
	 * cp.n_ctx/cp.type_k/cp.type_v, runtime_session.cpp's mp.n_gpu_
	 * layers, kv_residency_policy.h's MEMBRANE_KV_PLACEMENT_*,
	 * variant_selector.h/the registry's own variant field), each also
	 * exposed read-only via `membrane plan --ctx/--gpu-layers/--kv/
	 * --quant`. Concurrency: PARTIAL -- decode_concurrency.h's gate is
	 * real and does bound concurrent decodes, but only via the
	 * MEMBRANE_MAX_CONCURRENT_DECODE environment variable read once at
	 * `membrane serve` startup (server.cpp), never a live per-request
	 * or per-plan dial, and membrane_plan_t itself has no concurrency
	 * field at all. Device selection: real, `--device` on membrane-run
	 * (runtime_session.cpp's membrane_select_gpu_device()/membrane_gpu_
	 * match_device()). Memory/headroom telemetry (planning-time):
	 * real -- membrane_plan_t's own feasibility.host_headroom_bytes/
	 * max_feasible_context, `membrane plan --json`. */
	c->context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->quant_variant_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->kv_precision_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->concurrency_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->device_selection = MEMBRANE_CAPABILITY_SUPPORTED;
	c->memory_headroom_telemetry = MEMBRANE_CAPABILITY_SUPPORTED;

	/* INFERENCE. All three real and end-to-end (server.cpp's POST
	 * /v1/chat/completions, SSE chunked_content_provider streaming,
	 * and the real cancel_flag/gen_cancel_flag path from a dropped
	 * client connection through to the decode loop). */
	c->chat_completions = MEMBRANE_CAPABILITY_SUPPORTED;
	c->streaming = MEMBRANE_CAPABILITY_SUPPORTED;
	c->cancellation = MEMBRANE_CAPABILITY_SUPPORTED;

	/* OBSERVABILITY. Current model: real, live (/v1/status's own
	 * resident_models[].model/state). Active context: PARTIAL -- a
	 * completed chat response echoes its own ctx_size (server.cpp's
	 * "context" response field), but no ONGOING per-resident-model
	 * context size is exposed by /v1/status. RAM/VRAM/KV-cache/loaded-
	 * model memory: PARTIAL, all four -- `membrane doctor`/`membrane
	 * plan` and /v1/status only ever expose planner-time BYTE ESTIMATES
	 * (estimated_model_bytes/estimated_kv_bytes) or host/device-wide
	 * free/total snapshots, never a live measurement of what THIS
	 * loaded model is actually using right now (no RSS read, no
	 * nvidia-smi-equivalent per-process VRAM read anywhere in this
	 * codebase). */
	c->current_model = MEMBRANE_CAPABILITY_SUPPORTED;
	c->active_context = MEMBRANE_CAPABILITY_PARTIAL;
	c->ram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->vram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->kv_cache_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->loaded_model_memory = MEMBRANE_CAPABILITY_PARTIAL;

	/* ADVANCED. Neither exists anywhere in this codebase -- see this
	 * struct's own field comment in the header. */
	c->live_kv_migration = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->dynamic_reconfiguration = MEMBRANE_CAPABILITY_UNSUPPORTED;
}

/*
 * The one real runtime descriptor. availability is unconditionally
 * AVAILABLE: the native engine is linked directly into this very
 * binary (see MEMBRANE_RUNTIME_EXEC_EMBEDDED_NATIVE's own doc comment)
 * -- there is no build configuration in which `membrane`/`membrane-run`
 * exists but the native runtime does not. This module never checks
 * whether `membrane serve`/the background service is currently running
 * (service_state.h/status_client.h) -- Part 12's own "runtime available
 * != service running" distinction, kept real by simply never reading
 * that state here at all. See docs/runtime-abstraction.md.
 */
static void	fill_native_descriptor(membrane_runtime_descriptor_t *d)
{
	memset(d, 0, sizeof(*d));
	copy_str(d->id, sizeof(d->id), MEMBRANE_RUNTIME_ID_NATIVE);
	copy_str(d->display_name, sizeof(d->display_name),
		"MEMBRANE native (llama.cpp)");
	d->type = MEMBRANE_RUNTIME_TYPE_NATIVE;
	d->execution_mode = MEMBRANE_RUNTIME_EXEC_EMBEDDED_NATIVE;
	d->availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	d->version_known = 0;			/* no separate native-runtime version
								 * distinct from MEMBRANE_VERSION itself
								 * exists to report here */
	d->endpoint_known = 0;			/* embedded -- no separate address */
	d->capability_provenance = MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT;
	fill_native_capabilities(&d->capabilities);
}

size_t	membrane_runtime_discover(membrane_runtime_descriptor_t *out,
			size_t max_out)
{
	if (max_out == 0)
		return (0);
	fill_native_descriptor(&out[0]);
	return (1);
}

int	membrane_runtime_describe(const char *runtime_id,
			membrane_runtime_descriptor_t *out)
{
	if (runtime_id == NULL)
		return (0);
	if (strcmp(runtime_id, MEMBRANE_RUNTIME_ID_NATIVE) != 0)
		return (0);
	fill_native_descriptor(out);
	return (1);
}

static void	push_unsupported(membrane_negotiation_outcome_t *out,
				const char *dim)
{
	if (out->unsupported_count >= MEMBRANE_NEGOTIATION_MAX_DIMS)
		return ;
	copy_str(out->unsupported[out->unsupported_count],
		sizeof(out->unsupported[0]), dim);
	out->unsupported_count++;
}

membrane_negotiation_outcome_t	membrane_runtime_negotiate_plan(
			const membrane_runtime_capabilities_t *caps,
			const membrane_plan_t *plan)
{
	membrane_negotiation_outcome_t	out;
	int		needed = 0;
	int		satisfied = 0;

	memset(&out, 0, sizeof(out));
	if (caps == NULL || plan == NULL)
	{
		out.result = MEMBRANE_NEGOTIATION_UNSUPPORTED;
		return (out);
	}
	if (plan->decisions.has_decisions)
	{
		needed++;
		if (caps->context_control == MEMBRANE_CAPABILITY_SUPPORTED)
			satisfied++;
		else
			push_unsupported(&out, MEMBRANE_NEGOTIATION_DIM_CONTEXT_CONTROL);
		needed++;
		if (caps->gpu_layer_control == MEMBRANE_CAPABILITY_SUPPORTED)
			satisfied++;
		else
			push_unsupported(&out,
				MEMBRANE_NEGOTIATION_DIM_GPU_LAYERS_CONTROL);
		needed++;
		if (caps->kv_precision_control == MEMBRANE_CAPABILITY_SUPPORTED)
			satisfied++;
		else
			push_unsupported(&out,
				MEMBRANE_NEGOTIATION_DIM_KV_PRECISION_CONTROL);
		needed++;
		if (caps->kv_placement_control == MEMBRANE_CAPABILITY_SUPPORTED)
			satisfied++;
		else
			push_unsupported(&out,
				MEMBRANE_NEGOTIATION_DIM_KV_PLACEMENT_CONTROL);
	}
	if (plan->identity.variant_known)
	{
		needed++;
		if (caps->quant_variant_control == MEMBRANE_CAPABILITY_SUPPORTED)
			satisfied++;
		else
			push_unsupported(&out,
				MEMBRANE_NEGOTIATION_DIM_QUANT_VARIANT_CONTROL);
	}
	if (needed == 0 || satisfied == needed)
		out.result = MEMBRANE_NEGOTIATION_FULLY_SUPPORTED;
	else if (satisfied == 0)
		out.result = MEMBRANE_NEGOTIATION_UNSUPPORTED;
	else
		out.result = MEMBRANE_NEGOTIATION_PARTIAL;
	return (out);
}
