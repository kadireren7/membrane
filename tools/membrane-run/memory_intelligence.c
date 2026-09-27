#include "memory_intelligence.h"

#include <stdio.h>
#include <string.h>

/*
 * See memory_intelligence.h's own top comment for the module contract.
 *
 * THRESHOLD POLICY (Part 5 of the I3 task) -- explicit, deterministic,
 * conservative, heuristic. These are WARNING TRIPWIRES, not safety
 * guarantees: crossing a threshold does not mean an allocation failure is
 * imminent, and staying under one does not mean it is safe, guaranteed or
 * optimal (Part 5's own prohibition on those three words -- neither
 * appears anywhere in this file's own generated text). They were chosen
 * to be loose enough not to fire on ordinary, healthy hosts and tight
 * enough to fire meaningfully before a host that is actually out of
 * memory. They are not tuned to this project's own 5.6 GiB development
 * host (Part 5's own "do not overfit to this host").
 *
 * RAM headroom uses BOTH a ratio and an absolute floor (Part 14: "also
 * consider absolute thresholds so tiny-memory systems are handled
 * sensibly"), because a ratio alone is misleading at both extremes: 5%
 * of 512 GiB is 25.6 GiB (plenty), while 15% of 2 GiB is 300 MiB (not
 * much). Either condition alone is enough to classify -- whichever is
 * more severe wins. VRAM uses ratio only: this project has no evidence
 * a VRAM-specific absolute floor should differ from the RAM one, and
 * Part 15 does not ask for one.
 *
 * Boundary semantics (Part 25), all strict "<" (a value exactly AT a
 * threshold does NOT cross it -- tested at exactly the boundary, one
 * unit below and one unit above in test_memory_intelligence.c):
 *   ratio  < CRITICAL_RATIO        -> critical
 *   bytes  < CRITICAL_ABS_BYTES    -> critical
 *   ratio  < WARNING_RATIO         -> warning (already excluded critical)
 *   bytes  < WARNING_ABS_BYTES     -> warning (already excluded critical)
 *   context_ratio >= NEAR_MAX_RATIO -> near max (">=", the mirror image:
 *     reaching the boundary IS "near")
 */
#define MEMBRANE_INTEL_RAM_CRITICAL_RATIO		0.05
#define MEMBRANE_INTEL_RAM_WARNING_RATIO		0.15
#define MEMBRANE_INTEL_RAM_CRITICAL_ABS_BYTES	(256ull * 1024 * 1024)
#define MEMBRANE_INTEL_RAM_WARNING_ABS_BYTES	(768ull * 1024 * 1024)

#define MEMBRANE_INTEL_VRAM_CRITICAL_RATIO		0.05
#define MEMBRANE_INTEL_VRAM_WARNING_RATIO		0.15

#define MEMBRANE_INTEL_CONTEXT_NEAR_MAX_RATIO	0.90

/* KV_FOOTPRINT_ESTIMATE_HIGH: the planner's own ESTIMATED KV bytes
 * against the host's own MEASURED ram headroom. ">" the fraction below
 * (not ">="), and a known ram_headroom_bytes of exactly 0 with any
 * nonzero KV estimate always counts as "high" (a 0-byte headroom makes
 * the fraction undefined, and there is nothing conservative about
 * treating undefined as "not high"). */
#define MEMBRANE_INTEL_KV_HEADROOM_FRACTION		0.5

static const char	*membrane_intel_status_names[] =
	{"ok", "notice", "warning", "critical", "insufficient_data"};

const char	*membrane_intel_status_name(membrane_intel_status_t s)
{
	if ((int)s < 0 || (size_t)s >= sizeof(membrane_intel_status_names)
			/ sizeof(membrane_intel_status_names[0]))
		return ("ok");
	return (membrane_intel_status_names[s]);
}

static const char	*membrane_intel_severity_names[] =
	{"info", "notice", "warning", "critical"};

const char	*membrane_intel_severity_name(membrane_intel_severity_t s)
{
	if ((int)s < 0 || (size_t)s >= sizeof(membrane_intel_severity_names)
			/ sizeof(membrane_intel_severity_names[0]))
		return ("info");
	return (membrane_intel_severity_names[s]);
}

static const char	*membrane_intel_dimension_names[] =
	{"host_memory", "vram", "context", "kv", "model_residency",
		"runtime_capability", "telemetry"};

const char	*membrane_intel_dimension_name(membrane_intel_dimension_t d)
{
	if ((int)d < 0 || (size_t)d >= sizeof(membrane_intel_dimension_names)
			/ sizeof(membrane_intel_dimension_names[0]))
		return ("telemetry");
	return (membrane_intel_dimension_names[d]);
}

static const char	*membrane_intel_applicability_names[] =
	{"advisory_only", "unavailable_on_runtime", "partially_controllable",
		"controllable"};

const char	*membrane_intel_applicability_name(
				membrane_intel_applicability_t a)
{
	if ((int)a < 0 || (size_t)a
			>= sizeof(membrane_intel_applicability_names)
			/ sizeof(membrane_intel_applicability_names[0]))
		return ("advisory_only");
	return (membrane_intel_applicability_names[a]);
}

/* ------------------------------------------------------------------ */
/* Small formatting helpers -- evidence text only, never used for logic */
/* ------------------------------------------------------------------ */

static void	fmt_bytes(char *out, size_t out_size, uint64_t bytes)
{
	if (bytes >= (1ull << 30))
		snprintf(out, out_size, "%.2f GiB",
			(double)bytes / (double)(1ull << 30));
	else
		snprintf(out, out_size, "%.1f MiB",
			(double)bytes / (double)(1ull << 20));
}

static void	fmt_pct(char *out, size_t out_size, double ratio)
{
	snprintf(out, out_size, "%.1f%%", ratio * 100.0);
}

/* ------------------------------------------------------------------ */
/* Capability -> applicability (Part 8)                                */
/* ------------------------------------------------------------------ */

static membrane_intel_applicability_t	applicability_from_capability(
				membrane_capability_state_t c)
{
	if (c == MEMBRANE_CAPABILITY_SUPPORTED)
		return (MEMBRANE_INTEL_APPLICABILITY_CONTROLLABLE);
	if (c == MEMBRANE_CAPABILITY_PARTIAL)
		return (MEMBRANE_INTEL_APPLICABILITY_PARTIALLY_CONTROLLABLE);
	if (c == MEMBRANE_CAPABILITY_UNSUPPORTED)
		return (MEMBRANE_INTEL_APPLICABILITY_UNAVAILABLE_ON_RUNTIME);
	/* MEMBRANE_CAPABILITY_UNKNOWN, or caps == NULL callers passing the
	 * zero value -- never upgraded to CONTROLLABLE (this header's own
	 * "fail-closed default" comment). */
	return (MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY);
}

/* ------------------------------------------------------------------ */
/* Finding / reason / recommendation builders                          */
/* ------------------------------------------------------------------ */

static void	copy_bounded(char *dst, size_t dst_size, const char *src)
{
	snprintf(dst, dst_size, "%s", src != NULL ? src : "");
}

static void	push_reason(membrane_memory_intelligence_t *out, const char *code)
{
	size_t	i;

	i = 0;
	while (i < out->reason_count)
	{
		if (strcmp(out->reasons[i], code) == 0)
			return ;
		i++;
	}
	if (out->reason_count >= MEMBRANE_INTEL_MAX_TOP_REASONS)
		return ;
	copy_bounded(out->reasons[out->reason_count], MEMBRANE_INTEL_CODE_MAX,
		code);
	out->reason_count++;
}

/* Returns the finding just appended, or NULL if the array is full --
 * every call site below checks membrane_intel_finding_count() itself
 * before deciding a recommendation depends on this finding, so a full
 * array (12 slots, far more than any single assessment below produces)
 * never silently drops a recommendation's own evidence. */
static membrane_intel_finding_t	*push_finding(
				membrane_memory_intelligence_t *out, const char *code,
				membrane_intel_severity_t severity, const char *summary,
				membrane_obs_provenance_t prov,
				membrane_intel_dimension_t dim)
{
	membrane_intel_finding_t	*f;

	if (out->finding_count >= MEMBRANE_INTEL_MAX_FINDINGS)
		return (NULL);
	f = &out->findings[out->finding_count];
	memset(f, 0, sizeof(*f));
	copy_bounded(f->code, MEMBRANE_INTEL_CODE_MAX, code);
	f->severity = severity;
	copy_bounded(f->summary, MEMBRANE_INTEL_SUMMARY_MAX, summary);
	f->provenance = prov;
	f->dimension = dim;
	out->finding_count++;
	push_reason(out, code);
	return (f);
}

static void	push_evidence(membrane_intel_finding_t *f, const char *text)
{
	if (f == NULL || f->evidence_count >= MEMBRANE_INTEL_MAX_EVIDENCE)
		return ;
	copy_bounded(f->evidence[f->evidence_count], MEMBRANE_INTEL_EVIDENCE_MAX,
		text);
	f->evidence_count++;
}

static void	push_recommendation(membrane_memory_intelligence_t *out,
				const char *code, const char *action,
				const char *reasons[], size_t reason_count,
				const char *runtime_id, membrane_capability_state_t cap,
				membrane_intel_applicability_t applicability_override,
				int use_override)
{
	membrane_intel_recommendation_t	*r;
	size_t								i;

	if (out->recommendation_count >= MEMBRANE_INTEL_MAX_RECOMMENDATIONS)
		return ;
	r = &out->recommendations[out->recommendation_count];
	memset(r, 0, sizeof(*r));
	copy_bounded(r->code, MEMBRANE_INTEL_CODE_MAX, code);
	copy_bounded(r->action, MEMBRANE_INTEL_ACTION_MAX, action);
	i = 0;
	while (i < reason_count && i < MEMBRANE_INTEL_MAX_REASON_CODES)
	{
		copy_bounded(r->reason_codes[i], MEMBRANE_INTEL_CODE_MAX, reasons[i]);
		i++;
	}
	r->reason_code_count = i;
	copy_bounded(r->affected_runtime, MEMBRANE_RUNTIME_ID_MAX, runtime_id);
	r->runtime_capability = cap;
	r->applicability = use_override ? applicability_override
		: applicability_from_capability(cap);
	r->mutates_state = 0;
	out->recommendation_count++;
}

/* ------------------------------------------------------------------ */
/* Assembly                                                            */
/* ------------------------------------------------------------------ */

static int	all_core_dims_unknown(const membrane_observation_snapshot_t *s)
{
	if (s->ram_headroom_bytes.known || s->ram_total_bytes.known)
		return (0);
	if (s->vram_headroom_bytes.known || s->vram_total_bytes.known)
		return (0);
	if (s->context_planned.known || s->context_model_max.known)
		return (0);
	if (s->kv_estimated_bytes.known)
		return (0);
	if (s->resident_model_count.known)
		return (0);
	return (1);
}

static void	assess_ram(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out, int *ram_present,
				char *ram_code_out, size_t ram_code_out_size)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];
	char						avail[32];
	char						total[32];
	char						pct[16];
	double						ratio;
	int							critical;
	int							warning;

	*ram_present = 0;
	ram_code_out[0] = '\0';
	if (!s->ram_headroom_bytes.known || !s->ram_total_bytes.known
			|| s->ram_total_bytes.value == 0)
		return ;
	ratio = (double)s->ram_headroom_bytes.value
		/ (double)s->ram_total_bytes.value;
	critical = ratio < MEMBRANE_INTEL_RAM_CRITICAL_RATIO
		|| s->ram_headroom_bytes.value < MEMBRANE_INTEL_RAM_CRITICAL_ABS_BYTES;
	warning = !critical && (ratio < MEMBRANE_INTEL_RAM_WARNING_RATIO
		|| s->ram_headroom_bytes.value < MEMBRANE_INTEL_RAM_WARNING_ABS_BYTES);
	if (!critical && !warning)
		return ;
	fmt_bytes(avail, sizeof(avail), s->ram_headroom_bytes.value);
	fmt_bytes(total, sizeof(total), s->ram_total_bytes.value);
	fmt_pct(pct, sizeof(pct), ratio);
	snprintf(buf, sizeof(buf), "Available: %s / %s (%s)", avail, total, pct);
	f = push_finding(out, critical
			? MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL
			: MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW,
		critical ? MEMBRANE_INTEL_SEVERITY_CRITICAL
			: MEMBRANE_INTEL_SEVERITY_WARNING,
		critical ? "Host memory headroom is critically low."
			: "Host memory headroom is low.",
		s->ram_headroom_bytes.provenance, MEMBRANE_INTEL_DIM_HOST_MEMORY);
	push_evidence(f, buf);
	*ram_present = 1;
	copy_bounded(ram_code_out, ram_code_out_size, critical
		? MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL
		: MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW);
}

static void	assess_vram(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out, int *vram_present,
				char *vram_code_out, size_t vram_code_out_size)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];
	char						free_s[32];
	char						total[32];
	char						pct[16];
	double						ratio;
	int							critical;
	int							warning;

	*vram_present = 0;
	vram_code_out[0] = '\0';
	if (!s->gpu_device_count.known || s->gpu_device_count.value == 0)
		return ;
	if (!s->vram_headroom_bytes.known || !s->vram_total_bytes.known
			|| s->vram_total_bytes.value == 0)
		return ;
	ratio = (double)s->vram_headroom_bytes.value
		/ (double)s->vram_total_bytes.value;
	critical = ratio < MEMBRANE_INTEL_VRAM_CRITICAL_RATIO;
	warning = !critical && ratio < MEMBRANE_INTEL_VRAM_WARNING_RATIO;
	if (!critical && !warning)
		return ;
	fmt_bytes(free_s, sizeof(free_s), s->vram_headroom_bytes.value);
	fmt_bytes(total, sizeof(total), s->vram_total_bytes.value);
	fmt_pct(pct, sizeof(pct), ratio);
	snprintf(buf, sizeof(buf), "Free: %s / %s (%s)", free_s, total, pct);
	f = push_finding(out, critical
			? MEMBRANE_INTEL_CODE_VRAM_HEADROOM_CRITICAL
			: MEMBRANE_INTEL_CODE_VRAM_HEADROOM_LOW,
		critical ? MEMBRANE_INTEL_SEVERITY_CRITICAL
			: MEMBRANE_INTEL_SEVERITY_WARNING,
		critical ? "Device VRAM headroom is critically low."
			: "Device VRAM headroom is low.",
		s->vram_headroom_bytes.provenance, MEMBRANE_INTEL_DIM_VRAM);
	push_evidence(f, buf);
	*vram_present = 1;
	copy_bounded(vram_code_out, vram_code_out_size, critical
		? MEMBRANE_INTEL_CODE_VRAM_HEADROOM_CRITICAL
		: MEMBRANE_INTEL_CODE_VRAM_HEADROOM_LOW);
}

static void	assess_context(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];
	double						ratio;

	if (!s->context_planned.known || !s->context_model_max.known
			|| s->context_model_max.value == 0)
		return ;
	snprintf(buf, sizeof(buf), "Planned: %llu / model max: %llu",
		(unsigned long long)s->context_planned.value,
		(unsigned long long)s->context_model_max.value);
	if (s->context_planned.value > s->context_model_max.value)
	{
		f = push_finding(out, MEMBRANE_INTEL_CODE_CONTEXT_EXCEEDS_MODEL_MAXIMUM,
			MEMBRANE_INTEL_SEVERITY_WARNING,
			"Planned context exceeds the model's own trained maximum.",
			s->context_planned.provenance, MEMBRANE_INTEL_DIM_CONTEXT);
		push_evidence(f, buf);
		return ;
	}
	ratio = (double)s->context_planned.value
		/ (double)s->context_model_max.value;
	if (ratio >= MEMBRANE_INTEL_CONTEXT_NEAR_MAX_RATIO)
	{
		f = push_finding(out, MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM,
			MEMBRANE_INTEL_SEVERITY_NOTICE,
			"Planned context is near the model's own trained maximum.",
			s->context_planned.provenance, MEMBRANE_INTEL_DIM_CONTEXT);
		push_evidence(f, buf);
	}
}

static void	assess_kv(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out, int *kv_present,
				char *kv_code_out, size_t kv_code_out_size)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];
	char						kv_s[32];
	char						head_s[32];
	int							high;

	*kv_present = 0;
	kv_code_out[0] = '\0';
	if (!s->kv_estimated_bytes.known || !s->ram_headroom_bytes.known)
		return ;
	if (s->ram_headroom_bytes.value == 0)
		high = s->kv_estimated_bytes.value > 0;
	else
		high = (double)s->kv_estimated_bytes.value
			/ (double)s->ram_headroom_bytes.value
			> MEMBRANE_INTEL_KV_HEADROOM_FRACTION;
	if (!high)
		return ;
	fmt_bytes(kv_s, sizeof(kv_s), s->kv_estimated_bytes.value);
	fmt_bytes(head_s, sizeof(head_s), s->ram_headroom_bytes.value);
	snprintf(buf, sizeof(buf), "Estimated KV: %s vs. current headroom: %s",
		kv_s, head_s);
	f = push_finding(out, MEMBRANE_INTEL_CODE_KV_FOOTPRINT_ESTIMATE_HIGH,
		MEMBRANE_INTEL_SEVERITY_NOTICE,
		"The planner's KV footprint estimate is high relative to current "
		"host memory headroom.",
		s->kv_estimated_bytes.provenance, MEMBRANE_INTEL_DIM_KV);
	push_evidence(f, buf);
	*kv_present = 1;
	copy_bounded(kv_code_out, kv_code_out_size,
		MEMBRANE_INTEL_CODE_KV_FOOTPRINT_ESTIMATE_HIGH);
}

static void	assess_residency(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out, int *residency_present)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];

	*residency_present = 0;
	if (!s->resident_model_count.known || s->resident_model_count.value <= 1)
		return ;
	snprintf(buf, sizeof(buf), "%llu models resident",
		(unsigned long long)s->resident_model_count.value);
	f = push_finding(out, MEMBRANE_INTEL_CODE_MULTIPLE_MODELS_RESIDENT,
		MEMBRANE_INTEL_SEVERITY_INFO,
		"More than one model is currently resident.",
		s->resident_model_count.provenance,
		MEMBRANE_INTEL_DIM_MODEL_RESIDENCY);
	push_evidence(f, buf);
	*residency_present = 1;
}

static void	assess_capability_disclosures(
				const membrane_observation_snapshot_t *s,
				const membrane_runtime_capabilities_t *caps,
				membrane_memory_intelligence_t *out)
{
	membrane_intel_finding_t	*f;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];
	int							kv_relevant;

	if (caps == NULL)
		return ;
	/* Relevant whenever there is something to control the KV precision/
	 * placement OF: either Planner v2 produced a KV plan (native), or a
	 * model is simply resident (Ollama -- Planner v2 never runs for an
	 * externally-owned model, so kv_planned_precision/_placement stay
	 * unknown there by construction; gating on them alone would make
	 * these two findings unreachable for Ollama, which is wrong -- see
	 * docs/observability.md section 8's own native/Ollama field split). */
	kv_relevant = s->kv_planned_precision.known
		|| (s->resident_model_count.known && s->resident_model_count.value
			>= 1);
	if (kv_relevant
			&& caps->kv_precision_control != MEMBRANE_CAPABILITY_SUPPORTED)
	{
		snprintf(buf, sizeof(buf), "kv_precision_control = %s",
			membrane_capability_state_name(caps->kv_precision_control));
		f = push_finding(out,
			MEMBRANE_INTEL_CODE_KV_PRECISION_CONTROL_UNAVAILABLE,
			MEMBRANE_INTEL_SEVERITY_NOTICE,
			"This runtime does not expose KV precision control through "
			"MEMBRANE.", MEMBRANE_OBS_PROV_STATIC_METADATA,
			MEMBRANE_INTEL_DIM_RUNTIME_CAPABILITY);
		push_evidence(f, buf);
	}
	if (kv_relevant
			&& caps->kv_placement_control != MEMBRANE_CAPABILITY_SUPPORTED)
	{
		snprintf(buf, sizeof(buf), "kv_placement_control = %s",
			membrane_capability_state_name(caps->kv_placement_control));
		f = push_finding(out,
			MEMBRANE_INTEL_CODE_KV_PLACEMENT_CONTROL_UNAVAILABLE,
			MEMBRANE_INTEL_SEVERITY_NOTICE,
			"This runtime does not expose KV placement control through "
			"MEMBRANE.", MEMBRANE_OBS_PROV_STATIC_METADATA,
			MEMBRANE_INTEL_DIM_RUNTIME_CAPABILITY);
		push_evidence(f, buf);
	}
	if (s->gpu_device_count.known && s->gpu_device_count.value > 0
			&& caps->gpu_layer_control == MEMBRANE_CAPABILITY_PARTIAL)
	{
		f = push_finding(out, MEMBRANE_INTEL_CODE_GPU_LAYER_CONTROL_PARTIAL,
			MEMBRANE_INTEL_SEVERITY_INFO,
			"This runtime only partially controls GPU layer offload.",
			MEMBRANE_OBS_PROV_STATIC_METADATA,
			MEMBRANE_INTEL_DIM_RUNTIME_CAPABILITY);
		push_evidence(f, "gpu_layer_control = partial");
	}
	if (s->context_planned.known
			&& caps->context_control == MEMBRANE_CAPABILITY_SUPPORTED)
	{
		f = push_finding(out, MEMBRANE_INTEL_CODE_CONTEXT_CONTROL_AVAILABLE,
			MEMBRANE_INTEL_SEVERITY_INFO,
			"This runtime exposes explicit context control through "
			"MEMBRANE.", MEMBRANE_OBS_PROV_STATIC_METADATA,
			MEMBRANE_INTEL_DIM_RUNTIME_CAPABILITY);
		push_evidence(f, "context_control = supported");
	}
}

static void	assess_telemetry(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out)
{
	membrane_intel_finding_t	*f;
	size_t						known;
	size_t						total;
	char						buf[MEMBRANE_INTEL_EVIDENCE_MAX];

	if (s->status != MEMBRANE_OBS_STATUS_PARTIAL)
		return ;
	membrane_obs_snapshot_count(s, &known, &total);
	snprintf(buf, sizeof(buf), "%zu/%zu observation fields known", known,
		total);
	f = push_finding(out, MEMBRANE_INTEL_CODE_OBSERVATION_PARTIAL,
		MEMBRANE_INTEL_SEVERITY_INFO,
		"Some observation fields are unknown; this assessment only "
		"uses the ones that are.", MEMBRANE_OBS_PROV_MEASURED,
		MEMBRANE_INTEL_DIM_TELEMETRY);
	push_evidence(f, buf);
}

static void	set_model_identity(const membrane_observation_snapshot_t *s,
				membrane_memory_intelligence_t *out)
{
	if (s->resident_model_len > 0 && s->resident_models[0].name.known)
	{
		out->model_known = 1;
		copy_bounded(out->model_id, MEMBRANE_INTEL_ID_MAX,
			s->resident_models[0].name.value);
		return ;
	}
	if (s->configured_model.known)
	{
		out->model_known = 1;
		copy_bounded(out->model_id, MEMBRANE_INTEL_ID_MAX,
			s->configured_model.value);
		return ;
	}
	out->model_known = 0;
	out->model_id[0] = '\0';
}

/* highest_below_critical: 0 = nothing above INFO seen yet, 1 = at least
 * one NOTICE, 2 = at least one WARNING -- a plain running max over
 * finding_count entries, not a stored/opaque score (Part 17). */
static membrane_intel_status_t	status_from_findings(
				const membrane_memory_intelligence_t *out)
{
	int		highest_below_critical;
	size_t	i;

	highest_below_critical = 0;
	i = 0;
	while (i < out->finding_count)
	{
		if (out->findings[i].severity == MEMBRANE_INTEL_SEVERITY_CRITICAL)
			return (MEMBRANE_INTEL_STATUS_CRITICAL);
		if (out->findings[i].severity == MEMBRANE_INTEL_SEVERITY_WARNING)
			highest_below_critical = 2;
		else if (out->findings[i].severity == MEMBRANE_INTEL_SEVERITY_NOTICE
				&& highest_below_critical < 2)
			highest_below_critical = 1;
		i++;
	}
	if (highest_below_critical == 2)
		return (MEMBRANE_INTEL_STATUS_WARNING);
	if (highest_below_critical == 1)
		return (MEMBRANE_INTEL_STATUS_NOTICE);
	return (MEMBRANE_INTEL_STATUS_OK);
}

membrane_intel_status_t	membrane_memory_intelligence_assemble(
		const membrane_observation_snapshot_t *snap,
		const membrane_runtime_capabilities_t *caps,
		membrane_memory_intelligence_t *out)
{
	int		ram_present;
	int		vram_present;
	int		kv_present;
	int		residency_present;
	char	ram_code[MEMBRANE_INTEL_CODE_MAX];
	char	vram_code[MEMBRANE_INTEL_CODE_MAX];
	char	kv_code[MEMBRANE_INTEL_CODE_MAX];
	const char	*reasons[MEMBRANE_INTEL_MAX_REASON_CODES];
	size_t		reason_n;
	int			kv_relevant;

	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_INTEL_SCHEMA_VERSION;
	if (snap == NULL)
	{
		out->status = MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA;
		push_finding(out, MEMBRANE_INTEL_CODE_TELEMETRY_INCOMPLETE,
			MEMBRANE_INTEL_SEVERITY_WARNING,
			"No observation snapshot was available to assess.",
			MEMBRANE_OBS_PROV_UNKNOWN, MEMBRANE_INTEL_DIM_TELEMETRY);
		return (out->status);
	}
	copy_bounded(out->runtime_id, MEMBRANE_RUNTIME_ID_MAX, snap->runtime_id);
	set_model_identity(snap, out);
	if (snap->status == MEMBRANE_OBS_STATUS_UNAVAILABLE
			|| all_core_dims_unknown(snap))
	{
		out->status = MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA;
		push_finding(out, MEMBRANE_INTEL_CODE_TELEMETRY_INCOMPLETE,
			MEMBRANE_INTEL_SEVERITY_WARNING,
			snap->status == MEMBRANE_OBS_STATUS_UNAVAILABLE
				? "The runtime could not be observed at all."
				: "No memory-relevant fields were known in this snapshot.",
			MEMBRANE_OBS_PROV_UNKNOWN, MEMBRANE_INTEL_DIM_TELEMETRY);
		return (out->status);
	}

	/* Fixed phase order == the Part 17 ranking; see this file's own top
	 * comment and memory_intelligence.h's own assemble() doc comment. */
	assess_ram(snap, out, &ram_present, ram_code, sizeof(ram_code));
	assess_vram(snap, out, &vram_present, vram_code, sizeof(vram_code));
	assess_residency(snap, out, &residency_present);
	assess_context(snap, out);
	assess_kv(snap, out, &kv_present, kv_code, sizeof(kv_code));
	assess_capability_disclosures(snap, caps, out);
	assess_telemetry(snap, out);

	/* Same relevance rule as assess_capability_disclosures() above --
	 * a KV plan (native) or a merely-resident model (Ollama) both make
	 * "could this runtime's KV precision/placement be changed" a
	 * meaningful question. */
	kv_relevant = snap->kv_planned_precision.known
		|| (snap->resident_model_count.known
			&& snap->resident_model_count.value >= 1);

	/* Recommendations -- capability-aware (Part 8), never applied. */
	if (ram_present)
	{
		reasons[0] = ram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_REDUCE_CONTEXT,
			"Reduce context if you experience allocation failures.",
			reasons, 1, snap->runtime_id,
			caps != NULL ? caps->context_control : MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 0);
	}
	if (strcmp(ram_code, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL)
			== 0)
	{
		reasons[0] = ram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_USE_LOWER_MEMORY_VARIANT,
			"A smaller/lower-precision model variant would need less host "
			"memory.", reasons, 1, snap->runtime_id,
			caps != NULL ? caps->quant_variant_control
				: MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 0);
	}
	if (vram_present)
	{
		reasons[0] = vram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_REDUCE_GPU_LAYERS,
			"Reduce GPU layers if you experience allocation failures.",
			reasons, 1, snap->runtime_id,
			caps != NULL ? caps->gpu_layer_control
				: MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 0);
	}
	if (residency_present && (ram_present || vram_present))
	{
		reason_n = 0;
		reasons[reason_n++] = MEMBRANE_INTEL_CODE_MULTIPLE_MODELS_RESIDENT;
		if (reason_n < MEMBRANE_INTEL_MAX_REASON_CODES)
			reasons[reason_n++] = ram_present ? ram_code : vram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL,
			"Consider unloading an unused model if memory pressure "
			"continues.", reasons, reason_n, snap->runtime_id,
			caps != NULL ? caps->model_load_unload
				: MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 1);
	}
	if (kv_relevant && (kv_present || vram_present || ram_present))
	{
		reasons[0] = kv_present ? kv_code : vram_present ? vram_code
			: ram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION,
			"Lower KV precision could reduce KV memory.", reasons, 1,
			snap->runtime_id,
			caps != NULL ? caps->kv_precision_control
				: MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 0);
	}
	if (kv_relevant && (kv_present || vram_present || ram_present))
	{
		reasons[0] = kv_present ? kv_code : vram_present ? vram_code
			: ram_code;
		push_recommendation(out, MEMBRANE_INTEL_REC_CHANGE_KV_PLACEMENT,
			"Changing KV placement could relieve memory pressure on the "
			"constrained device.", reasons, 1, snap->runtime_id,
			caps != NULL ? caps->kv_placement_control
				: MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 0);
	}
	if (caps != NULL && strcmp(snap->runtime_id, MEMBRANE_RUNTIME_ID_NATIVE)
			!= 0 && (ram_present || vram_present)
			&& caps->kv_precision_control == MEMBRANE_CAPABILITY_UNSUPPORTED
			&& caps->kv_placement_control == MEMBRANE_CAPABILITY_UNSUPPORTED)
	{
		reasons[0] = MEMBRANE_INTEL_CODE_RUNTIME_CONTROL_LIMITED;
		reasons[1] = ram_present ? ram_code : vram_code;
		push_recommendation(out,
			MEMBRANE_INTEL_REC_SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL,
			"This runtime does not expose KV precision or KV placement "
			"control through MEMBRANE; MEMBRANE-native exposes both.",
			reasons, 2, snap->runtime_id, caps->kv_precision_control,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 1);
	}

	out->status = status_from_findings(out);
	if (out->status == MEMBRANE_INTEL_STATUS_OK)
		push_recommendation(out, MEMBRANE_INTEL_REC_NO_ACTION_NEEDED,
			"No memory-pressure action is indicated right now.", NULL, 0,
			snap->runtime_id, MEMBRANE_CAPABILITY_UNKNOWN,
			MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY, 1);
	return (out->status);
}
