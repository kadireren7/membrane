#include "observation.h"

#include <stdio.h>
#include <string.h>

/* See observation.h's own top comment for the full contract. */

const char	*membrane_obs_provenance_name(membrane_obs_provenance_t p)
{
	switch (p)
	{
		case MEMBRANE_OBS_PROV_MEASURED:
			return ("measured");
		case MEMBRANE_OBS_PROV_RUNTIME_REPORTED:
			return ("runtime_reported");
		case MEMBRANE_OBS_PROV_ESTIMATED:
			return ("estimated");
		case MEMBRANE_OBS_PROV_STATIC_METADATA:
			return ("static_metadata");
		case MEMBRANE_OBS_PROV_CONFIGURED:
			return ("configured");
		case MEMBRANE_OBS_PROV_UNKNOWN:
		default:
			return ("unknown");
	}
}

const char	*membrane_obs_status_name(membrane_obs_status_t s)
{
	switch (s)
	{
		case MEMBRANE_OBS_STATUS_COMPLETE:
			return ("complete");
		case MEMBRANE_OBS_STATUS_PARTIAL:
			return ("partial");
		case MEMBRANE_OBS_STATUS_UNAVAILABLE:
		default:
			return ("unavailable");
	}
}

static void	copy_bounded(char *dst, size_t dst_size, const char *src)
{
	snprintf(dst, dst_size, "%s", src != NULL ? src : "");
}

void	membrane_obs_u64_unknown(membrane_obs_u64_t *f, const char *reason)
{
	f->known = 0;
	f->value = 0;
	f->provenance = MEMBRANE_OBS_PROV_UNKNOWN;
	copy_bounded(f->source, sizeof(f->source), reason);
}

void	membrane_obs_u64_set(membrane_obs_u64_t *f, uint64_t value,
			membrane_obs_provenance_t prov, const char *source)
{
	if (prov == MEMBRANE_OBS_PROV_UNKNOWN)
	{
		membrane_obs_u64_unknown(f, source);
		return ;
	}
	f->known = 1;
	f->value = value;
	f->provenance = prov;
	copy_bounded(f->source, sizeof(f->source), source);
}

void	membrane_obs_bool_unknown(membrane_obs_bool_t *f, const char *reason)
{
	f->known = 0;
	f->value = 0;
	f->provenance = MEMBRANE_OBS_PROV_UNKNOWN;
	copy_bounded(f->source, sizeof(f->source), reason);
}

void	membrane_obs_bool_set(membrane_obs_bool_t *f, int value,
			membrane_obs_provenance_t prov, const char *source)
{
	if (prov == MEMBRANE_OBS_PROV_UNKNOWN)
	{
		membrane_obs_bool_unknown(f, source);
		return ;
	}
	f->known = 1;
	f->value = value ? 1 : 0;
	f->provenance = prov;
	copy_bounded(f->source, sizeof(f->source), source);
}

void	membrane_obs_str_unknown(membrane_obs_str_t *f, const char *reason)
{
	f->known = 0;
	f->value[0] = '\0';
	f->provenance = MEMBRANE_OBS_PROV_UNKNOWN;
	copy_bounded(f->source, sizeof(f->source), reason);
}

void	membrane_obs_str_set(membrane_obs_str_t *f, const char *value,
			membrane_obs_provenance_t prov, const char *source)
{
	if (prov == MEMBRANE_OBS_PROV_UNKNOWN || value == NULL)
	{
		membrane_obs_str_unknown(f, source);
		return ;
	}
	f->known = 1;
	copy_bounded(f->value, sizeof(f->value), value);
	f->provenance = prov;
	copy_bounded(f->source, sizeof(f->source), source);
}

/* The field table, in JSON/display order. Kept as one list so JSON,
 * unknown_fields and status can never disagree (observation.h). */
#define OBS_FIELD(p, k, member) \
	do { \
		if (n < max_out) \
		{ \
			out[n].path = (p); \
			out[n].kind = (k); \
			out[n].field = &s->member; \
		} \
		n++; \
	} while (0)

size_t	membrane_obs_snapshot_fields(const membrane_observation_snapshot_t *s,
			membrane_obs_field_ref_t *out, size_t max_out)
{
	size_t	n;

	n = 0;
	OBS_FIELD("runtime.availability", MEMBRANE_OBS_KIND_STR,
		runtime_availability);
	OBS_FIELD("host_memory.total_bytes", MEMBRANE_OBS_KIND_U64,
		ram_total_bytes);
	OBS_FIELD("host_memory.available_bytes", MEMBRANE_OBS_KIND_U64,
		ram_available_bytes);
	OBS_FIELD("host_memory.used_bytes", MEMBRANE_OBS_KIND_U64,
		ram_used_bytes);
	OBS_FIELD("host_memory.swap_total_bytes", MEMBRANE_OBS_KIND_U64,
		swap_total_bytes);
	OBS_FIELD("host_memory.swap_free_bytes", MEMBRANE_OBS_KIND_U64,
		swap_free_bytes);
	OBS_FIELD("host_memory.process_rss_bytes", MEMBRANE_OBS_KIND_U64,
		process_rss_bytes);
	OBS_FIELD("gpu.device_count", MEMBRANE_OBS_KIND_U64, gpu_device_count);
	OBS_FIELD("gpu.backend", MEMBRANE_OBS_KIND_STR, gpu_backend);
	OBS_FIELD("gpu.device_name", MEMBRANE_OBS_KIND_STR, gpu_device_name);
	OBS_FIELD("gpu.device_description", MEMBRANE_OBS_KIND_STR,
		gpu_device_description);
	OBS_FIELD("gpu.vram_total_bytes", MEMBRANE_OBS_KIND_U64,
		vram_total_bytes);
	OBS_FIELD("gpu.vram_free_bytes", MEMBRANE_OBS_KIND_U64, vram_free_bytes);
	OBS_FIELD("gpu.vram_used_bytes", MEMBRANE_OBS_KIND_U64, vram_used_bytes);
	OBS_FIELD("model.configured", MEMBRANE_OBS_KIND_STR, configured_model);
	OBS_FIELD("model.configured_registered", MEMBRANE_OBS_KIND_BOOL,
		configured_model_registered);
	OBS_FIELD("model.file_size_bytes", MEMBRANE_OBS_KIND_U64,
		model_file_size_bytes);
	OBS_FIELD("model.quant", MEMBRANE_OBS_KIND_STR, model_quant);
	OBS_FIELD("model.arch", MEMBRANE_OBS_KIND_STR, model_arch);
	OBS_FIELD("model.resident_count", MEMBRANE_OBS_KIND_U64,
		resident_model_count);
	OBS_FIELD("context.active", MEMBRANE_OBS_KIND_U64, context_active);
	OBS_FIELD("context.planned", MEMBRANE_OBS_KIND_U64, context_planned);
	OBS_FIELD("context.model_max", MEMBRANE_OBS_KIND_U64, context_model_max);
	OBS_FIELD("kv.planned_precision", MEMBRANE_OBS_KIND_STR,
		kv_planned_precision);
	OBS_FIELD("kv.planned_placement", MEMBRANE_OBS_KIND_STR,
		kv_planned_placement);
	OBS_FIELD("kv.estimated_bytes", MEMBRANE_OBS_KIND_U64,
		kv_estimated_bytes);
	OBS_FIELD("kv.measured_bytes", MEMBRANE_OBS_KIND_U64, kv_measured_bytes);
	OBS_FIELD("service.manager", MEMBRANE_OBS_KIND_STR, service_manager);
	OBS_FIELD("service.installed", MEMBRANE_OBS_KIND_BOOL, service_installed);
	OBS_FIELD("service.active", MEMBRANE_OBS_KIND_BOOL, service_active);
	OBS_FIELD("service.endpoint", MEMBRANE_OBS_KIND_STR, server_endpoint);
	OBS_FIELD("service.reachable", MEMBRANE_OBS_KIND_BOOL, server_reachable);
	OBS_FIELD("service.server_version", MEMBRANE_OBS_KIND_STR,
		server_version);
	OBS_FIELD("headroom.ram_bytes", MEMBRANE_OBS_KIND_U64,
		ram_headroom_bytes);
	OBS_FIELD("headroom.vram_bytes", MEMBRANE_OBS_KIND_U64,
		vram_headroom_bytes);
	return (n);
}

#undef OBS_FIELD

int	membrane_obs_field_known(const membrane_obs_field_ref_t *r)
{
	if (r->kind == MEMBRANE_OBS_KIND_U64)
		return (((const membrane_obs_u64_t *)r->field)->known);
	if (r->kind == MEMBRANE_OBS_KIND_BOOL)
		return (((const membrane_obs_bool_t *)r->field)->known);
	return (((const membrane_obs_str_t *)r->field)->known);
}

membrane_obs_provenance_t	membrane_obs_field_provenance(
								const membrane_obs_field_ref_t *r)
{
	if (r->kind == MEMBRANE_OBS_KIND_U64)
		return (((const membrane_obs_u64_t *)r->field)->provenance);
	if (r->kind == MEMBRANE_OBS_KIND_BOOL)
		return (((const membrane_obs_bool_t *)r->field)->provenance);
	return (((const membrane_obs_str_t *)r->field)->provenance);
}

const char	*membrane_obs_field_source(const membrane_obs_field_ref_t *r)
{
	if (r->kind == MEMBRANE_OBS_KIND_U64)
		return (((const membrane_obs_u64_t *)r->field)->source);
	if (r->kind == MEMBRANE_OBS_KIND_BOOL)
		return (((const membrane_obs_bool_t *)r->field)->source);
	return (((const membrane_obs_str_t *)r->field)->source);
}

void	membrane_obs_snapshot_count(const membrane_observation_snapshot_t *s,
			size_t *known, size_t *total)
{
	membrane_obs_field_ref_t	refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t						n;
	size_t						i;
	size_t						k;

	n = membrane_obs_snapshot_fields(s, refs, MEMBRANE_OBS_MAX_FIELDS);
	if (n > MEMBRANE_OBS_MAX_FIELDS)
		n = MEMBRANE_OBS_MAX_FIELDS;
	k = 0;
	i = 0;
	while (i < n)
	{
		if (membrane_obs_field_known(&refs[i]))
			k++;
		i++;
	}
	*known = k;
	*total = n;
}

void	membrane_obs_snapshot_init(membrane_observation_snapshot_t *s,
			const char *runtime_id)
{
	membrane_obs_field_ref_t	refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t						n;
	size_t						i;

	memset(s, 0, sizeof(*s));
	s->schema_version = MEMBRANE_OBSERVATION_SCHEMA_VERSION;
	copy_bounded(s->runtime_id, sizeof(s->runtime_id), runtime_id);
	s->status = MEMBRANE_OBS_STATUS_UNAVAILABLE;
	n = membrane_obs_snapshot_fields(s, refs, MEMBRANE_OBS_MAX_FIELDS);
	i = 0;
	while (i < n && i < MEMBRANE_OBS_MAX_FIELDS)
	{
		/* The table hands out const pointers; init owns *s, so casting
		 * the constness back off here is sound. */
		if (refs[i].kind == MEMBRANE_OBS_KIND_U64)
			membrane_obs_u64_unknown((membrane_obs_u64_t *)refs[i].field,
				"not_collected");
		else if (refs[i].kind == MEMBRANE_OBS_KIND_BOOL)
			membrane_obs_bool_unknown((membrane_obs_bool_t *)refs[i].field,
				"not_collected");
		else
			membrane_obs_str_unknown((membrane_obs_str_t *)refs[i].field,
				"not_collected");
		i++;
	}
}

/* out = a - b, iff both known and b <= a. Provenance: the weaker of the
 * two inputs -- equal (MEASURED) for every current caller. */
static void	derive_difference(membrane_obs_u64_t *out,
				const membrane_obs_u64_t *a, const membrane_obs_u64_t *b,
				const char *source)
{
	if (!a->known || !b->known)
	{
		membrane_obs_u64_unknown(out, "inputs_unknown");
		return ;
	}
	if (b->value > a->value)
	{
		membrane_obs_u64_unknown(out, "inputs_inconsistent");
		return ;
	}
	if (a->provenance != b->provenance)
	{
		membrane_obs_u64_unknown(out, "inputs_mixed_provenance");
		return ;
	}
	membrane_obs_u64_set(out, a->value - b->value, a->provenance, source);
}

static void	derive_copy(membrane_obs_u64_t *out, const membrane_obs_u64_t *in,
				const char *source)
{
	if (!in->known)
	{
		membrane_obs_u64_unknown(out, "inputs_unknown");
		return ;
	}
	membrane_obs_u64_set(out, in->value, in->provenance, source);
}

void	membrane_obs_snapshot_finalize(membrane_observation_snapshot_t *s)
{
	size_t	known;
	size_t	total;

	derive_difference(&s->ram_used_bytes, &s->ram_total_bytes,
		&s->ram_available_bytes, "derived:total_minus_available");
	/* free > total is inconsistent device data; headroom is then as
	 * unknown as used is. */
	if (s->vram_free_bytes.known && s->vram_total_bytes.known
		&& s->vram_free_bytes.value > s->vram_total_bytes.value)
		membrane_obs_u64_unknown(&s->vram_headroom_bytes,
			"inputs_inconsistent");
	else
		derive_copy(&s->vram_headroom_bytes, &s->vram_free_bytes,
			"derived:vram_free");
	derive_difference(&s->vram_used_bytes, &s->vram_total_bytes,
		&s->vram_free_bytes, "derived:total_minus_free");
	derive_copy(&s->ram_headroom_bytes, &s->ram_available_bytes,
		"derived:ram_available");

	if (!s->runtime_observable)
	{
		s->status = MEMBRANE_OBS_STATUS_UNAVAILABLE;
		return ;
	}
	membrane_obs_snapshot_count(s, &known, &total);
	s->status = (known == total) ? MEMBRANE_OBS_STATUS_COMPLETE
		: MEMBRANE_OBS_STATUS_PARTIAL;
}

/* Howard Hinnant's days_from_civil inverse (civil_from_days) -- public
 * domain, exact for the proleptic Gregorian calendar. */
static void	civil_from_days(int64_t z, int64_t *y, unsigned *m, unsigned *d)
{
	int64_t		era;
	unsigned	doe;
	unsigned	yoe;
	unsigned	doy;
	unsigned	mp;

	z += 719468;
	era = (z >= 0 ? z : z - 146096) / 146097;
	doe = (unsigned)(z - era * 146097);
	yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
	*y = (int64_t)yoe + era * 400;
	doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	mp = (5 * doy + 2) / 153;
	*d = doy - (153 * mp + 2) / 5 + 1;
	*m = mp < 10 ? mp + 3 : mp - 9;
	if (*m <= 2)
		*y += 1;
}

int	membrane_obs_format_utc(int64_t unix_ms, char *out, size_t out_size)
{
	int64_t		secs;
	int64_t		days;
	int64_t		rem;
	int64_t		year;
	unsigned	month;
	unsigned	day;

	if (out == NULL || out_size < 25 || unix_ms < 0)
		return (0);
	secs = unix_ms / 1000;
	days = secs / 86400;
	rem = secs % 86400;
	civil_from_days(days, &year, &month, &day);
	snprintf(out, out_size, "%04lld-%02u-%02uT%02d:%02d:%02d.%03dZ",
		(long long)year, month, day, (int)(rem / 3600),
		(int)((rem % 3600) / 60), (int)(rem % 60), (int)(unix_ms % 1000));
	return (1);
}
