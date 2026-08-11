#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>

#include "membrane/quant_select.h"
#include "runtime_core.h"

const char	*membrane_runtime_mode_name(membrane_runtime_mode_t m)
{
	if (m == MEMBRANE_RUNTIME_MODE_BASELINE)
		return ("baseline");
	if (m == MEMBRANE_RUNTIME_MODE_SHADOW_Q8)
		return ("shadow-q8");
	if (m == MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE)
		return ("shadow-adaptive");
	return ("unknown");
}

int	membrane_runtime_mode_from_name(const char *name,
		membrane_runtime_mode_t *out)
{
	if (name == NULL || out == NULL)
		return (0);
	if (strcmp(name, "baseline") == 0)
		return (*out = MEMBRANE_RUNTIME_MODE_BASELINE, 1);
	if (strcmp(name, "shadow-q8") == 0)
		return (*out = MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 1);
	if (strcmp(name, "shadow-adaptive") == 0)
		return (*out = MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE, 1);
	return (0);
}

typedef struct s_membrane_runtime_collector
{
	membrane_runtime_mode_t	mode;
	uint32_t					max_layers;
	uint8_t						*layer_seen;
	uint32_t					layers_seen_count;
	uint64_t					k_blocks;
	uint64_t					v_blocks;
	uint64_t					total_values_observed;
	uint64_t					tail_values_excluded;
	membrane_workload_accum_t	accum;
	uint64_t					inference_steps;
	uint64_t					generated_tokens;
	double						membrane_seconds;
	double						inference_seconds;
	uint64_t					step_block_count;
}	membrane_runtime_collector_t;

membrane_runtime_collector_t	*membrane_runtime_collector_create(
									membrane_runtime_mode_t mode,
									uint32_t max_layers)
{
	membrane_runtime_collector_t	*c;

	c = calloc(1, sizeof(*c));
	if (c == NULL)
		return (NULL);
	c->layer_seen = calloc(max_layers > 0 ? max_layers : 1, sizeof(uint8_t));
	if (c->layer_seen == NULL)
		return (free(c), NULL);
	c->mode = mode;
	c->max_layers = max_layers;
	return ((membrane_runtime_collector_t *)c);
}

void	membrane_runtime_collector_destroy(membrane_runtime_collector_t *c)
{
	membrane_runtime_collector_t	*impl;

	if (c == NULL)
		return ;
	impl = (membrane_runtime_collector_t *)c;
	free(impl->layer_seen);
	free(impl);
}

membrane_status_t	membrane_runtime_observe_tensor(
						membrane_runtime_collector_t *c,
						membrane_simd_backend_t backend, int32_t layer,
						int is_v, const uint16_t *values_f16,
						uint64_t n_elems)
{
	membrane_runtime_collector_t	*impl;
	uint64_t							full_blocks;
	uint64_t							tail;
	uint64_t							i;
	membrane_bench_policy_t			policy;
	membrane_status_t					st;

	impl = (membrane_runtime_collector_t *)c;
	if (impl == NULL || (values_f16 == NULL && n_elems > 0)
		|| layer < 0 || (uint32_t)layer >= impl->max_layers)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (impl->mode == MEMBRANE_RUNTIME_MODE_BASELINE)
		return (MEMBRANE_OK);
	if (impl->layer_seen[layer] == 0)
	{
		impl->layer_seen[layer] = 1;
		impl->layers_seen_count++;
	}
	full_blocks = n_elems / MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	tail = n_elems % MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	impl->total_values_observed += n_elems;
	impl->tail_values_excluded += tail;
	policy = impl->mode == MEMBRANE_RUNTIME_MODE_SHADOW_Q8
		? MEMBRANE_BENCH_POLICY_Q8_ONLY : MEMBRANE_BENCH_POLICY_ADAPTIVE;
	i = 0;
	while (i < full_blocks)
	{
		st = membrane_bench_process_block(backend, policy,
				values_f16 + i * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
				MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
				MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR,
				&impl->accum);
		if (st != MEMBRANE_OK)
			return (st);
		/* Incremented per block, immediately after it succeeds --
		 * matches accum (updated inside process_block above) exactly,
		 * including on an early return from a later block's failure:
		 * this tensor's already-successful blocks stay counted, rather
		 * than only being counted after the whole loop finishes (which
		 * would silently undercount k_blocks/v_blocks/step_block_count
		 * relative to accum.q4_blocks+accum.q8_blocks on a partial
		 * failure). */
		if (is_v)
			impl->v_blocks++;
		else
			impl->k_blocks++;
		impl->step_block_count++;
		i++;
	}
	return (MEMBRANE_OK);
}

void	membrane_runtime_begin_step(membrane_runtime_collector_t *c)
{
	membrane_runtime_collector_t	*impl;

	impl = (membrane_runtime_collector_t *)c;
	if (impl == NULL)
		return ;
	impl->inference_steps++;
	impl->step_block_count = 0;
}

void	membrane_runtime_end_step(membrane_runtime_collector_t *c)
{
	(void)c;
}

uint64_t	membrane_runtime_step_block_count(
				const membrane_runtime_collector_t *c)
{
	const membrane_runtime_collector_t	*impl;

	impl = (const membrane_runtime_collector_t *)c;
	if (impl == NULL)
		return (0);
	return (impl->step_block_count);
}

void	membrane_runtime_add_inference_seconds(
			membrane_runtime_collector_t *c, double seconds)
{
	membrane_runtime_collector_t	*impl;

	impl = (membrane_runtime_collector_t *)c;
	if (impl == NULL)
		return ;
	impl->inference_seconds += seconds;
}

void	membrane_runtime_add_membrane_seconds(
			membrane_runtime_collector_t *c, double seconds)
{
	membrane_runtime_collector_t	*impl;

	impl = (membrane_runtime_collector_t *)c;
	if (impl == NULL)
		return ;
	impl->membrane_seconds += seconds;
}

void	membrane_runtime_set_generated_tokens(membrane_runtime_collector_t *c,
			uint64_t n)
{
	membrane_runtime_collector_t	*impl;

	impl = (membrane_runtime_collector_t *)c;
	if (impl == NULL)
		return ;
	impl->generated_tokens = n;
}

void	membrane_runtime_finalize(const membrane_runtime_collector_t *c,
			membrane_runtime_telemetry_t *out)
{
	const membrane_runtime_collector_t	*impl;

	impl = (const membrane_runtime_collector_t *)c;
	memset(out, 0, sizeof(*out));
	if (impl == NULL)
		return ;
	out->mode = impl->mode;
	out->generated_tokens = impl->generated_tokens;
	out->inference_steps = impl->inference_steps;
	out->layers_seen = impl->layers_seen_count;
	out->k_blocks = impl->k_blocks;
	out->v_blocks = impl->v_blocks;
	out->total_blocks = impl->k_blocks + impl->v_blocks;
	out->total_values_observed = impl->total_values_observed;
	out->tail_values_excluded = impl->tail_values_excluded;
	out->accum = impl->accum;
	out->membrane_seconds = impl->membrane_seconds;
	out->inference_seconds = impl->inference_seconds;
}

double	membrane_runtime_weighted_mean_rel_l2_error(
			const membrane_runtime_telemetry_t *t)
{
	uint64_t	total;

	total = t->accum.q4_blocks + t->accum.q8_blocks;
	if (total == 0)
		return (0.0);
	return ((t->accum.q4_err.sum + t->accum.q8_err.sum) / (double)total);
}

double	membrane_runtime_max_rel_l2_error(
			const membrane_runtime_telemetry_t *t)
{
	if (t->accum.q4_err.max > t->accum.q8_err.max)
		return (t->accum.q4_err.max);
	return (t->accum.q8_err.max);
}

uint64_t	membrane_runtime_fp16_bytes_observed(
				const membrane_runtime_telemetry_t *t)
{
	return (t->total_values_observed * 2u);
}

double	membrane_runtime_theoretical_payload_reduction(
			const membrane_runtime_telemetry_t *t)
{
	uint64_t	encoded_values;
	uint64_t	fp16_bytes_full;

	encoded_values = t->total_values_observed - t->tail_values_excluded;
	fp16_bytes_full = encoded_values * 2u;
	if (fp16_bytes_full == 0 || fp16_bytes_full < t->accum.encoded_bytes)
		return (0.0);
	return ((double)(fp16_bytes_full - t->accum.encoded_bytes)
		/ (double)fp16_bytes_full);
}

int	membrane_runtime_tokens_equal(const int32_t *a, size_t na,
		const int32_t *b, size_t nb)
{
	size_t	i;

	if (na != nb)
		return (0);
	if (na == 0)
		return (1);
	if (a == NULL || b == NULL)
		return (0);
	i = 0;
	while (i < na)
	{
		if (a[i] != b[i])
			return (0);
		i++;
	}
	return (1);
}

const char	*membrane_runtime_safe_basename(const char *path)
{
	const char	*slash;

	if (path == NULL)
		return (NULL);
	slash = strrchr(path, '/');
	if (slash != NULL)
		return (slash + 1);
	return (path);
}

/* ------------------------------------------------------------------ */
/* Printers                                                             */
/* ------------------------------------------------------------------ */

void	membrane_runtime_print_human(const membrane_runtime_telemetry_t *t,
			FILE *f)
{
	double	overhead_pct;

	fprintf(f, "MEMBRANE live llama.cpp shadow runtime -- mode: %s "
		"(SHADOW: native llama KV remains authoritative; no process-"
		"memory reduction claim)\n\n", membrane_runtime_mode_name(t->mode));
	fprintf(f, "Runtime\n  generated_tokens       %llu\n"
		"  inference_steps        %llu\n",
		(unsigned long long)t->generated_tokens,
		(unsigned long long)t->inference_steps);
	fprintf(f, "KV observation\n  layers_seen             %u\n"
		"  K blocks                %llu\n  V blocks                %llu\n"
		"  total blocks            %llu\n"
		"  total values observed   %llu\n"
		"  tail values excluded    %llu\n",
		t->layers_seen, (unsigned long long)t->k_blocks,
		(unsigned long long)t->v_blocks,
		(unsigned long long)t->total_blocks,
		(unsigned long long)t->total_values_observed,
		(unsigned long long)t->tail_values_excluded);
	fprintf(f, "Precision\n  Q4 blocks               %llu\n"
		"  Q8 blocks               %llu\n",
		(unsigned long long)t->accum.q4_blocks,
		(unsigned long long)t->accum.q8_blocks);
	fprintf(f, "Storage accounting (encoded payload reduction for "
		"observed KV blocks -- NOT process memory)\n"
		"  FP16 bytes observed         %llu\n"
		"  MEMBRANE encoded payload    %llu\n"
		"  theoretical payload reduction  %.2f%%\n",
		(unsigned long long)membrane_runtime_fp16_bytes_observed(t),
		(unsigned long long)t->accum.encoded_bytes,
		membrane_runtime_theoretical_payload_reduction(t) * 100.0);
	fprintf(f, "Accuracy\n  Q4 rel-L2   mean=%.6f max=%.6f\n"
		"  Q8 rel-L2   mean=%.6f max=%.6f\n"
		"  global weighted mean rel-L2  %.6f\n",
		membrane_err_stats_mean(&t->accum.q4_err), t->accum.q4_err.max,
		membrane_err_stats_mean(&t->accum.q8_err), t->accum.q8_err.max,
		membrane_runtime_weighted_mean_rel_l2_error(t));
	overhead_pct = t->inference_seconds > 0.0
		? t->membrane_seconds / t->inference_seconds * 100.0 : 0.0;
	fprintf(f, "Runtime overhead (MEMBRANE shadow processing adds "
		"overhead by design; this is expected, not a regression)\n"
		"  MEMBRANE processing time    %.6fs\n"
		"  mean time per KV block      %.9fs\n"
		"  total inference wall time   %.6fs\n"
		"  overhead ratio              %.2f%%\n",
		t->membrane_seconds,
		t->total_blocks > 0 ? t->membrane_seconds / (double)t->total_blocks
			: 0.0,
		t->inference_seconds, overhead_pct);
}

static void	json_escape(FILE *f, const char *s)
{
	unsigned char	c;

	while (*s != '\0')
	{
		c = (unsigned char)*s;
		if (c == '"' || c == '\\')
			fprintf(f, "\\%c", c);
		else if (c == '\n')
			fprintf(f, "\\n");
		else if (c < 0x20)
			fprintf(f, "\\u%04x", c);
		else
			fputc(c, f);
		s++;
	}
}

void	membrane_runtime_print_json(const membrane_runtime_telemetry_t *t,
			const char *model_label, const char *prompt_fixture,
			const int32_t *token_ids, size_t n_token_ids, FILE *f)
{
	size_t	i;
	double	overhead_pct;

	overhead_pct = t->inference_seconds > 0.0
		? t->membrane_seconds / t->inference_seconds * 100.0 : 0.0;
	fprintf(f, "{\"schema_version\":1,\"runtime\":{\"mode\":\"%s\","
		"\"model_label\":\"", membrane_runtime_mode_name(t->mode));
	json_escape(f, model_label != NULL ? model_label : "");
	fprintf(f, "\",\"prompt_fixture\":\"");
	json_escape(f, prompt_fixture != NULL ? prompt_fixture : "");
	fprintf(f, "\",\"generated_tokens\":%llu,\"inference_steps\":%llu,"
		"\"token_ids\":[",
		(unsigned long long)t->generated_tokens,
		(unsigned long long)t->inference_steps);
	i = 0;
	while (i < n_token_ids)
	{
		if (i > 0)
			fprintf(f, ",");
		fprintf(f, "%d", token_ids[i]);
		i++;
	}
	fprintf(f, "]},\"kv_observation\":{\"layers_seen\":%u,\"k_blocks\":%llu,"
		"\"v_blocks\":%llu,\"total_blocks\":%llu,"
		"\"total_values_observed\":%llu,\"tail_values_excluded\":%llu},",
		t->layers_seen, (unsigned long long)t->k_blocks,
		(unsigned long long)t->v_blocks,
		(unsigned long long)t->total_blocks,
		(unsigned long long)t->total_values_observed,
		(unsigned long long)t->tail_values_excluded);
	fprintf(f, "\"precision\":{\"q4_blocks\":%llu,\"q8_blocks\":%llu},",
		(unsigned long long)t->accum.q4_blocks,
		(unsigned long long)t->accum.q8_blocks);
	fprintf(f, "\"storage\":{\"fp16_bytes_observed\":%llu,"
		"\"encoded_payload_bytes\":%llu,"
		"\"theoretical_payload_reduction_ratio\":%.6f},",
		(unsigned long long)membrane_runtime_fp16_bytes_observed(t),
		(unsigned long long)t->accum.encoded_bytes,
		membrane_runtime_theoretical_payload_reduction(t));
	fprintf(f, "\"accuracy\":{\"q4_mean_rel_l2_error\":%.6f,"
		"\"q4_max_rel_l2_error\":%.6f,\"q8_mean_rel_l2_error\":%.6f,"
		"\"q8_max_rel_l2_error\":%.6f,\"weighted_mean_rel_l2_error\":%.6f},",
		membrane_err_stats_mean(&t->accum.q4_err), t->accum.q4_err.max,
		membrane_err_stats_mean(&t->accum.q8_err), t->accum.q8_err.max,
		membrane_runtime_weighted_mean_rel_l2_error(t));
	fprintf(f, "\"overhead\":{\"membrane_processing_seconds\":%.6f,"
		"\"mean_seconds_per_block\":%.9f,\"inference_wall_seconds\":%.6f,"
		"\"overhead_ratio_pct\":%.4f}}\n",
		t->membrane_seconds,
		t->total_blocks > 0 ? t->membrane_seconds / (double)t->total_blocks
			: 0.0,
		t->inference_seconds, overhead_pct);
}
