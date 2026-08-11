#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/quant_select.h"
#include "runtime_core.h"

/* ------------------------------------------------------------------ */
/* Mode                                                                 */
/* ------------------------------------------------------------------ */

const char	*membrane_runtime_mode_name(membrane_runtime_mode_t m)
{
	if (m == MEMBRANE_RUNTIME_MODE_BASELINE)
		return ("baseline");
	if (m == MEMBRANE_RUNTIME_MODE_SHADOW_Q8)
		return ("shadow-q8");
	if (m == MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE)
		return ("shadow-adaptive");
	if (m == MEMBRANE_RUNTIME_MODE_INJECT_Q8)
		return ("inject-q8");
	if (m == MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE)
		return ("inject-adaptive");
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
	if (strcmp(name, "inject-q8") == 0)
		return (*out = MEMBRANE_RUNTIME_MODE_INJECT_Q8, 1);
	if (strcmp(name, "inject-adaptive") == 0)
		return (*out = MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE, 1);
	return (0);
}

int	membrane_runtime_mode_is_inject(membrane_runtime_mode_t m)
{
	return (m == MEMBRANE_RUNTIME_MODE_INJECT_Q8
		|| m == MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE);
}

int	membrane_runtime_mode_is_shadow(membrane_runtime_mode_t m)
{
	return (m == MEMBRANE_RUNTIME_MODE_SHADOW_Q8
		|| m == MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE);
}

/* ------------------------------------------------------------------ */
/* Injection scope filter                                              */
/* ------------------------------------------------------------------ */

void	membrane_runtime_scope_init(membrane_runtime_scope_t *s)
{
	memset(s, 0, sizeof(*s));
	s->tensor_filter = MEMBRANE_RUNTIME_TENSOR_BOTH;
}

void	membrane_runtime_scope_add_layer(membrane_runtime_scope_t *s,
			uint32_t layer)
{
	s->has_layer_filter = 1;
	if (layer < MEMBRANE_RUNTIME_MAX_LAYERS)
		s->layer_eligible[layer] = 1;
}

void	membrane_runtime_scope_set_tensor(membrane_runtime_scope_t *s,
			int tensor_filter)
{
	s->tensor_filter = tensor_filter;
}

void	membrane_runtime_scope_set_token_range(membrane_runtime_scope_t *s,
			uint64_t start, uint64_t end)
{
	s->has_token_range = 1;
	s->token_start = start;
	s->token_end = end;
}

int	membrane_runtime_scope_matches_layer_tensor(
		const membrane_runtime_scope_t *s, uint32_t layer, int is_v)
{
	if (s->tensor_filter != MEMBRANE_RUNTIME_TENSOR_BOTH
		&& s->tensor_filter != (is_v ? MEMBRANE_RUNTIME_TENSOR_V
			: MEMBRANE_RUNTIME_TENSOR_K))
		return (0);
	if (!s->has_layer_filter)
		return (1);
	if (layer >= MEMBRANE_RUNTIME_MAX_LAYERS)
		return (0);
	return (s->layer_eligible[layer]);
}

int	membrane_runtime_scope_matches_token(const membrane_runtime_scope_t *s,
		uint64_t abs_token_pos)
{
	if (!s->has_token_range)
		return (1);
	return (abs_token_pos >= s->token_start && abs_token_pos <= s->token_end);
}

/* ------------------------------------------------------------------ */
/* Injection reconstruction                                            */
/* ------------------------------------------------------------------ */

membrane_status_t	membrane_runtime_inject_reconstruct(
						membrane_simd_backend_t backend,
						membrane_bench_policy_t policy,
						const membrane_runtime_scope_t *scope,
						uint32_t layer, int is_v, uint64_t token_start_abs,
						uint64_t n_tokens, uint64_t elements_per_token,
						uint16_t *values_f16,
						membrane_runtime_inject_result_t *out)
{
	uint64_t			tok;
	uint64_t			full_blocks_per_token;
	uint64_t			tail_per_token;
	uint64_t			b;
	uint16_t			*token_slice;
	uint64_t			decode_failures_before;
	membrane_status_t	st;

	if (scope == NULL || values_f16 == NULL || out == NULL
		|| elements_per_token == 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(out, 0, sizeof(*out));
	out->ok = 1;
	if (!membrane_runtime_scope_matches_layer_tensor(scope, layer, is_v))
		return (MEMBRANE_OK);
	full_blocks_per_token = elements_per_token
		/ MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	tail_per_token = elements_per_token % MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	tok = 0;
	while (tok < n_tokens)
	{
		if (!membrane_runtime_scope_matches_token(scope,
				token_start_abs + tok))
		{
			tok++;
			continue ;
		}
		out->tokens_in_scope++;
		out->native_tail_values += tail_per_token;
		token_slice = values_f16 + tok * elements_per_token;
		b = 0;
		while (b < full_blocks_per_token)
		{
			out->eligible_blocks++;
			decode_failures_before = out->accum.decode_failures;
			st = membrane_bench_process_block_reconstruct(backend, policy,
					token_slice + b * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
					MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
					MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR,
					&out->accum,
					token_slice + b * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK);
			if (st != MEMBRANE_OK
				|| out->accum.decode_failures != decode_failures_before)
			{
				out->ok = 0;
				out->failed_blocks = 1;
				return (MEMBRANE_OK);
			}
			out->injected_blocks++;
			b++;
		}
		tok++;
	}
	return (MEMBRANE_OK);
}

/* ------------------------------------------------------------------ */
/* Behavior comparison                                                  */
/* ------------------------------------------------------------------ */

void	membrane_runtime_detect_divergence(const int32_t *a, size_t na,
			const int32_t *b, size_t nb, membrane_runtime_divergence_t *out)
{
	size_t	n_common;
	size_t	i;

	memset(out, 0, sizeof(*out));
	out->first_divergence_step = -1;
	n_common = na < nb ? na : nb;
	i = 0;
	while (i < n_common)
	{
		if (a[i] != b[i])
		{
			out->divergent_positions++;
			if (out->first_divergence_step < 0)
				out->first_divergence_step = (int64_t)i;
		}
		i++;
	}
	if (na != nb && out->first_divergence_step < 0)
		out->first_divergence_step = (int64_t)n_common;
	out->identical = (na == nb && out->divergent_positions == 0);
}

# define MEMBRANE_RUNTIME_TOPK_MAX	64

static int	argmax_f(const float *v, int32_t n)
{
	int32_t	best;
	int32_t	i;

	best = 0;
	i = 1;
	while (i < n)
	{
		if (v[i] > v[best])
			best = i;
		i++;
	}
	return (best);
}

/* Fills idx[0..k) with the indices of the k largest values in v (ties
 * broken by lowest index first), via a simple O(n*k) partial
 * selection -- k is small (<= MEMBRANE_RUNTIME_TOPK_MAX) so this is
 * fine, and it avoids allocating/sorting a full copy of v. */
static void	topk_indices(const float *v, int32_t n, int k, int32_t *idx)
{
	int		taken[MEMBRANE_RUNTIME_TOPK_MAX];
	int		t;
	int		j;
	int32_t	best;
	int32_t	i;
	int		best_is_taken;

	t = 0;
	while (t < k)
	{
		best = -1;
		i = 0;
		while (i < n)
		{
			best_is_taken = 0;
			j = 0;
			while (j < t)
			{
				if (taken[j] == i)
					best_is_taken = 1;
				j++;
			}
			if (!best_is_taken && (best < 0 || v[i] > v[best]))
				best = i;
			i++;
		}
		taken[t] = (int)best;
		idx[t] = best;
		t++;
	}
}

membrane_status_t	membrane_runtime_compare_logits(const float *a,
						const float *b, int32_t n_vocab, int k,
						membrane_runtime_step_logit_diff_t *out)
{
	double	sum_abs;
	double	sum_sq_diff;
	double	sum_sq_a;
	double	d;
	int32_t	i;
	int32_t	top_a[MEMBRANE_RUNTIME_TOPK_MAX];
	int32_t	top_b[MEMBRANE_RUNTIME_TOPK_MAX];
	int		overlap;
	int		ii;
	int		jj;

	if (a == NULL || b == NULL || out == NULL || n_vocab <= 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(out, 0, sizeof(*out));
	sum_abs = 0.0;
	sum_sq_diff = 0.0;
	sum_sq_a = 0.0;
	i = 0;
	while (i < n_vocab)
	{
		d = (double)b[i] - (double)a[i];
		sum_abs += fabs(d);
		sum_sq_diff += d * d;
		sum_sq_a += (double)a[i] * (double)a[i];
		if (fabs(d) > out->abs_diff_max)
			out->abs_diff_max = fabs(d);
		i++;
	}
	out->abs_diff_mean = sum_abs / (double)n_vocab;
	if (sum_sq_a > 0.0)
		out->rel_l2 = sqrt(sum_sq_diff / sum_sq_a);
	else
		out->rel_l2 = sqrt(sum_sq_diff) > 0.0 ? 1.0 : 0.0;
	out->top1_preserved = (argmax_f(a, n_vocab) == argmax_f(b, n_vocab));
	if (k <= 0)
	{
		out->topk_overlap = -1;
		return (MEMBRANE_OK);
	}
	if (k > MEMBRANE_RUNTIME_TOPK_MAX)
		k = MEMBRANE_RUNTIME_TOPK_MAX;
	if (k > n_vocab)
		k = n_vocab;
	topk_indices(a, n_vocab, k, top_a);
	topk_indices(b, n_vocab, k, top_b);
	overlap = 0;
	ii = 0;
	while (ii < k)
	{
		jj = 0;
		while (jj < k)
		{
			if (top_a[ii] == top_b[jj])
			{
				overlap++;
				break ;
			}
			jj++;
		}
		ii++;
	}
	out->topk_overlap = overlap;
	return (MEMBRANE_OK);
}

double	membrane_runtime_nll(const float *logits, int32_t n_vocab,
			int32_t target_token)
{
	double	max_logit;
	double	sum_exp;
	int32_t	i;

	max_logit = logits[0];
	i = 1;
	while (i < n_vocab)
	{
		if (logits[i] > max_logit)
			max_logit = logits[i];
		i++;
	}
	sum_exp = 0.0;
	i = 0;
	while (i < n_vocab)
	{
		sum_exp += exp((double)logits[i] - max_logit);
		i++;
	}
	return (-(((double)logits[target_token] - max_logit) - log(sum_exp)));
}

typedef struct s_membrane_runtime_behavior_accum
{
	uint64_t	steps;
	double		abs_diff_mean_sum;
	double		abs_diff_max;
	double		rel_l2_sum;
	uint64_t	top1_preserved_count;
	double		topk_overlap_sum;
	uint64_t	topk_overlap_count;
	double		nll_baseline_sum;
	double		nll_injected_sum;
}	membrane_runtime_behavior_accum_t;

membrane_runtime_behavior_accum_t	*membrane_runtime_behavior_create(void)
{
	return (calloc(1, sizeof(membrane_runtime_behavior_accum_t)));
}

void	membrane_runtime_behavior_destroy(
			membrane_runtime_behavior_accum_t *b)
{
	free(b);
}

void	membrane_runtime_behavior_add_step(
			membrane_runtime_behavior_accum_t *b,
			const membrane_runtime_step_logit_diff_t *diff,
			double baseline_nll, double injected_nll)
{
	if (b == NULL || diff == NULL)
		return ;
	b->steps++;
	b->abs_diff_mean_sum += diff->abs_diff_mean;
	if (diff->abs_diff_max > b->abs_diff_max)
		b->abs_diff_max = diff->abs_diff_max;
	b->rel_l2_sum += diff->rel_l2;
	if (diff->top1_preserved)
		b->top1_preserved_count++;
	if (diff->topk_overlap >= 0)
	{
		b->topk_overlap_sum += (double)diff->topk_overlap;
		b->topk_overlap_count++;
	}
	b->nll_baseline_sum += baseline_nll;
	b->nll_injected_sum += injected_nll;
}

void	membrane_runtime_behavior_finalize(
			const membrane_runtime_behavior_accum_t *b,
			membrane_runtime_behavior_summary_t *out)
{
	memset(out, 0, sizeof(*out));
	if (b == NULL || b->steps == 0)
		return ;
	out->steps = b->steps;
	out->logit_abs_diff_mean = b->abs_diff_mean_sum / (double)b->steps;
	out->logit_abs_diff_max = b->abs_diff_max;
	out->logit_rel_l2_mean = b->rel_l2_sum / (double)b->steps;
	out->top1_preservation_rate = (double)b->top1_preserved_count
		/ (double)b->steps;
	if (b->topk_overlap_count > 0)
		out->topk_overlap_mean = b->topk_overlap_sum
			/ (double)b->topk_overlap_count;
	out->mean_nll_baseline = b->nll_baseline_sum / (double)b->steps;
	out->mean_nll_injected = b->nll_injected_sum / (double)b->steps;
	out->delta_nll = out->mean_nll_injected - out->mean_nll_baseline;
}

/* ------------------------------------------------------------------ */
/* Collector                                                           */
/* ------------------------------------------------------------------ */

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

	/* Phase 6 */
	uint8_t						*layer_targeted;
	uint8_t						*layer_injected;
	uint32_t					layers_targeted_count;
	uint32_t					layers_injected_count;
	uint64_t					injection_eligible_blocks;
	uint64_t					injected_blocks;
	uint64_t					failed_blocks;
	uint64_t					injection_native_tail_values;
	int							injection_failed;

	membrane_runtime_divergence_t			divergence;
	int										divergence_set;
	membrane_runtime_behavior_summary_t	behavior;
	int										behavior_set;
}	membrane_runtime_collector_t;

membrane_runtime_collector_t	*membrane_runtime_collector_create(
									membrane_runtime_mode_t mode,
									uint32_t max_layers)
{
	membrane_runtime_collector_t	*c;
	uint32_t						n;

	c = calloc(1, sizeof(*c));
	if (c == NULL)
		return (NULL);
	n = max_layers > 0 ? max_layers : 1;
	c->layer_seen = calloc(n, sizeof(uint8_t));
	c->layer_targeted = calloc(n, sizeof(uint8_t));
	c->layer_injected = calloc(n, sizeof(uint8_t));
	if (c->layer_seen == NULL || c->layer_targeted == NULL
		|| c->layer_injected == NULL)
	{
		free(c->layer_seen);
		free(c->layer_targeted);
		free(c->layer_injected);
		return (free(c), NULL);
	}
	c->mode = mode;
	c->max_layers = max_layers;
	return (c);
}

void	membrane_runtime_collector_destroy(membrane_runtime_collector_t *c)
{
	if (c == NULL)
		return ;
	free(c->layer_seen);
	free(c->layer_targeted);
	free(c->layer_injected);
	free(c);
}

membrane_status_t	membrane_runtime_observe_tensor(
						membrane_runtime_collector_t *c,
						membrane_simd_backend_t backend, int32_t layer,
						int is_v, const uint16_t *values_f16,
						uint64_t n_elems)
{
	uint64_t				full_blocks;
	uint64_t				tail;
	uint64_t				i;
	membrane_bench_policy_t	policy;
	membrane_status_t		st;

	if (c == NULL || (values_f16 == NULL && n_elems > 0)
		|| layer < 0 || (uint32_t)layer >= c->max_layers)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (c->mode == MEMBRANE_RUNTIME_MODE_BASELINE)
		return (MEMBRANE_OK);
	if (c->layer_seen[layer] == 0)
	{
		c->layer_seen[layer] = 1;
		c->layers_seen_count++;
	}
	full_blocks = n_elems / MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	tail = n_elems % MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	c->total_values_observed += n_elems;
	c->tail_values_excluded += tail;
	policy = c->mode == MEMBRANE_RUNTIME_MODE_SHADOW_Q8
		|| c->mode == MEMBRANE_RUNTIME_MODE_INJECT_Q8
		? MEMBRANE_BENCH_POLICY_Q8_ONLY : MEMBRANE_BENCH_POLICY_ADAPTIVE;
	i = 0;
	while (i < full_blocks)
	{
		st = membrane_bench_process_block(backend, policy,
				values_f16 + i * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
				MEMBRANE_RUNTIME_ELEMS_PER_BLOCK,
				MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR,
				&c->accum);
		if (st != MEMBRANE_OK)
			return (st);
		if (is_v)
			c->v_blocks++;
		else
			c->k_blocks++;
		c->step_block_count++;
		i++;
	}
	return (MEMBRANE_OK);
}

void	membrane_runtime_record_injection(membrane_runtime_collector_t *c,
			uint32_t layer, int is_v, const membrane_runtime_inject_result_t *r)
{
	(void)is_v;
	if (c == NULL || r == NULL)
		return ;
	if (r->eligible_blocks == 0 && r->tokens_in_scope == 0)
		return ;
	if (layer < c->max_layers && c->layer_targeted[layer] == 0)
	{
		c->layer_targeted[layer] = 1;
		c->layers_targeted_count++;
	}
	c->injection_eligible_blocks += r->eligible_blocks;
	c->injected_blocks += r->injected_blocks;
	c->failed_blocks += r->failed_blocks;
	c->injection_native_tail_values += r->native_tail_values;
	c->accum.q4_blocks += r->accum.q4_blocks;
	c->accum.q8_blocks += r->accum.q8_blocks;
	c->accum.encoded_bytes += r->accum.encoded_bytes;
	c->accum.blocks_decoded += r->accum.blocks_decoded;
	c->accum.decode_failures += r->accum.decode_failures;
	c->accum.encode_nondeterminism += r->accum.encode_nondeterminism;
	c->accum.q4_err.sum += r->accum.q4_err.sum;
	c->accum.q4_err.count += r->accum.q4_err.count;
	if (r->accum.q4_err.max > c->accum.q4_err.max)
		c->accum.q4_err.max = r->accum.q4_err.max;
	c->accum.q8_err.sum += r->accum.q8_err.sum;
	c->accum.q8_err.count += r->accum.q8_err.count;
	if (r->accum.q8_err.max > c->accum.q8_err.max)
		c->accum.q8_err.max = r->accum.q8_err.max;
	if (r->injected_blocks > 0 && layer < c->max_layers
		&& c->layer_injected[layer] == 0)
	{
		c->layer_injected[layer] = 1;
		c->layers_injected_count++;
	}
	if (!r->ok)
		c->injection_failed = 1;
}

int	membrane_runtime_injection_has_failed(
		const membrane_runtime_collector_t *c)
{
	if (c == NULL)
		return (0);
	return (c->injection_failed);
}

void	membrane_runtime_set_divergence(membrane_runtime_collector_t *c,
			const membrane_runtime_divergence_t *d)
{
	if (c == NULL || d == NULL)
		return ;
	c->divergence = *d;
	c->divergence_set = 1;
}

void	membrane_runtime_set_behavior(membrane_runtime_collector_t *c,
			const membrane_runtime_behavior_summary_t *b)
{
	if (c == NULL || b == NULL)
		return ;
	c->behavior = *b;
	c->behavior_set = 1;
}

void	membrane_runtime_begin_step(membrane_runtime_collector_t *c)
{
	if (c == NULL)
		return ;
	c->inference_steps++;
	c->step_block_count = 0;
}

void	membrane_runtime_end_step(membrane_runtime_collector_t *c)
{
	(void)c;
}

uint64_t	membrane_runtime_step_block_count(
				const membrane_runtime_collector_t *c)
{
	if (c == NULL)
		return (0);
	return (c->step_block_count);
}

void	membrane_runtime_add_inference_seconds(
			membrane_runtime_collector_t *c, double seconds)
{
	if (c == NULL)
		return ;
	c->inference_seconds += seconds;
}

void	membrane_runtime_add_membrane_seconds(
			membrane_runtime_collector_t *c, double seconds)
{
	if (c == NULL)
		return ;
	c->membrane_seconds += seconds;
}

void	membrane_runtime_set_generated_tokens(membrane_runtime_collector_t *c,
			uint64_t n)
{
	if (c == NULL)
		return ;
	c->generated_tokens = n;
}

void	membrane_runtime_finalize(const membrane_runtime_collector_t *c,
			membrane_runtime_telemetry_t *out)
{
	memset(out, 0, sizeof(*out));
	if (c == NULL)
		return ;
	out->mode = c->mode;
	out->generated_tokens = c->generated_tokens;
	out->inference_steps = c->inference_steps;
	out->layers_seen = c->layers_seen_count;
	out->k_blocks = c->k_blocks;
	out->v_blocks = c->v_blocks;
	out->total_blocks = c->k_blocks + c->v_blocks;
	out->total_values_observed = c->total_values_observed;
	out->tail_values_excluded = c->tail_values_excluded;
	out->accum = c->accum;
	out->membrane_seconds = c->membrane_seconds;
	out->inference_seconds = c->inference_seconds;
	out->injection_requested = membrane_runtime_mode_is_inject(c->mode);
	out->injection_succeeded = out->injection_requested
		&& !c->injection_failed;
	out->injection_eligible_blocks = c->injection_eligible_blocks;
	out->injected_blocks = c->injected_blocks;
	out->failed_blocks = c->failed_blocks;
	out->injection_native_tail_values = c->injection_native_tail_values;
	out->layers_targeted = c->layers_targeted_count;
	out->layers_injected = c->layers_injected_count;
	if (c->divergence_set)
		out->divergence = c->divergence;
	else
		out->divergence.first_divergence_step = -1;
	if (c->behavior_set)
	{
		out->behavior = c->behavior;
		out->behavior_available = 1;
	}
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

double	membrane_runtime_injection_coverage_ratio(
			const membrane_runtime_telemetry_t *t)
{
	uint64_t	injected_values;
	uint64_t	denom;

	injected_values = t->injected_blocks * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK;
	denom = injected_values + t->injection_native_tail_values;
	if (denom == 0)
		return (0.0);
	return ((double)injected_values / (double)denom);
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

	fprintf(f, "MEMBRANE live llama.cpp runtime -- mode: %s\n"
		"(SHADOW modes: native llama KV remains authoritative, no "
		"process-memory reduction claim. INJECT modes: reconstructed "
		"values ARE written back into the tensor that feeds the "
		"KV-cache write, but native FP16/F32 cache allocation is "
		"unchanged -- still NOT a process-memory reduction claim.)\n\n",
		membrane_runtime_mode_name(t->mode));
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
	if (t->injection_requested)
	{
		fprintf(f, "Injection\n  requested               yes\n"
			"  succeeded               %s\n"
			"  layers_targeted         %u\n"
			"  layers_injected         %u\n"
			"  eligible_blocks         %llu\n"
			"  injected_blocks         %llu\n"
			"  failed_blocks           %llu\n"
			"  native_tail_values      %llu\n"
			"  coverage_ratio          %.4f\n",
			t->injection_succeeded ? "yes" : "NO -- run failed",
			t->layers_targeted, t->layers_injected,
			(unsigned long long)t->injection_eligible_blocks,
			(unsigned long long)t->injected_blocks,
			(unsigned long long)t->failed_blocks,
			(unsigned long long)t->injection_native_tail_values,
			membrane_runtime_injection_coverage_ratio(t));
		fprintf(f, "Divergence (free-running generation vs. reference)\n"
			"  token_sequence_identical  %s\n"
			"  first_divergence_step     %lld\n"
			"  divergent_positions       %llu\n",
			t->divergence.identical ? "yes" : "no",
			(long long)t->divergence.first_divergence_step,
			(unsigned long long)t->divergence.divergent_positions);
		if (t->behavior_available)
			fprintf(f, "Behavior (aligned/teacher-forced evaluation, "
				"%llu steps)\n"
				"  logit abs diff   mean=%.6f max=%.6f\n"
				"  logit rel-L2     mean=%.6f\n"
				"  top1 preserved   %.2f%%\n"
				"  topk overlap     mean=%.4f\n"
				"  mean NLL         baseline=%.6f injected=%.6f "
				"delta=%.6f\n",
				(unsigned long long)t->behavior.steps,
				t->behavior.logit_abs_diff_mean,
				t->behavior.logit_abs_diff_max,
				t->behavior.logit_rel_l2_mean,
				t->behavior.top1_preservation_rate * 100.0,
				t->behavior.topk_overlap_mean,
				t->behavior.mean_nll_baseline,
				t->behavior.mean_nll_injected, t->behavior.delta_nll);
	}
	fprintf(f, "Storage accounting (encoded payload for observed/"
		"injected KV blocks -- NOT process memory)\n"
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
	fprintf(f, "Runtime overhead (MEMBRANE processing adds overhead by "
		"design; this is expected, not a regression)\n"
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
	fprintf(f, "\"injection\":{\"requested\":%s,\"succeeded\":%s,"
		"\"layers_targeted\":%u,\"layers_injected\":%u,"
		"\"eligible_blocks\":%llu,\"injected_blocks\":%llu,"
		"\"failed_blocks\":%llu,\"native_tail_values\":%llu,"
		"\"coverage_ratio\":%.6f},",
		t->injection_requested ? "true" : "false",
		t->injection_succeeded ? "true" : "false",
		t->layers_targeted, t->layers_injected,
		(unsigned long long)t->injection_eligible_blocks,
		(unsigned long long)t->injected_blocks,
		(unsigned long long)t->failed_blocks,
		(unsigned long long)t->injection_native_tail_values,
		membrane_runtime_injection_coverage_ratio(t));
	fprintf(f, "\"divergence\":{\"identical\":%s,"
		"\"first_divergence_step\":%lld,\"divergent_positions\":%llu},",
		t->divergence.identical ? "true" : "false",
		(long long)t->divergence.first_divergence_step,
		(unsigned long long)t->divergence.divergent_positions);
	fprintf(f, "\"behavior\":{\"available\":%s,\"steps\":%llu,"
		"\"logit_abs_diff_mean\":%.6f,\"logit_abs_diff_max\":%.6f,"
		"\"logit_rel_l2_mean\":%.6f,\"top1_preservation_rate\":%.6f,"
		"\"topk_overlap_mean\":%.6f,\"mean_nll_baseline\":%.6f,"
		"\"mean_nll_injected\":%.6f,\"delta_nll\":%.6f},",
		t->behavior_available ? "true" : "false",
		(unsigned long long)t->behavior.steps,
		t->behavior.logit_abs_diff_mean, t->behavior.logit_abs_diff_max,
		t->behavior.logit_rel_l2_mean, t->behavior.top1_preservation_rate,
		t->behavior.topk_overlap_mean, t->behavior.mean_nll_baseline,
		t->behavior.mean_nll_injected, t->behavior.delta_nll);
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
