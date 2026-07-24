#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/f16convert.h"
#include "membrane/kvquant.h"

typedef struct s_err_acc
{
	double		sum_sq;
	double		sum_abs;
	double		max_abs;
	double		sum_a_sq;
	double		sum_b_sq;
	double		dot;
	uint64_t	finite_n;
}	err_acc_t;

static void	accumulate_error(err_acc_t *acc, float a, float b)
{
	double	diff;

	if (!isfinite(a))
		return ;
	diff = (double)b - (double)a;
	acc->sum_sq += diff * diff;
	acc->sum_abs += fabs(diff);
	if (fabs(diff) > acc->max_abs)
		acc->max_abs = fabs(diff);
	acc->sum_a_sq += (double)a * (double)a;
	acc->sum_b_sq += (double)b * (double)b;
	acc->dot += (double)a * (double)b;
	acc->finite_n += 1;
}

static double	cosine_of(const err_acc_t *acc)
{
	double	norm_a;
	double	norm_b;

	norm_a = sqrt(acc->sum_a_sq);
	norm_b = sqrt(acc->sum_b_sq);
	if (norm_a == 0.0 && norm_b == 0.0)
		return (1.0);
	if (norm_a == 0.0 || norm_b == 0.0)
		return (0.0);
	return (acc->dot / (norm_a * norm_b));
}

static double	rel_l2_of(const err_acc_t *acc)
{
	if (acc->sum_a_sq == 0.0)
		return ((acc->sum_sq == 0.0) ? 0.0 : 1.0);
	return (sqrt(acc->sum_sq) / sqrt(acc->sum_a_sq));
}

static void	compare_payloads(const uint8_t *orig, const uint8_t *dec,
				size_t elements, membrane_kv_quant_metrics_t *out)
{
	err_acc_t	acc;
	size_t		i;
	float		a;
	float		b;

	memset(&acc, 0, sizeof(acc));
	i = 0;
	while (i < elements)
	{
		a = membrane_f16_to_f32(((const uint16_t *)(const void *)orig)[i]);
		b = membrane_f16_to_f32(((const uint16_t *)(const void *)dec)[i]);
		accumulate_error(&acc, a, b);
		i++;
	}
	if (acc.finite_n > 0)
	{
		out->mse = acc.sum_sq / (double)acc.finite_n;
		out->rmse = sqrt(out->mse);
		out->mae = acc.sum_abs / (double)acc.finite_n;
	}
	out->max_abs_err = acc.max_abs;
	out->cosine_similarity = cosine_of(&acc);
	out->rel_l2_error = rel_l2_of(&acc);
}

membrane_status_t	membrane_kv_quant_compute(const uint8_t *payload,
						size_t len, const membrane_q8_cfg_t *cfg,
						membrane_kv_quant_metrics_t *out)
{
	uint8_t				*enc;
	uint8_t				*dec;
	size_t				bound;
	size_t				enc_len;
	size_t				dec_len;
	membrane_q8_stats_t	stats;
	membrane_status_t	st;

	if (out == NULL || (payload == NULL && len > 0) || len % 2 != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	bound = membrane_q8_bound(len, cfg);
	if (bound == SIZE_MAX)
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(out, 0, sizeof(*out));
	out->raw_bytes = len;
	if (len == 0)
	{
		out->decode_ok = 1;
		return (MEMBRANE_OK);
	}
	enc = malloc(bound);
	dec = malloc(len);
	if (enc == NULL || dec == NULL)
		return (free(enc), free(dec), MEMBRANE_ERR_ALLOC_FAILED);
	st = membrane_q8_encode(cfg, payload, len, enc, bound, &enc_len, &stats);
	if (st != MEMBRANE_OK)
		return (free(enc), free(dec), st);
	out->encoded_bytes = enc_len;
	out->elements = stats.elements;
	out->groups = stats.groups;
	out->saturated = stats.saturated;
	out->nan_input = stats.nan_input;
	out->inf_input = stats.inf_input;
	out->degenerate_groups = stats.degenerate_groups;
	out->metadata_bytes = enc_len - stats.elements;
	out->decode_ok = (membrane_q8_decode(enc, enc_len, dec, len, &dec_len)
			== MEMBRANE_OK && dec_len == len);
	if (out->decode_ok)
		compare_payloads(payload, dec, stats.elements, out);
	free(enc);
	free(dec);
	return (MEMBRANE_OK);
}
