#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/policy.h"

# define POLICY_MAGIC32		0x314c4f4dU				/* "MOL1" */
# define POLICY_FORMAT_VERSION	1U
# define OFF_MAGIC			0
# define OFF_VERSION		4
# define OFF_MODEL_HASH		8
# define OFF_LLAMA_COMMIT	40
# define OFF_LAYER_COUNT	80
# define OFF_MODEL_NAME		84
# define OFF_TIER_NAME		148
# define OFF_COSINE_MIN		180
# define OFF_TOP1_MIN		188
# define OFF_TOP5_MIN		196
# define OFF_COSINE_MARGIN	204
# define OFF_TOP1_MARGIN	212
# define OFF_TOP5_MARGIN	220
# define OFF_SEARCH_BUDGET	228
# define OFF_EVALS_USED		232
# define OFF_CREATED_TIME	236
# define HEADER_SIZE		244U
# define LLAMA_COMMIT_LEN	40U
# define MODEL_NAME_CAP		64U
# define TIER_NAME_CAP		32U

typedef struct s_membrane_policy
{
	uint8_t					model_hash[MEMBRANE_SHA256_DIGEST_BYTES];
	char					llama_commit[LLAMA_COMMIT_LEN + 1];
	uint32_t				layer_count;
	char					model_name[MODEL_NAME_CAP + 1];
	char					tier_name[TIER_NAME_CAP + 1];
	double					cosine_min;
	double					top1_min;
	double					top5_min;
	double					cosine_margin;
	double					top1_margin;
	double					top5_margin;
	uint32_t				search_budget;
	uint32_t				evals_used;
	uint64_t				created_unix_time;
	membrane_precision_t	*k_prec;
	membrane_precision_t	*v_prec;
}	s_membrane_policy_t;

static void	put_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static void	put_le64(uint8_t *p, uint64_t v)
{
	put_le32(p, (uint32_t)v);
	put_le32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t	get_le32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static uint64_t	get_le64(const uint8_t *p)
{
	return ((uint64_t)get_le32(p) | ((uint64_t)get_le32(p + 4) << 32));
}

static void	put_f64(uint8_t *p, double v)
{
	uint64_t	bits;

	memcpy(&bits, &v, sizeof(bits));
	put_le64(p, bits);
}

static double	get_f64(const uint8_t *p)
{
	uint64_t	bits;
	double		v;

	bits = get_le64(p);
	memcpy(&v, &bits, sizeof(v));
	return (v);
}

static int	is_valid_precision(uint32_t v)
{
	return (v == MEMBRANE_PRECISION_FP16 || v == MEMBRANE_PRECISION_Q8
		|| v == MEMBRANE_PRECISION_Q4);
}

static membrane_status_t	validate_build_args(const membrane_policy_build_t *b)
{
	uint32_t	i;

	if (b == NULL || b->k_prec == NULL || b->v_prec == NULL
			|| b->layer_count == 0 || b->llama_cpp_commit == NULL
			|| b->model_name == NULL || b->tier_name == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (strlen(b->llama_cpp_commit) != LLAMA_COMMIT_LEN
			|| strlen(b->model_name) > MODEL_NAME_CAP
			|| strlen(b->tier_name) > TIER_NAME_CAP)
		return (MEMBRANE_ERR_INVALID_ARG);
	i = 0;
	while (i < b->layer_count)
	{
		if (!is_valid_precision((uint32_t)b->k_prec[i])
				|| !is_valid_precision((uint32_t)b->v_prec[i]))
			return (MEMBRANE_ERR_INVALID_ARG);
		i++;
	}
	return (MEMBRANE_OK);
}

static void	fill_header(uint8_t *h, const membrane_policy_build_t *b)
{
	put_le32(h + OFF_MAGIC, POLICY_MAGIC32);
	put_le32(h + OFF_VERSION, POLICY_FORMAT_VERSION);
	memcpy(h + OFF_MODEL_HASH, b->model_sha256, MEMBRANE_SHA256_DIGEST_BYTES);
	memcpy(h + OFF_LLAMA_COMMIT, b->llama_cpp_commit, LLAMA_COMMIT_LEN);
	put_le32(h + OFF_LAYER_COUNT, b->layer_count);
	memset(h + OFF_MODEL_NAME, 0, MODEL_NAME_CAP);
	memcpy(h + OFF_MODEL_NAME, b->model_name, strlen(b->model_name));
	memset(h + OFF_TIER_NAME, 0, TIER_NAME_CAP);
	memcpy(h + OFF_TIER_NAME, b->tier_name, strlen(b->tier_name));
	put_f64(h + OFF_COSINE_MIN, b->cosine_min);
	put_f64(h + OFF_TOP1_MIN, b->top1_min);
	put_f64(h + OFF_TOP5_MIN, b->top5_min);
	put_f64(h + OFF_COSINE_MARGIN, b->cosine_margin);
	put_f64(h + OFF_TOP1_MARGIN, b->top1_margin);
	put_f64(h + OFF_TOP5_MARGIN, b->top5_margin);
	put_le32(h + OFF_SEARCH_BUDGET, b->search_budget);
	put_le32(h + OFF_EVALS_USED, b->evals_used);
	put_le64(h + OFF_CREATED_TIME, b->created_unix_time);
}

membrane_status_t	membrane_policy_save(const char *path,
						const membrane_policy_build_t *b)
{
	membrane_status_t	st;
	uint8_t				*buf;
	size_t				body_size;
	size_t				total_size;
	uint32_t			i;
	uint32_t			checksum;
	FILE				*f;

	st = validate_build_args(b);
	if (st != MEMBRANE_OK)
		return (st);
	if (path == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	body_size = HEADER_SIZE + (size_t)b->layer_count * 2;
	total_size = body_size + 4;
	buf = malloc(total_size);
	if (buf == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	fill_header(buf, b);
	i = 0;
	while (i < b->layer_count)
	{
		buf[HEADER_SIZE + i * 2] = (uint8_t)b->k_prec[i];
		buf[HEADER_SIZE + i * 2 + 1] = (uint8_t)b->v_prec[i];
		i++;
	}
	checksum = membrane_block_checksum(buf, body_size);
	put_le32(buf + body_size, checksum);
	f = fopen(path, "wb");
	if (f == NULL)
		return (free(buf), MEMBRANE_ERR_IO);
	st = (fwrite(buf, 1, total_size, f) == total_size)
		? MEMBRANE_OK : MEMBRANE_ERR_IO;
	fclose(f);
	free(buf);
	return (st);
}

static long	file_size_of(FILE *f)
{
	long	sz;

	if (fseek(f, 0, SEEK_END) != 0)
		return (-1);
	sz = ftell(f);
	if (fseek(f, 0, SEEK_SET) != 0)
		return (-1);
	return (sz);
}

static membrane_status_t	parse_header(const uint8_t *h,
								s_membrane_policy_t *p)
{
	uint32_t	i;

	if (get_le32(h + OFF_MAGIC) != POLICY_MAGIC32)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (get_le32(h + OFF_VERSION) != POLICY_FORMAT_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	memcpy(p->model_hash, h + OFF_MODEL_HASH, MEMBRANE_SHA256_DIGEST_BYTES);
	memcpy(p->llama_commit, h + OFF_LLAMA_COMMIT, LLAMA_COMMIT_LEN);
	p->llama_commit[LLAMA_COMMIT_LEN] = '\0';
	p->layer_count = get_le32(h + OFF_LAYER_COUNT);
	if (p->layer_count == 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	memcpy(p->model_name, h + OFF_MODEL_NAME, MODEL_NAME_CAP);
	p->model_name[MODEL_NAME_CAP] = '\0';
	memcpy(p->tier_name, h + OFF_TIER_NAME, TIER_NAME_CAP);
	p->tier_name[TIER_NAME_CAP] = '\0';
	i = 0;
	while (i < MODEL_NAME_CAP && p->model_name[i] != '\0')
		i++;
	if (i == MODEL_NAME_CAP)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	i = 0;
	while (i < TIER_NAME_CAP && p->tier_name[i] != '\0')
		i++;
	if (i == TIER_NAME_CAP)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	p->cosine_min = get_f64(h + OFF_COSINE_MIN);
	p->top1_min = get_f64(h + OFF_TOP1_MIN);
	p->top5_min = get_f64(h + OFF_TOP5_MIN);
	p->cosine_margin = get_f64(h + OFF_COSINE_MARGIN);
	p->top1_margin = get_f64(h + OFF_TOP1_MARGIN);
	p->top5_margin = get_f64(h + OFF_TOP5_MARGIN);
	p->search_budget = get_le32(h + OFF_SEARCH_BUDGET);
	p->evals_used = get_le32(h + OFF_EVALS_USED);
	p->created_unix_time = get_le64(h + OFF_CREATED_TIME);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_policy_load(const char *path,
						membrane_policy_t **out)
{
	FILE				*f;
	long				sz;
	size_t				body_size;
	size_t				total_size;
	uint8_t				*buf;
	s_membrane_policy_t	*p;
	membrane_status_t	st;
	uint32_t			i;
	uint32_t			stored_checksum;

	if (out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	*out = NULL;
	if (path == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	f = fopen(path, "rb");
	if (f == NULL)
		return (MEMBRANE_ERR_IO);
	sz = file_size_of(f);
	if (sz < (long)HEADER_SIZE + 4)
		return (fclose(f), MEMBRANE_ERR_CORRUPT_DATA);
	buf = malloc((size_t)sz);
	if (buf == NULL)
		return (fclose(f), MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(buf, 1, (size_t)sz, f) != (size_t)sz)
		return (fclose(f), free(buf), MEMBRANE_ERR_IO);
	fclose(f);
	p = calloc(1, sizeof(*p));
	if (p == NULL)
		return (free(buf), MEMBRANE_ERR_ALLOC_FAILED);
	st = parse_header(buf, p);
	if (st != MEMBRANE_OK)
		return (free(buf), free(p), st);
	body_size = HEADER_SIZE + (size_t)p->layer_count * 2;
	total_size = body_size + 4;
	if ((size_t)sz != total_size)
		return (free(buf), free(p), MEMBRANE_ERR_CORRUPT_DATA);
	stored_checksum = get_le32(buf + body_size);
	if (membrane_block_checksum(buf, body_size) != stored_checksum)
		return (free(buf), free(p), MEMBRANE_ERR_CORRUPT_DATA);
	p->k_prec = malloc(sizeof(*p->k_prec) * p->layer_count);
	p->v_prec = malloc(sizeof(*p->v_prec) * p->layer_count);
	if (p->k_prec == NULL || p->v_prec == NULL)
	{
		free(buf);
		free(p->k_prec);
		free(p->v_prec);
		free(p);
		return (MEMBRANE_ERR_ALLOC_FAILED);
	}
	i = 0;
	while (i < p->layer_count)
	{
		if (!is_valid_precision(buf[HEADER_SIZE + i * 2])
				|| !is_valid_precision(buf[HEADER_SIZE + i * 2 + 1]))
		{
			free(buf);
			free(p->k_prec);
			free(p->v_prec);
			free(p);
			return (MEMBRANE_ERR_CORRUPT_DATA);
		}
		p->k_prec[i] = (membrane_precision_t)buf[HEADER_SIZE + i * 2];
		p->v_prec[i] = (membrane_precision_t)buf[HEADER_SIZE + i * 2 + 1];
		i++;
	}
	free(buf);
	*out = (membrane_policy_t *)p;
	return (MEMBRANE_OK);
}

static void	set_reason(char *buf, size_t cap, const char *msg)
{
	if (buf == NULL || cap == 0)
		return ;
	snprintf(buf, cap, "%s", msg);
}

membrane_status_t	membrane_policy_validate(const membrane_policy_t *policy,
						const membrane_policy_context_t *ctx,
						char *reason_buf, size_t reason_cap)
{
	const s_membrane_policy_t	*p;
	char						model_hex[MEMBRANE_SHA256_HEX_LEN + 1];
	char						ctx_hex[MEMBRANE_SHA256_HEX_LEN + 1];

	if (policy == NULL || ctx == NULL || ctx->llama_cpp_commit == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	p = (const s_membrane_policy_t *)policy;
	if (memcmp(p->model_hash, ctx->model_sha256,
			MEMBRANE_SHA256_DIGEST_BYTES) != 0)
	{
		membrane_sha256_to_hex(p->model_hash, model_hex);
		membrane_sha256_to_hex(ctx->model_sha256, ctx_hex);
		if (reason_buf != NULL)
			snprintf(reason_buf, reason_cap,
				"model hash mismatch: policy=%s actual=%s",
				model_hex, ctx_hex);
		return (MEMBRANE_ERR_MISMATCH);
	}
	if (strcmp(p->llama_commit, ctx->llama_cpp_commit) != 0)
	{
		if (reason_buf != NULL)
			snprintf(reason_buf, reason_cap,
				"llama.cpp commit mismatch: policy=%s actual=%s",
				p->llama_commit, ctx->llama_cpp_commit);
		return (MEMBRANE_ERR_MISMATCH);
	}
	if (p->layer_count != ctx->layer_count)
	{
		if (reason_buf != NULL)
			snprintf(reason_buf, reason_cap,
				"layer count mismatch: policy=%u actual=%u",
				p->layer_count, ctx->layer_count);
		return (MEMBRANE_ERR_MISMATCH);
	}
	set_reason(reason_buf, reason_cap, "");
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_policy_query(const membrane_policy_t *policy,
						uint32_t layer, int is_v,
						membrane_precision_t *out)
{
	const s_membrane_policy_t	*p;

	if (policy == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	p = (const s_membrane_policy_t *)policy;
	if (layer >= p->layer_count)
		return (MEMBRANE_ERR_INVALID_ARG);
	*out = is_v ? p->v_prec[layer] : p->k_prec[layer];
	return (MEMBRANE_OK);
}

uint32_t	membrane_policy_layer_count(const membrane_policy_t *policy)
{
	if (policy == NULL)
		return (0);
	return (((const s_membrane_policy_t *)policy)->layer_count);
}

void	membrane_policy_destroy(membrane_policy_t *policy)
{
	s_membrane_policy_t	*p;

	if (policy == NULL)
		return ;
	p = (s_membrane_policy_t *)policy;
	free(p->k_prec);
	free(p->v_prec);
	free(p);
}
