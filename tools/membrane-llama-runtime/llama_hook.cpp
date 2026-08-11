#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama_hook.h"

struct s_membrane_llama_hook_ctx
{
	membrane_runtime_collector_t		*collector;
	membrane_simd_backend_t			backend;
	int									debug;
	uint32_t							max_layers;
	std::vector<std::vector<uint16_t>>	pending_k;
	std::vector<std::vector<uint16_t>>	pending_v;
	std::vector<bool>					has_k;
	std::vector<bool>					has_v;
};

membrane_llama_hook_ctx_t	*membrane_llama_hook_create(
								membrane_runtime_collector_t *collector,
								membrane_simd_backend_t backend, int debug,
								uint32_t max_layers)
{
	membrane_llama_hook_ctx_t	*ctx;

	ctx = new (std::nothrow) membrane_llama_hook_ctx_t;
	if (ctx == NULL)
		return (NULL);
	ctx->collector = collector;
	ctx->backend = backend;
	ctx->debug = debug;
	ctx->max_layers = max_layers;
	ctx->pending_k.resize(max_layers);
	ctx->pending_v.resize(max_layers);
	ctx->has_k.assign(max_layers, false);
	ctx->has_v.assign(max_layers, false);
	return (ctx);
}

void	membrane_llama_hook_destroy(membrane_llama_hook_ctx_t *ctx)
{
	delete ctx;
}

/*
 * Parses "Kcur-%d" / "Vcur-%d" exactly (ggml_format_name's own "%s-%d"
 * pattern -- see llama_context::graph_get_cb(), src/llama-context.cpp,
 * and llm_graph_context::build_qkv(), src/llama-graph.cpp, on the
 * pinned commit). Any other name (every other graph node, including
 * "Qcur-%d", attention/norm/matmul intermediates, etc.) does not match
 * and this tensor is left alone.
 */
static bool	parse_kv_tensor_name(const char *name, bool *is_v,
				int32_t *layer)
{
	const char	*p;
	char		*end;
	long		val;

	if (strncmp(name, "Kcur-", 5) == 0)
	{
		*is_v = false;
		p = name + 5;
	}
	else if (strncmp(name, "Vcur-", 5) == 0)
	{
		*is_v = true;
		p = name + 5;
	}
	else
		return (false);
	if (*p == '\0')
		return (false);
	errno = 0;
	val = strtol(p, &end, 10);
	if (errno != 0 || end == p || *end != '\0' || val < 0 || val > INT32_MAX)
		return (false);
	*layer = (int32_t)val;
	return (true);
}

bool	membrane_llama_eval_callback(struct ggml_tensor *t, bool ask,
			void *user_data)
{
	membrane_llama_hook_ctx_t	*ctx;
	bool						is_v;
	int32_t						layer;
	int64_t						n_elems_signed;
	uint64_t					n_elems;
	struct timespec				t0;
	struct timespec				t1;

	ctx = static_cast<membrane_llama_hook_ctx_t *>(user_data);
	if (ctx == NULL || ctx->collector == NULL)
		return (false);
	if (!parse_kv_tensor_name(t->name, &is_v, &layer)
		|| (uint32_t)layer >= ctx->max_layers)
		return (false);
	if (ask)
		return (true);
	/* Materialized call: t->data (via ggml_backend_tensor_get) now
	 * holds this decode step's freshly computed K/V projection values
	 * for `layer`. Extract and buffer only -- membrane_llama_hook_
	 * flush_step() does the actual quantize/select/decode/validate
	 * work after the whole step's graph has finished executing, using
	 * whichever value was written here LAST (see this file's header
	 * comment on membrane_llama_eval_callback for why that matters for
	 * K specifically). */
	n_elems_signed = ggml_nelements(t);
	if (n_elems_signed <= 0)
		return (true);
	if (!ggml_is_contiguous(t))
	{
		if (ctx->debug)
			fprintf(stderr, "membrane-llama-run: skipping non-contiguous "
				"%s (layer=%d) -- not the expected fresh Kcur/Vcur "
				"layout\n", t->name, layer);
		return (true);
	}
	n_elems = (uint64_t)n_elems_signed;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	std::vector<uint16_t>	f16_buf(n_elems);
	if (t->type == GGML_TYPE_F16)
		ggml_backend_tensor_get(t, f16_buf.data(), 0, ggml_nbytes(t));
	else if (t->type == GGML_TYPE_F32)
	{
		std::vector<float>	f32_buf(n_elems);
		uint64_t			i;

		ggml_backend_tensor_get(t, f32_buf.data(), 0, ggml_nbytes(t));
		i = 0;
		while (i < n_elems)
		{
			f16_buf[i] = ggml_fp32_to_fp16(f32_buf[i]);
			i++;
		}
	}
	else
	{
		if (ctx->debug)
			fprintf(stderr, "membrane-llama-run: skipping %s (layer=%d): "
				"unsupported dtype %d\n", t->name, layer, (int)t->type);
		return (true);
	}
	if (is_v)
	{
		ctx->pending_v[layer] = std::move(f16_buf);
		ctx->has_v[layer] = true;
	}
	else
	{
		ctx->pending_k[layer] = std::move(f16_buf);
		ctx->has_k[layer] = true;
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	membrane_runtime_add_membrane_seconds(ctx->collector,
		(double)(t1.tv_sec - t0.tv_sec)
			+ (double)(t1.tv_nsec - t0.tv_nsec) / 1e9);
	if (ctx->debug)
		fprintf(stderr, "  [debug-runtime] extracted %s elems=%llu\n",
			t->name, (unsigned long long)n_elems);
	return (true);
}

void	membrane_llama_hook_flush_step(membrane_llama_hook_ctx_t *ctx)
{
	struct timespec		t0;
	struct timespec		t1;
	uint32_t			il;
	membrane_status_t	st;

	if (ctx == NULL || ctx->collector == NULL)
		return ;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	il = 0;
	while (il < ctx->max_layers)
	{
		if (ctx->has_k[il])
		{
			st = membrane_runtime_observe_tensor(ctx->collector,
					ctx->backend, (int32_t)il, 0, ctx->pending_k[il].data(),
					ctx->pending_k[il].size());
			if (st != MEMBRANE_OK)
				fprintf(stderr, "membrane-llama-run: WARNING: shadow "
					"processing failed for layer=%u K (status=%d) -- "
					"generation continues, this observation is not "
					"reflected in telemetry\n", il, (int)st);
			ctx->has_k[il] = false;
		}
		if (ctx->has_v[il])
		{
			st = membrane_runtime_observe_tensor(ctx->collector,
					ctx->backend, (int32_t)il, 1, ctx->pending_v[il].data(),
					ctx->pending_v[il].size());
			if (st != MEMBRANE_OK)
				fprintf(stderr, "membrane-llama-run: WARNING: shadow "
					"processing failed for layer=%u V (status=%d) -- "
					"generation continues, this observation is not "
					"reflected in telemetry\n", il, (int)st);
			ctx->has_v[il] = false;
		}
		il++;
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	membrane_runtime_add_membrane_seconds(ctx->collector,
		(double)(t1.tv_sec - t0.tv_sec)
			+ (double)(t1.tv_nsec - t0.tv_nsec) / 1e9);
}
