#include <algorithm>
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

	/* Phase 6: injection. */
	int									is_inject;
	membrane_bench_policy_t			inject_policy;
	membrane_runtime_scope_t			inject_scope;
	int									debug_perturb_injection;
	uint64_t							step_token_start_abs;
	uint64_t							step_n_tokens;
	std::vector<uint32_t>				k_occurrence_this_step;
};

static membrane_bench_policy_t	policy_for_mode(membrane_runtime_mode_t mode)
{
	if (mode == MEMBRANE_RUNTIME_MODE_INJECT_Q8)
		return (MEMBRANE_BENCH_POLICY_Q8_ONLY);
	return (MEMBRANE_BENCH_POLICY_ADAPTIVE);
}

membrane_llama_hook_ctx_t	*membrane_llama_hook_create(
								membrane_runtime_collector_t *collector,
								membrane_simd_backend_t backend, int debug,
								uint32_t max_layers, membrane_runtime_mode_t mode,
								const membrane_runtime_scope_t *inject_scope,
								int debug_perturb_injection)
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
	ctx->is_inject = membrane_runtime_mode_is_inject(mode);
	ctx->inject_policy = policy_for_mode(mode);
	if (ctx->is_inject && inject_scope != NULL)
		ctx->inject_scope = *inject_scope;
	else
		membrane_runtime_scope_init(&ctx->inject_scope);
	ctx->debug_perturb_injection = ctx->is_inject
		&& debug_perturb_injection;
	ctx->step_token_start_abs = 0;
	ctx->step_n_tokens = 0;
	ctx->k_occurrence_this_step.assign(max_layers, 0);
	return (ctx);
}

void	membrane_llama_hook_destroy(membrane_llama_hook_ctx_t *ctx)
{
	delete ctx;
}

/*
 * Injection targets exactly the MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE-th
 * materialized call for a given layer's "Kcur-%d" this step -- verified
 * (see llama_hook.h/docs/live-runtime.md, and re-verified by directly
 * reading src/models/llama.cpp + src/llama-graph.cpp on the pinned
 * commit) to be the post-RoPE occurrence, for LLM_ARCH_LLAMA on the
 * pinned commit, for a layer with no K bias: `build_qkv()`
 * (llama-graph.cpp) itself tags Kcur-%d twice when `layer.wk_b` is
 * absent -- once right after the raw linear projection, once after the
 * final reshape (both still pre-RoPE) -- and the CALLER
 * (src/models/llama.cpp's build loop) tags it a third time, after
 * applying `ggml_rope_ext`, which is the tensor that actually flows
 * into `build_attn` and the cache write. A layer WITH a K bias would add
 * one more internal tag (four total; the earlier "tagged twice" claim in
 * this project's Phase 5 docs undercounted this). Rather than hardcode a
 * second, possibly-also-wrong guess, this is verified against the
 * ACTUAL model this project validates against (SmolLM2-135M-Instruct,
 * no K bias -- confirmed by the empirical occurrence count matching
 * this trace exactly). If a future model/config/llama.cpp revision
 * tags Kcur-%d a different number of times, the previous step's
 * occurrence count for an in-model layer ends at something other than 0
 * (never touched, always fine -- e.g. layers past this model's real
 * depth) or MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE (correct). Rather
 * than silently never injecting K for that layer or injecting into the
 * wrong occurrence, this is recorded as a hard injection failure --
 * never a silently-successful run whose reported injected_blocks/
 * coverage undercounts, or misattributes, what was actually requested.
 */
# define MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE	3u

static void	check_k_occurrence_sanity(membrane_llama_hook_ctx_t *ctx)
{
	uint32_t							il;
	uint32_t							occ;
	membrane_runtime_inject_result_t	bad;

	il = 0;
	while (il < ctx->max_layers)
	{
		occ = ctx->k_occurrence_this_step[il];
		if (occ != 0 && occ != MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE)
		{
			memset(&bad, 0, sizeof(bad));
			bad.ok = 0;
			bad.eligible_blocks = 1;
			bad.failed_blocks = 1;
			membrane_runtime_record_injection(ctx->collector, il, 0, &bad);
		}
		il++;
	}
}

void	membrane_llama_hook_set_step_context(membrane_llama_hook_ctx_t *ctx,
			uint64_t token_start_abs, uint64_t n_tokens)
{
	if (ctx == NULL)
		return ;
	if (ctx->is_inject)
		check_k_occurrence_sanity(ctx);
	ctx->step_token_start_abs = token_start_abs;
	ctx->step_n_tokens = n_tokens;
	std::fill(ctx->k_occurrence_this_step.begin(),
		ctx->k_occurrence_this_step.end(), 0u);
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

/*
 * DEBUG-ONLY correctness proof, never used in a reported run: corrupts
 * every element `buf` actually changed relative to `before` (i.e. every
 * value the reconstruction pass just touched, never a native/out-of-
 * scope value) so that, IF the caller's write-back is actually consumed
 * by the graph, downstream logits visibly change relative to a normal
 * run. Deliberately crude (large multiplicative inversion) -- the goal
 * is an unmistakable signal attributable specifically to the injected
 * values, not a realistic perturbation and not incidental corruption of
 * values injection was never supposed to touch.
 */
static void	debug_perturb_buffer(std::vector<uint16_t> *buf,
				const std::vector<uint16_t> &before)
{
	size_t	i;
	float	v;

	i = 0;
	while (i < buf->size())
	{
		if ((*buf)[i] != before[i])
		{
			v = ggml_fp16_to_fp32((*buf)[i]);
			(*buf)[i] = ggml_fp32_to_fp16(v * -8.0f - 1.0f);
		}
		i++;
	}
}

/*
 * Reconstructs (under ctx's inject policy/scope) and, on success, writes
 * `f16_buf` back into `t` via ggml_backend_tensor_set -- called only for
 * the tensor occurrence that is authoritative for the actual cache
 * write (see membrane_llama_eval_callback's doc comment). Always folds
 * the result into ctx->collector via membrane_runtime_record_injection,
 * whether or not anything was in scope or anything failed. Never writes
 * back on failure -- `t` keeps its original, genuine data in that case.
 *
 * `orig_f32_buf` is the tensor's original data, BEFORE the F16 downcast
 * every extracted value goes through so precision_policy's F16-native
 * codec can operate on it (non-NULL iff t->type == GGML_TYPE_F32).
 * membrane_runtime_inject_reconstruct() only mutates values inside
 * eligible, in-scope, full blocks -- so comparing `f16_buf` against a
 * pre-reconstruction snapshot tells us exactly which elements it
 * touched. For an F32 tensor, every UNTOUCHED element is written back
 * from `orig_f32_buf` bit-for-bit (never round-tripped through F16, and
 * therefore never losing precision it never needed to lose); only
 * touched elements use the reconstructed value.
 */
static void	inject_into_tensor(membrane_llama_hook_ctx_t *ctx,
				struct ggml_tensor *t, int32_t layer, bool is_v,
				std::vector<uint16_t> *f16_buf, uint64_t n_elems,
				const std::vector<float> *orig_f32_buf)
{
	membrane_runtime_inject_result_t	result;
	uint64_t							elements_per_token;
	std::vector<uint16_t>				f16_before;

	if (ctx->step_n_tokens == 0 || n_elems % ctx->step_n_tokens != 0)
	{
		/* Invariant violation: the caller must always set a valid step
		 * context before decoding. Fail loudly through the normal
		 * telemetry path rather than silently skipping injection. */
		memset(&result, 0, sizeof(result));
		result.ok = 0;
		result.eligible_blocks = 1;
		result.failed_blocks = 1;
		membrane_runtime_record_injection(ctx->collector, (uint32_t)layer,
			is_v ? 1 : 0, &result);
		return ;
	}
	elements_per_token = n_elems / ctx->step_n_tokens;
	f16_before = *f16_buf;
	membrane_runtime_inject_reconstruct(ctx->backend, ctx->inject_policy,
		&ctx->inject_scope, (uint32_t)layer, is_v ? 1 : 0,
		ctx->step_token_start_abs, ctx->step_n_tokens, elements_per_token,
		f16_buf->data(), &result);
	if (result.ok)
	{
		if (ctx->debug_perturb_injection && result.injected_blocks > 0)
			debug_perturb_buffer(f16_buf, f16_before);
		if (t->type == GGML_TYPE_F16)
			ggml_backend_tensor_set(t, f16_buf->data(), 0, ggml_nbytes(t));
		else if (t->type == GGML_TYPE_F32 && orig_f32_buf != NULL)
		{
			std::vector<float>	f32_out(*orig_f32_buf);
			uint64_t			i;

			i = 0;
			while (i < n_elems)
			{
				if ((*f16_buf)[i] != f16_before[i])
					f32_out[i] = ggml_fp16_to_fp32((*f16_buf)[i]);
				i++;
			}
			ggml_backend_tensor_set(t, f32_out.data(), 0, ggml_nbytes(t));
		}
		if (ctx->debug && result.injected_blocks > 0)
			fprintf(stderr, "  [debug-runtime] injected %s (layer=%d): "
				"%llu block(s), native_tail=%llu\n", t->name, layer,
				(unsigned long long)result.injected_blocks,
				(unsigned long long)result.native_tail_values);
	}
	else if (ctx->debug)
		fprintf(stderr, "  [debug-runtime] injection FAILED for %s "
			"(layer=%d) -- native value retained for this tensor, run "
			"will report failure\n", t->name, layer);
	membrane_runtime_record_injection(ctx->collector, (uint32_t)layer,
		is_v ? 1 : 0, &result);
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
	 * for `layer`. SHADOW modes: extract and buffer only --
	 * membrane_llama_hook_flush_step() does the actual quantize/select/
	 * decode/validate work after the whole step's graph has finished
	 * executing, using whichever value was written here LAST (see this
	 * file's header comment on membrane_llama_eval_callback for why that
	 * matters for K specifically). INJECT modes: for K, only the
	 * MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE-th call this step (post-
	 * RoPE, tracked via k_occurrence_this_step -- see check_k_occurrence_
	 * sanity's doc comment for exactly which call site that is and why)
	 * is authoritative -- every earlier call is observed here for timing
	 * symmetry only and never reconstructed/written back, since whatever
	 * we wrote there would just be transformed again downstream (bias-
	 * add, reshape, or RoPE) before reaching the tensor that actually
	 * feeds the cache write. */
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
	std::vector<float>		f32_buf;	/* only populated for F32 tensors --
										 * see inject_into_tensor's doc
										 * comment on why the original F32
										 * values are kept around, not just
										 * their F16 downcast. */
	if (t->type == GGML_TYPE_F16)
		ggml_backend_tensor_get(t, f16_buf.data(), 0, ggml_nbytes(t));
	else if (t->type == GGML_TYPE_F32)
	{
		uint64_t	i;

		f32_buf.resize(n_elems);
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
	if (ctx->is_inject)
	{
		bool	authoritative;

		if (is_v)
			authoritative = true;
		else
		{
			ctx->k_occurrence_this_step[layer]++;
			authoritative = (ctx->k_occurrence_this_step[layer]
				== MEMBRANE_LLAMA_K_AUTHORITATIVE_OCCURRENCE);
		}
		if (authoritative)
			inject_into_tensor(ctx, t, layer, is_v, &f16_buf, n_elems,
				t->type == GGML_TYPE_F32 ? &f32_buf : NULL);
	}
	else if (is_v)
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
	/* INJECT modes never set has_k/has_v (see membrane_llama_eval_
	 * callback -- injection processing happens synchronously, in-graph,
	 * not deferred here), so this loop is naturally a no-op for them. */
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
