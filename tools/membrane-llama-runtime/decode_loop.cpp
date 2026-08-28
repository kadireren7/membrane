#include "decode_loop.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

std::string	read_file(const char *path)
{
	FILE		*f;
	std::string	s;
	char		buf[4096];
	size_t		n;

	f = fopen(path, "rb");
	if (f == NULL)
		return (s);
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		s.append(buf, n);
	fclose(f);
	return (s);
}

double	seconds_since(const struct timespec *t0)
{
	struct timespec	t1;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	return ((double)(t1.tv_sec - t0->tv_sec)
		+ (double)(t1.tv_nsec - t0->tv_nsec) / 1e9);
}

int	argmax(const float *v, int n)
{
	int	best;
	int	i;

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

/* Times llama_decode() together with this step's MEMBRANE flush (see
 * llama_hook.h's membrane_llama_hook_flush_step) as one span:
 * inference_seconds is therefore the OUTER measurement for this step,
 * and membrane_seconds (accumulated inside both the eval callback's
 * extraction/injection and the flush itself) is always a SUBSET of it,
 * never additive -- "overhead ratio" is the fraction of this step's wall
 * time MEMBRANE actually spent in, not extra time layered on top. Also
 * responsible for telling the hook the absolute token-position window
 * this decode call covers (membrane_llama_hook_set_step_context) --
 * required for INJECT modes' token-range scoping, harmless bookkeeping
 * otherwise. */
bool	timed_decode(llama_context *ctx, llama_batch batch,
				uint64_t abs_pos_start, uint64_t n_tok,
				membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, int debug,
				const char *step_label)
{
	struct timespec	t0;
	int				rc;

	membrane_llama_hook_set_step_context(hook_ctx, abs_pos_start, n_tok);
	membrane_runtime_begin_step(collector);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	rc = llama_decode(ctx, batch);
	membrane_llama_hook_flush_step(hook_ctx);
	membrane_runtime_add_inference_seconds(collector, seconds_since(&t0));
	/* Live-interleaving proof (section 11): printed AFTER this exact
	 * llama_decode() call returns, using the block count MEMBRANE
	 * processed strictly within this same step -- not a post-run
	 * summary. */
	if (debug)
		fprintf(stderr, "%s: llama decode; membrane processed %llu KV "
			"blocks\n", step_label,
			(unsigned long long)membrane_runtime_step_block_count(
				collector));
	membrane_runtime_end_step(collector);
	return (rc == 0);
}

bool	decode_prompt(llama_context *ctx,
				const std::vector<llama_token> &prompt_tokens,
				int32_t n_batch, membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, int debug,
				uint64_t *abs_pos)
{
	size_t	off;
	size_t	chunk;

	off = 0;
	while (off < prompt_tokens.size())
	{
		chunk = prompt_tokens.size() - off;
		if (chunk > (size_t)n_batch)
			chunk = (size_t)n_batch;
		if (!timed_decode(ctx,
				llama_batch_get_one(
					const_cast<llama_token *>(prompt_tokens.data()) + off,
					(int32_t)chunk),
				*abs_pos, (uint64_t)chunk, collector, hook_ctx, debug,
				"prompt decode"))
			return (false);
		*abs_pos += chunk;
		off += chunk;
	}
	return (true);
}

/*
 * Free-running (teacher_force == NULL): argmax's its own logits each
 * step, stops at EOG or gen_tokens steps, whichever first.
 * Teacher-forced (teacher_force != NULL): ignores its own argmax and
 * decodes exactly teacher_force's tokens, in order -- used to replay
 * a reference sequence into a comparison context for an aligned
 * evaluation. Captures the logits available BEFORE each step's token
 * is decoded, iff capture_logits (these are the logits that "chose"/
 * "predict" that token). Appends decoded text to *text_out iff
 * non-NULL. token_cb, iff non-NULL, is called once per free-running
 * step immediately after that token's decode succeeds (streaming) --
 * never for a teacher-forced replay.
 */
void	run_generation(llama_context *ctx, const llama_vocab *vocab,
				int32_t n_vocab, int gen_tokens,
				membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, int debug,
				bool capture_logits,
				const std::vector<int32_t> *teacher_force,
				uint64_t *abs_pos, std::string *text_out,
				gen_run_result_t *out, membrane_token_cb_t token_cb,
				void *token_cb_ud, double *out_first_token_ms)
{
	int			step;
	int			limit;
	const float	*logits;
	llama_token	tok;
	char		step_label[32];
	char		piece[256];
	int			piece_len;
	struct timespec	first_token_t0;
	bool		timing_first_token;

	if (out_first_token_ms != NULL)
		*out_first_token_ms = -1.0;
	out->ok = true;
	limit = teacher_force != NULL ? (int)teacher_force->size() : gen_tokens;
	step = 0;
	while (step < limit)
	{
		/* Phase 24: only the free-running path (never teacher-forced --
		 * that replays a reference sequence, not "generation") and only
		 * step 0 (Section 25: one stage boundary, not per-step timing). */
		timing_first_token = (out_first_token_ms != NULL
			&& teacher_force == NULL && step == 0);
		if (timing_first_token)
			clock_gettime(CLOCK_MONOTONIC, &first_token_t0);
		logits = llama_get_logits_ith(ctx, -1);
		if (logits == NULL)
		{
			fprintf(stderr, "membrane: logits unavailable\n");
			out->ok = false;
			break ;
		}
		if (capture_logits)
			out->logits.emplace_back(logits, logits + n_vocab);
		if (teacher_force != NULL)
			tok = (llama_token)(*teacher_force)[step];
		else
		{
			tok = (llama_token)argmax(logits, n_vocab);
			if (llama_vocab_is_eog(vocab, tok))
				break ;
		}
		out->tokens.push_back((int32_t)tok);
		piece_len = 0;
		if (text_out != NULL || (token_cb != NULL && teacher_force == NULL))
		{
			piece_len = llama_token_to_piece(vocab, tok, piece,
					sizeof(piece), 0, true);
			if (piece_len > 0 && text_out != NULL)
				text_out->append(piece, piece_len);
		}
		if (piece_len > 0 && token_cb != NULL && teacher_force == NULL)
			token_cb(piece, (size_t)piece_len, step, token_cb_ud);
		snprintf(step_label, sizeof(step_label), "gen step %d", step + 1);
		if (!timed_decode(ctx, llama_batch_get_one(&tok, 1), *abs_pos, 1,
				collector, hook_ctx, debug, step_label))
		{
			fprintf(stderr, "membrane: generation decode failed\n");
			out->ok = false;
			break ;
		}
		if (timing_first_token)
			*out_first_token_ms = seconds_since(&first_token_t0) * 1000.0;
		*abs_pos += 1;
		step++;
	}
}

/* Phase 12H: the real llama_context_params.kv_dev_override callback --
 * user_data is the caller's membrane_kv_placement_map_t*. default_dev
 * is exactly the device upstream would already use for this layer's KV
 * (the same device as its weights, since this product path never sets
 * kv_dev_override without also having resolved weight placement first
 * -- Section 21): returning default_dev unchanged for a GPU-resident
 * layer is a genuine no-op, never a redundant re-placement. An
 * out-of-range il (should not happen: the map is sized to the real
 * model's own n_layer) falls back to default_dev, matching the
 * patch's own documented "NULL return -> default_dev" contract. A
 * non-NULL map with a NULL layer_on_gpu (should not happen from any
 * real call site either -- membrane_kv_placement_map_t's only
 * producer always sets both together) also falls back to default_dev
 * rather than dereferencing a null pointer -- not `static` so
 * test_decode_loop.cpp can drive this fallback directly. */
ggml_backend_dev_t	kv_placement_dev_override_cb(int32_t il,
				ggml_backend_dev_t default_dev, void *user_data)
{
	const membrane_kv_placement_map_t	*m;

	m = (const membrane_kv_placement_map_t *)user_data;
	if (m == NULL || m->layer_on_gpu == NULL || il < 0
		|| il >= m->n_layer || m->layer_on_gpu[il])
		return (default_dev);
	return (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
}

bool	run_kv_store_pass(llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				int gen_tokens, int kv_store_mode, uint32_t ctx_size,
				int debug, const std::vector<int32_t> *teacher_force,
				bool capture_logits, int32_t n_vocab_for_scratch,
				std::string *text_out, membrane_kv_store_telemetry_t *tel,
				gen_run_result_t *out, membrane_token_cb_t token_cb,
				void *token_cb_ud, const membrane_kv_placement_map_t *kv_placement,
				int *out_failure_stage)
{
	llama_context				*ctx;
	llama_context_params		cp;
	const llama_vocab			*vocab;
	int32_t						n_vocab;
	uint64_t					abs_pos;
	membrane_runtime_collector_t	*collector;
	struct timespec				t0;
	double						prompt_seconds;
	double						gen_seconds;
	uint64_t					scratch_bytes;

	if (out_failure_stage != NULL)
		*out_failure_stage = MEMBRANE_KV_PASS_STAGE_NONE;
	vocab = llama_model_get_vocab(model);
	n_vocab = llama_vocab_n_tokens(vocab);
	cp = llama_context_default_params();
	cp.n_ctx = ctx_size;
	cp.n_batch = 256;
	cp.n_threads = 4;
	cp.n_threads_batch = 4;
	cp.no_perf = true;
	/* Required by upstream llama.cpp for q8: quantized V cache is a hard
	 * error (llama_init_from_model returns NULL) unless flash attention
	 * is enabled -- see llama-context.cpp's "V cache quantization
	 * requires flash_attn" check. Forced ENABLED (not AUTO) for BOTH
	 * modes, unconditionally -- not just q8 -- so a native-vs-q8
	 * comparison never mixes "KV cache dtype" with "which attention
	 * kernel ran" as a second, uncontrolled variable; AUTO's runtime
	 * resolution is not relied on for either mode. */
	cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
	if (kv_store_mode == MEMBRANE_KV_STORE_Q8)
	{
		cp.type_k = GGML_TYPE_Q8_0;
		cp.type_v = GGML_TYPE_Q8_0;
	}
	else if (kv_store_mode == MEMBRANE_KV_STORE_Q5)
	{
		/* Phase 10C: GGML_TYPE_Q5_1 specifically -- the product "q5"
		 * mode always maps to Q5_1, never Q5_0, per the Phase 10B
		 * evaluation (Q5_1 was consistently the stronger of the two
		 * on quality, at a modest memory cost). Same "requires
		 * flash_attn" upstream rule as Q8 above. */
		cp.type_k = GGML_TYPE_Q5_1;
		cp.type_v = GGML_TYPE_Q5_1;
	}
	/* Phase 12H: kv_dev_override left NULL (byte-identical to every
	 * prior phase) unless a caller explicitly passed a placement map --
	 * every membrane-llama-run call site, and membrane-run's own calls
	 * with --kv-placement default, leave kv_placement NULL here. */
	if (kv_placement != NULL)
	{
		cp.kv_dev_override = kv_placement_dev_override_cb;
		cp.kv_dev_override_ud = (void *)kv_placement;
	}
	collector = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_BASELINE, MEMBRANE_RUNTIME_MAX_LAYERS);
	if (collector == NULL)
		return (out->ok = false, false);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	ctx = llama_init_from_model(model, cp);
	/* Phase 24: recorded regardless of success/failure below -- a
	 * FAILED construction attempt's own duration is real evidence too
	 * (e.g. Section 6/16's fallback-cost interest), never silently
	 * dropped just because this specific attempt didn't succeed. */
	tel->context_create_ms = seconds_since(&t0) * 1000.0;
	if (ctx == NULL)
	{
		/* Review fix: this used to unconditionally say "for q8" even
		 * when kv_store_mode was native or q5 -- name the actual
		 * requested precision instead of a hardcoded, sometimes-wrong
		 * one. */
		fprintf(stderr, "membrane: kv-store context creation failed (see "
			"llama's own stderr diagnostics above -- for %s, this fails "
			"closed rather than silently falling back to native "
			"storage)\n",
			kv_store_mode == MEMBRANE_KV_STORE_Q8 ? "q8"
				: kv_store_mode == MEMBRANE_KV_STORE_Q5 ? "q5" : "native");
		membrane_runtime_collector_destroy(collector);
		if (out_failure_stage != NULL)
			*out_failure_stage = MEMBRANE_KV_PASS_STAGE_CONTEXT_CREATE;
		return (out->ok = false, false);
	}
	membrane_kv_store_read_rss(&tel->rss_after_context);
	abs_pos = 0;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (!decode_prompt(ctx, prompt_tokens, cp.n_batch, collector, NULL,
			debug, &abs_pos))
	{
		fprintf(stderr, "membrane: kv-store prompt decode failed\n");
		membrane_runtime_collector_destroy(collector);
		if (out_failure_stage != NULL)
			*out_failure_stage = MEMBRANE_KV_PASS_STAGE_DECODE;
		return (llama_free(ctx), out->ok = false, false);
	}
	prompt_seconds = seconds_since(&t0);
	membrane_kv_store_read_rss(&tel->rss_after_prompt);
	out->ok = true;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	run_generation(ctx, vocab, n_vocab, gen_tokens, collector, NULL, debug,
		capture_logits, teacher_force, &abs_pos, text_out, out, token_cb,
		token_cb_ud, &tel->first_token_ms);
	gen_seconds = seconds_since(&t0);
	membrane_kv_store_read_rss(&tel->rss_final);
	tel->prompt_ms = prompt_seconds * 1000.0;
	tel->generation_ms = gen_seconds * 1000.0;
	tel->prompt_tok_per_s = prompt_seconds > 0.0
		? (double)prompt_tokens.size() / prompt_seconds : 0.0;
	tel->generation_tok_per_s = (gen_seconds > 0.0 && !out->tokens.empty())
		? (double)out->tokens.size() / gen_seconds : 0.0;
	/* Only MEMBRANE-owned allocation on this path: run_generation's
	 * per-step logit capture (gen_run_result_t::logits), when the
	 * caller actually asked for it -- never held for a normal/canonical
	 * pass (that doesn't need its own logits, only a comparison pass
	 * does), so a plain q8 run reports this as genuinely 0, not a
	 * placeholder. */
	if (capture_logits)
	{
		scratch_bytes = (uint64_t)out->logits.size()
			* (uint64_t)n_vocab_for_scratch * sizeof(float);
		if (scratch_bytes > tel->scratch_peak_bytes)
			tel->scratch_peak_bytes = scratch_bytes;
	}
	membrane_runtime_collector_destroy(collector);
	llama_free(ctx);
	if (!out->ok && out_failure_stage != NULL)
		*out_failure_stage = MEMBRANE_KV_PASS_STAGE_DECODE;
	return (out->ok);
}

bool	record_kv_store_behavior(const gen_run_result_t &a,
				const gen_run_result_t &c, int32_t n_vocab,
				membrane_kv_store_telemetry_t *tel)
{
	membrane_runtime_behavior_accum_t		*acc;
	membrane_runtime_behavior_summary_t	summary;
	membrane_runtime_step_logit_diff_t		diff;
	size_t									n;
	size_t									i;
	double									nll_a;
	double									nll_c;

	n = std::min(a.logits.size(), c.logits.size());
	n = std::min(n, a.tokens.size());
	acc = membrane_runtime_behavior_create();
	if (acc == NULL)
		return (false);
	i = 0;
	while (i < n)
	{
		membrane_runtime_compare_logits(a.logits[i].data(),
			c.logits[i].data(), n_vocab, 5, &diff);
		nll_a = membrane_runtime_nll(a.logits[i].data(), n_vocab,
				a.tokens[i]);
		nll_c = membrane_runtime_nll(c.logits[i].data(), n_vocab,
				a.tokens[i]);
		membrane_runtime_behavior_add_step(acc, &diff, nll_a, nll_c);
		i++;
	}
	membrane_runtime_behavior_finalize(acc, &summary);
	membrane_runtime_behavior_destroy(acc);
	tel->logit_rel_l2 = summary.logit_rel_l2_mean;
	tel->top1_preservation = summary.top1_preservation_rate;
	tel->delta_nll = summary.delta_nll;
	return (true);
}
