#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "llama_hook.h"
#include "runtime_core.h"
#include "cli_parse.h"
#include "kv_store_telemetry.h"

/*
 * membrane-llama-run: MEMBRANE Product Phase 5/6, live llama.cpp
 * shadow/injection runtime. Orchestration only -- model load, prompt
 * tokenization, decode loops, mode dispatch, and telemetry printing. All
 * llama-facing tensor extraction/write-back lives in llama_hook.cpp; all
 * MEMBRANE-side policy/quantization/telemetry/behavior-comparison logic
 * lives in runtime_core.c.
 *
 * SHADOW modes: native llama KV remains the authoritative store for
 * attention in every mode here. baseline installs no eval callback at
 * all (the true, zero-overhead native path); shadow-q8/shadow-adaptive
 * additionally observe and quantize/dequantize live K/V projection
 * values for measurement, never replacing what llama itself stores or
 * reads.
 *
 * INJECT modes (Phase 6): reconstructed K/V values ARE written back into
 * the tensor that feeds the actual KV-cache write (see llama_hook.h).
 * Native FP16/F32 cache allocation is unchanged either way -- this is
 * never a process-memory reduction claim. An inject-* run performs
 * THREE decode passes over the same prompt/model, one llama_context at a
 * time (never more than one alive simultaneously, to bound memory):
 *   A. pure baseline, free-running (cb_eval never installed) -- produces
 *      the reference token sequence and, for every generation step, the
 *      logits that led to each reference token (captured before that
 *      token is itself decoded).
 *   B. injection, free-running (the model's own greedy choices, under
 *      injection) -- produces the "wild" token sequence used for
 *      free-running divergence against A, and is the run whose telemetry
 *      (injected/failed block counts, coverage, storage/accuracy
 *      accounting) is reported as canonical.
 *   C. injection, teacher-forced on A's reference token sequence -- at
 *      each step, compares C's own logits against A's logits for that
 *      same step (both "predicting" the same next reference token),
 *      isolating the effect of KV perturbation from any autoregressive
 *      cascade a wild run like B would otherwise introduce. Feeds the
 *      aligned behavior summary (logit diff, top1/top-k, NLL/delta-NLL).
 * See docs/live-runtime.md.
 */

static std::string	read_file(const char *path)
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

static double	seconds_since(const struct timespec *t0)
{
	struct timespec	t1;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	return ((double)(t1.tv_sec - t0->tv_sec)
		+ (double)(t1.tv_nsec - t0->tv_nsec) / 1e9);
}

static int	argmax(const float *v, int n)
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
static bool	timed_decode(llama_context *ctx, llama_batch batch,
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

static bool	decode_prompt(llama_context *ctx,
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

typedef struct s_gen_run_result
{
	std::vector<int32_t>				tokens;
	std::vector<std::vector<float>>	logits;	/* captured iff requested */
	bool								ok;
}	gen_run_result_t;

/*
 * Free-running (teacher_force == NULL): argmax's its own logits each
 * step, stops at EOG or gen_tokens steps, whichever first.
 * Teacher-forced (teacher_force != NULL): ignores its own argmax and
 * decodes exactly teacher_force's tokens, in order -- used to replay
 * run A's reference sequence into an injecting context for an aligned
 * comparison (see this file's module comment, pass C). Captures the
 * logits available BEFORE each step's token is decoded, iff
 * capture_logits (these are the logits that "chose"/"predict" that
 * token). Appends decoded text to *text_out iff non-NULL.
 */
static void	run_generation(llama_context *ctx, const llama_vocab *vocab,
				int32_t n_vocab, int gen_tokens,
				membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, int debug,
				bool capture_logits,
				const std::vector<int32_t> *teacher_force,
				uint64_t *abs_pos, std::string *text_out,
				gen_run_result_t *out)
{
	int			step;
	int			limit;
	const float	*logits;
	llama_token	tok;
	char		step_label[32];

	out->ok = true;
	limit = teacher_force != NULL ? (int)teacher_force->size() : gen_tokens;
	step = 0;
	while (step < limit)
	{
		logits = llama_get_logits_ith(ctx, -1);
		if (logits == NULL)
		{
			fprintf(stderr, "membrane-llama-run: logits unavailable\n");
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
		if (text_out != NULL)
		{
			char	piece[256];
			int		n;

			n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0,
					true);
			if (n > 0)
				text_out->append(piece, n);
		}
		snprintf(step_label, sizeof(step_label), "gen step %d", step + 1);
		if (!timed_decode(ctx, llama_batch_get_one(&tok, 1), *abs_pos, 1,
				collector, hook_ctx, debug, step_label))
		{
			fprintf(stderr, "membrane-llama-run: generation decode "
				"failed\n");
			out->ok = false;
			break ;
		}
		*abs_pos += 1;
		step++;
	}
}

/* Loads the model once and creates ONE fresh llama_context for this
 * call, decodes the prompt, runs generation via run_generation, then
 * destroys the context before returning -- callers needing multiple
 * passes (INJECT modes) call this once per pass, never holding more
 * than one llama_context alive at a time. */
static bool	run_one_pass(llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				int gen_tokens, membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, /* NULL: no cb_eval */
				int debug, bool capture_logits,
				const std::vector<int32_t> *teacher_force,
				std::string *text_out, gen_run_result_t *out)
{
	llama_context			*ctx;
	llama_context_params	cp;
	const llama_vocab		*vocab;
	int32_t					n_vocab;
	uint64_t				abs_pos;

	vocab = llama_model_get_vocab(model);
	n_vocab = llama_vocab_n_tokens(vocab);
	cp = llama_context_default_params();
	cp.n_ctx = (uint32_t)prompt_tokens.size() + (uint32_t)gen_tokens + 8;
	cp.n_batch = 256;
	cp.n_threads = 4;
	cp.n_threads_batch = 4;
	cp.no_perf = true;
	if (hook_ctx != NULL)
	{
		cp.cb_eval = membrane_llama_eval_callback;
		cp.cb_eval_user_data = hook_ctx;
	}
	ctx = llama_init_from_model(model, cp);
	if (ctx == NULL)
	{
		fprintf(stderr, "membrane-llama-run: context creation failed\n");
		out->ok = false;
		return (false);
	}
	abs_pos = 0;
	if (!decode_prompt(ctx, prompt_tokens, cp.n_batch, collector, hook_ctx,
			debug, &abs_pos))
	{
		fprintf(stderr, "membrane-llama-run: prompt decode failed\n");
		out->ok = false;
		return (llama_free(ctx), false);
	}
	run_generation(ctx, vocab, n_vocab, gen_tokens, collector, hook_ctx,
		debug, capture_logits, teacher_force, &abs_pos, text_out, out);
	/* The per-step sanity check inside membrane_llama_hook_set_step_
	 * context() only ever validates the PREVIOUS decode step -- there is
	 * no next call to trigger it for this pass's true last step, so it
	 * must be validated explicitly here, before this pass's telemetry is
	 * ever read. */
	membrane_llama_hook_finish(hook_ctx);
	llama_free(ctx);
	return (out->ok);
}

/*
 * Product Phase 7: creates ONE llama_context whose KV cache TENSORS are
 * allocated at `kv_store_mode`'s ggml type (never a shadow/injected
 * copy -- there is no cb_eval here at all, no MEMBRANE-authored write
 * step; llama.cpp's own cpy_k/cpy_v graph ops quantize on write and
 * ggml_mul_mat/ggml_flash_attn_ext dequantize on read, exactly as they
 * would for `-ctk q8_0 -ctv q8_0 -fa` on stock llama.cpp -- see
 * docs/live-runtime.md for why this needs no llama.cpp patch), decodes
 * the prompt and generates, and destroys the context before returning
 * -- one context alive at a time, same discipline as run_one_pass.
 * `ctx_size` must already be resolved (never 0). Captures RSS
 * checkpoints into *tel at the three points Section 7 asks for that are
 * reachable from a single pass (after model load is the caller's job,
 * since it happens before any context exists at all).
 */
static bool	run_kv_store_pass(llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				int gen_tokens, int kv_store_mode, uint32_t ctx_size,
				int debug, const std::vector<int32_t> *teacher_force,
				bool capture_logits, int32_t n_vocab_for_scratch,
				std::string *text_out, membrane_kv_store_telemetry_t *tel,
				gen_run_result_t *out)
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
	 * modes, unconditionally -- not just q8 -- so the native-vs-q8
	 * comparison never mixes "KV cache dtype" with "which attention
	 * kernel ran" as a second, uncontrolled variable; AUTO's runtime
	 * resolution is not relied on for either mode. */
	cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
	if (kv_store_mode == MEMBRANE_KV_STORE_Q8)
	{
		cp.type_k = GGML_TYPE_Q8_0;
		cp.type_v = GGML_TYPE_Q8_0;
	}
	collector = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_BASELINE, MEMBRANE_RUNTIME_MAX_LAYERS);
	if (collector == NULL)
		return (out->ok = false, false);
	ctx = llama_init_from_model(model, cp);
	if (ctx == NULL)
	{
		fprintf(stderr, "membrane-llama-run: kv-store context creation "
			"failed (see llama's own stderr diagnostics above -- for "
			"q8, this fails closed rather than silently falling back to "
			"native storage)\n");
		membrane_runtime_collector_destroy(collector);
		return (out->ok = false, false);
	}
	membrane_kv_store_read_rss(&tel->rss_after_context);
	abs_pos = 0;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (!decode_prompt(ctx, prompt_tokens, cp.n_batch, collector, NULL,
			debug, &abs_pos))
	{
		fprintf(stderr, "membrane-llama-run: kv-store prompt decode "
			"failed\n");
		membrane_runtime_collector_destroy(collector);
		return (llama_free(ctx), out->ok = false, false);
	}
	prompt_seconds = seconds_since(&t0);
	membrane_kv_store_read_rss(&tel->rss_after_prompt);
	out->ok = true;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	run_generation(ctx, vocab, n_vocab, gen_tokens, collector, NULL, debug,
		capture_logits, teacher_force, &abs_pos, text_out, out);
	gen_seconds = seconds_since(&t0);
	membrane_kv_store_read_rss(&tel->rss_final);
	tel->prompt_tok_per_s = prompt_seconds > 0.0
		? (double)prompt_tokens.size() / prompt_seconds : 0.0;
	tel->generation_tok_per_s = (gen_seconds > 0.0 && !out->tokens.empty())
		? (double)out->tokens.size() / gen_seconds : 0.0;
	/* Only MEMBRANE-owned allocation on this path: run_generation's
	 * per-step logit capture (gen_run_result_t::logits), when the
	 * caller actually asked for it -- never held for the canonical
	 * pass B itself (that pass doesn't need its own logits, only pass A
	 * and C do, for the aligned comparison), so a plain q8 run reports
	 * this as genuinely 0, not a placeholder. */
	if (capture_logits)
	{
		scratch_bytes = (uint64_t)out->logits.size()
			* (uint64_t)n_vocab_for_scratch * sizeof(float);
		if (scratch_bytes > tel->scratch_peak_bytes)
			tel->scratch_peak_bytes = scratch_bytes;
	}
	membrane_runtime_collector_destroy(collector);
	llama_free(ctx);
	return (out->ok);
}

/* Compares run A's (baseline) and run C's (teacher-forced injection)
 * per-step logits/NLL and folds the result into `collector` as this
 * run's aligned behavior summary. A and C must have decoded the exact
 * same number of steps (C was teacher-forced on A's own token
 * sequence) -- uses the shorter of the two if they somehow differ
 * (e.g. either aborted early), which is always safe (min(), never an
 * out-of-bounds read). */
static void	record_aligned_behavior(const gen_run_result_t &a,
				const gen_run_result_t &c, int32_t n_vocab,
				membrane_runtime_collector_t *collector)
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
	membrane_runtime_set_behavior(collector, &summary);
}

/* Phase 7 analogue of record_aligned_behavior() above, for the
 * kv-store telemetry struct instead of the injection collector -- same
 * comparison machinery (membrane_runtime_compare_logits/nll/behavior_*,
 * llama-free, in runtime_core.h), different destination struct. `a` is
 * the native-storage reference pass, `c` is the q8-storage pass
 * teacher-forced on `a`'s own tokens. */
static bool	record_kv_store_behavior(const gen_run_result_t &a,
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

int	main(int argc, char **argv)
{
	run_opts_t					o;
	llama_model					*model;
	const llama_vocab			*vocab;
	std::vector<llama_token>	prompt_tokens;
	std::string					prompt_text;
	std::string					generated_text;
	int							rc;
	membrane_runtime_telemetry_t	t;
	gen_run_result_t			result_primary;
	int							exit_code;

	rc = parse_opts(argc, argv, &o);
	if (o.want_help)
		return (usage(stdout), 0);
	if (rc != 0)
		return (usage(stderr), 2);
	llama_backend_init();
	model = llama_model_load_from_file(o.model_path,
			llama_model_default_params());
	if (model == NULL)
	{
		fprintf(stderr, "membrane-llama-run: failed to load model\n");
		return (1);
	}
	vocab = llama_model_get_vocab(model);
	prompt_text = read_file(o.prompt_path);
	if (prompt_text.empty())
	{
		fprintf(stderr, "membrane-llama-run: empty or unreadable prompt "
			"file\n");
		return (llama_model_free(model), 1);
	}
	prompt_tokens.resize(prompt_text.size() + 8);
	rc = llama_tokenize(vocab, prompt_text.c_str(),
			(int32_t)prompt_text.size(), prompt_tokens.data(),
			(int32_t)prompt_tokens.size(), true, false);
	if (rc < 0)
	{
		fprintf(stderr, "membrane-llama-run: tokenization failed\n");
		return (llama_model_free(model), 1);
	}
	prompt_tokens.resize(rc);
	exit_code = 0;
	if (o.have_kv_store)
	{
		membrane_kv_store_telemetry_t	tel;
		membrane_kv_store_telemetry_t	tel_scratch;
		membrane_kv_store_rss_t		rss_after_model_load;
		uint32_t						ctx_size;
		int32_t							n_layer;
		int32_t							n_embd;
		int32_t							n_head;
		int32_t							n_head_kv;
		int64_t							n_embd_gqa;
		membrane_kv_store_bytes_t		native_b;
		membrane_kv_store_bytes_t		q8_b;
		membrane_runtime_divergence_t	divergence;
		gen_run_result_t				result_a;
		gen_run_result_t				result_b;
		gen_run_result_t				result_c;
		int32_t							n_vocab;

		membrane_kv_store_read_rss(&rss_after_model_load);
		memset(&tel, 0, sizeof(tel));
		tel.kv_store_requested = 1;
		tel.kv_store_mode_name = o.kv_store_mode == MEMBRANE_KV_STORE_Q8
			? "q8" : "native";
		ctx_size = o.kv_store_ctx > 0 ? o.kv_store_ctx
			: (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
		tel.ctx_size = ctx_size;
		tel.rss_after_model_load = rss_after_model_load;
		tel.no_fallback_occurred = 1;
		tel.first_divergence = -1;
		/* Real per-model constants (public llama_model_n_* getters) fed
		 * through ggml's OWN row-size rule (ggml_row_size, the exact
		 * arithmetic llama.cpp's tensor allocator uses) -- not a
		 * MEMBRANE-invented formula. n_embd_head = n_embd/n_head is an
		 * LLM_ARCH_LLAMA-scoped assumption (matches this project's
		 * existing scope limitation, see docs/live-runtime.md); cross-
		 * checked against real RSS deltas below, which need no such
		 * assumption at all. */
		n_layer = llama_model_n_layer(model);
		n_embd = llama_model_n_embd(model);
		n_head = llama_model_n_head(model);
		n_head_kv = llama_model_n_head_kv(model);
		n_embd_gqa = n_head > 0
			? (int64_t)(n_embd / n_head) * n_head_kv : 0;
		native_b.n_layer = (uint64_t)n_layer;
		native_b.kv_size = ctx_size;
		native_b.bytes_per_token_k = ggml_row_size(GGML_TYPE_F16,
				n_embd_gqa);
		native_b.bytes_per_token_v = ggml_row_size(GGML_TYPE_F16,
				n_embd_gqa);
		q8_b = native_b;
		q8_b.bytes_per_token_k = ggml_row_size(GGML_TYPE_Q8_0, n_embd_gqa);
		q8_b.bytes_per_token_v = ggml_row_size(GGML_TYPE_Q8_0, n_embd_gqa);
		tel.native_kv_allocated_bytes = membrane_kv_store_total_bytes(
				&native_b);
		tel.compressed_kv_allocated_bytes =
			o.kv_store_mode == MEMBRANE_KV_STORE_Q8
			? membrane_kv_store_total_bytes(&q8_b)
			: tel.native_kv_allocated_bytes;
		if (o.kv_store_mode == MEMBRANE_KV_STORE_Q8)
		{
			/* Pass B (q8, canonical/memory-reported) runs FIRST, as the
			 * very first llama_context in this process after model
			 * load -- vm_hwm_kb/ru_maxrss_kb are process-wide monotonic
			 * counters, so if the native reference pass ran first, its
			 * high-water mark would leak into pass B's own "peak"
			 * reading even after its context is freed (a real
			 * methodology bug caught in review). Passes A and C run
			 * afterward, comparison-only, into tel_scratch so their
			 * memory footprint is never folded into the reported
			 * telemetry. Neither A nor C captures print_text output --
			 * that's the canonical pass B's job. */
			memset(&tel_scratch, 0, sizeof(tel_scratch));
			if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
					MEMBRANE_KV_STORE_Q8, ctx_size, o.debug_runtime, NULL,
					false, 0, o.print_text ? &generated_text : NULL, &tel,
					&result_b))
				return (llama_model_free(model), 1);
			tel.generated_tokens = result_b.tokens.size();
			n_vocab = llama_vocab_n_tokens(vocab);
			memset(&tel_scratch, 0, sizeof(tel_scratch));
			if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
					MEMBRANE_KV_STORE_NATIVE, ctx_size, o.debug_runtime,
					NULL, true, n_vocab, NULL, &tel_scratch, &result_a))
				return (llama_model_free(model), 1);
			if (tel_scratch.scratch_peak_bytes > tel.scratch_peak_bytes)
				tel.scratch_peak_bytes = tel_scratch.scratch_peak_bytes;
			if (!result_b.ok)
				exit_code = 1;
			membrane_runtime_detect_divergence(result_a.tokens.data(),
				result_a.tokens.size(), result_b.tokens.data(),
				result_b.tokens.size(), &divergence);
			if (!result_a.tokens.empty())
			{
				memset(&tel_scratch, 0, sizeof(tel_scratch));
				if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
						MEMBRANE_KV_STORE_Q8, ctx_size, o.debug_runtime,
						&result_a.tokens, true, n_vocab, NULL, &tel_scratch,
						&result_c))
					fprintf(stderr, "membrane-llama-run: aligned "
						"teacher-forced kv-store pass failed -- logit/"
						"NLL comparison unavailable, storage result "
						"otherwise still reported\n");
				else
				{
					if (tel_scratch.scratch_peak_bytes
						> tel.scratch_peak_bytes)
						tel.scratch_peak_bytes =
							tel_scratch.scratch_peak_bytes;
					/* Only claim the quality bundle is available once
					 * every field in it (identity, divergence, AND the
					 * logit-level stats below) genuinely has data --
					 * setting this before record_kv_store_behavior()
					 * succeeded left top1_preservation/delta_nll/
					 * logit_rel_l2 at their memset zero while still
					 * claiming "available": a silent-zero, not "not
					 * measured". */
					if (record_kv_store_behavior(result_a, result_c,
							n_vocab, &tel))
					{
						tel.quality_available = 1;
						tel.token_identity = divergence.identical;
						tel.first_divergence =
							(int32_t)divergence.first_divergence_step;
					}
					else
						fprintf(stderr, "membrane-llama-run: behavior "
							"accumulator allocation failed -- logit/NLL "
							"comparison unavailable, storage result "
							"otherwise still reported\n");
				}
			}
		}
		else
		{
			if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
					MEMBRANE_KV_STORE_NATIVE, ctx_size, o.debug_runtime,
					NULL, false, 0, o.print_text ? &generated_text : NULL,
					&tel, &result_b))
				return (llama_model_free(model), 1);
			tel.generated_tokens = result_b.tokens.size();
		}
		if (o.print_text)
			fprintf(stderr, "\n[generated text]\n%s\n",
				generated_text.c_str());
		membrane_kv_store_rss_max(&tel.rss_after_model_load,
			&tel.rss_after_context, &tel.rss_peak);
		membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_after_prompt,
			&tel.rss_peak);
		membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_final,
			&tel.rss_peak);
		if (o.want_json)
			membrane_kv_store_print_json(&tel, stdout);
		else
			membrane_kv_store_print_human(&tel, stdout);
		llama_model_free(model);
		llama_backend_free();
		return (exit_code);
	}
	if (!membrane_runtime_mode_is_inject(o.mode))
	{
		membrane_runtime_collector_t	*collector;
		membrane_llama_hook_ctx_t		*hook_ctx;

		collector = membrane_runtime_collector_create(o.mode,
				MEMBRANE_RUNTIME_MAX_LAYERS);
		if (collector == NULL)
		{
			fprintf(stderr, "membrane-llama-run: telemetry collector "
				"allocation failed\n");
			return (llama_model_free(model), 1);
		}
		hook_ctx = membrane_llama_hook_create(collector,
				membrane_simd_best_backend(), o.debug_runtime,
				MEMBRANE_RUNTIME_MAX_LAYERS, o.mode, NULL, 0);
		if (hook_ctx == NULL)
		{
			fprintf(stderr, "membrane-llama-run: hook context allocation "
				"failed\n");
			membrane_runtime_collector_destroy(collector);
			return (llama_model_free(model), 1);
		}
		if (!run_one_pass(model, prompt_tokens, o.gen_tokens, collector,
				o.mode == MEMBRANE_RUNTIME_MODE_BASELINE ? NULL : hook_ctx,
				o.debug_runtime, false, NULL,
				o.print_text ? &generated_text : NULL, &result_primary))
		{
			membrane_llama_hook_destroy(hook_ctx);
			membrane_runtime_collector_destroy(collector);
			return (llama_model_free(model), 1);
		}
		membrane_runtime_set_generated_tokens(collector,
			(uint64_t)result_primary.tokens.size());
		membrane_runtime_finalize(collector, &t);
		membrane_llama_hook_destroy(hook_ctx);
		membrane_runtime_collector_destroy(collector);
	}
	else
	{
		membrane_runtime_scope_t		inject_scope;
		membrane_runtime_collector_t	*collector_a;
		membrane_runtime_collector_t	*collector_b;
		membrane_runtime_collector_t	*collector_c;
		membrane_llama_hook_ctx_t		*hook_b;
		membrane_llama_hook_ctx_t		*hook_c;
		gen_run_result_t				result_a;
		gen_run_result_t				result_c;
		membrane_runtime_divergence_t	divergence;
		int32_t							n_vocab;
		size_t							i;

		n_vocab = llama_vocab_n_tokens(vocab);
		membrane_runtime_scope_init(&inject_scope);
		for (i = 0; i < o.inject_layers.size(); i++)
			membrane_runtime_scope_add_layer(&inject_scope,
				o.inject_layers[i]);
		membrane_runtime_scope_set_tensor(&inject_scope, o.inject_tensor);
		if (o.have_token_start)
			membrane_runtime_scope_set_token_range(&inject_scope,
				o.inject_token_start, o.inject_token_end);
		/* Pass A: pure baseline, free-running, no cb_eval installed --
		 * the reference token sequence + reference per-step logits. */
		collector_a = membrane_runtime_collector_create(
				MEMBRANE_RUNTIME_MODE_BASELINE, MEMBRANE_RUNTIME_MAX_LAYERS);
		if (collector_a == NULL)
			return (llama_model_free(model), 1);
		if (!run_one_pass(model, prompt_tokens, o.gen_tokens, collector_a,
				NULL, o.debug_runtime, true, NULL, NULL, &result_a))
		{
			membrane_runtime_collector_destroy(collector_a);
			return (llama_model_free(model), 1);
		}
		membrane_runtime_collector_destroy(collector_a);
		/* Pass B: injection, free-running -- this pass's telemetry is
		 * the one reported (canonical injected/failed block counts,
		 * coverage, storage/accuracy accounting). */
		collector_b = membrane_runtime_collector_create(o.mode,
				MEMBRANE_RUNTIME_MAX_LAYERS);
		if (collector_b == NULL)
			return (llama_model_free(model), 1);
		hook_b = membrane_llama_hook_create(collector_b,
				membrane_simd_best_backend(), o.debug_runtime,
				MEMBRANE_RUNTIME_MAX_LAYERS, o.mode, &inject_scope,
				o.debug_perturb_injection);
		if (hook_b == NULL)
		{
			membrane_runtime_collector_destroy(collector_b);
			return (llama_model_free(model), 1);
		}
		if (!run_one_pass(model, prompt_tokens, o.gen_tokens, collector_b,
				hook_b, o.debug_runtime, false, NULL,
				o.print_text ? &generated_text : NULL, &result_primary))
		{
			membrane_llama_hook_destroy(hook_b);
			membrane_runtime_collector_destroy(collector_b);
			return (llama_model_free(model), 1);
		}
		/* Pass C: injection, teacher-forced on A's reference sequence --
		 * feeds the aligned behavior summary onto collector_b. Skipped
		 * (behavior stays unavailable) if A produced no tokens at all --
		 * nothing to teacher-force or compare. */
		if (!result_a.tokens.empty())
		{
			collector_c = membrane_runtime_collector_create(o.mode,
					MEMBRANE_RUNTIME_MAX_LAYERS);
			if (collector_c == NULL)
			{
				membrane_llama_hook_destroy(hook_b);
				membrane_runtime_collector_destroy(collector_b);
				return (llama_model_free(model), 1);
			}
			hook_c = membrane_llama_hook_create(collector_c,
					membrane_simd_best_backend(), o.debug_runtime,
					MEMBRANE_RUNTIME_MAX_LAYERS, o.mode, &inject_scope,
					o.debug_perturb_injection);
			if (hook_c == NULL)
			{
				membrane_runtime_collector_destroy(collector_c);
				membrane_llama_hook_destroy(hook_b);
				membrane_runtime_collector_destroy(collector_b);
				return (llama_model_free(model), 1);
			}
			if (!run_one_pass(model, prompt_tokens, o.gen_tokens,
					collector_c, hook_c, o.debug_runtime, true,
					&result_a.tokens, NULL, &result_c))
			{
				fprintf(stderr, "membrane-llama-run: aligned/teacher-"
					"forced pass failed -- behavior comparison "
					"unavailable, injection result otherwise still "
					"reported\n");
			}
			else
				record_aligned_behavior(result_a, result_c, n_vocab,
					collector_b);
			if (membrane_runtime_injection_has_failed(collector_c))
				exit_code = 1;
			membrane_llama_hook_destroy(hook_c);
			membrane_runtime_collector_destroy(collector_c);
		}
		membrane_runtime_detect_divergence(result_a.tokens.data(),
			result_a.tokens.size(), result_primary.tokens.data(),
			result_primary.tokens.size(), &divergence);
		membrane_runtime_set_divergence(collector_b, &divergence);
		membrane_runtime_set_generated_tokens(collector_b,
			(uint64_t)result_primary.tokens.size());
		membrane_runtime_finalize(collector_b, &t);
		if (membrane_runtime_injection_has_failed(collector_b))
			exit_code = 1;
		/* t.injection_succeeded already folds in "requested but nothing
		 * was ever actually injected" (see membrane_runtime_finalize) --
		 * that must fail the run's exit code too, not just the printed
		 * telemetry. */
		if (!t.injection_succeeded)
			exit_code = 1;
		membrane_llama_hook_destroy(hook_b);
		membrane_runtime_collector_destroy(collector_b);
	}
	if (o.want_json)
		membrane_runtime_print_json(&t,
			membrane_runtime_safe_basename(o.model_path),
			membrane_runtime_safe_basename(o.prompt_path),
			result_primary.tokens.data(), result_primary.tokens.size(),
			stdout);
	else
		membrane_runtime_print_human(&t, stdout);
	if (o.print_text)
	{
		/* Always stderr, never stdout: --json's primary output must stay
		 * pipeable into a JSON parser even when --print-text is also
		 * given (the two flags are documented as independent, so this
		 * combination is reachable, not just theoretical). */
		fprintf(stderr, "\n[generated text]\n%s\n", generated_text.c_str());
	}
	if (membrane_runtime_mode_is_inject(o.mode) && exit_code != 0)
		fprintf(stderr, "membrane-llama-run: INJECTION FAILED -- at "
			"least one targeted block could not be reconstructed; "
			"native KV was retained for it, but this run's result is "
			"reported as a failure, never silently as success\n");
	llama_model_free(model);
	llama_backend_free();
	return (exit_code);
}
