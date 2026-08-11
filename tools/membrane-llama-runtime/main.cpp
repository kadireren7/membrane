#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "llama.h"
#include "llama_hook.h"
#include "runtime_core.h"

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

typedef struct s_run_opts
{
	const char					*model_path;
	const char					*prompt_path;
	membrane_runtime_mode_t		mode;
	int							gen_tokens;
	int							want_json;
	int							print_text;
	int							debug_runtime;
	int							want_help;

	/* Phase 6: injection scope + debug proof. */
	std::vector<uint32_t>		inject_layers;
	int							inject_tensor;
	int							have_token_start;
	int							have_token_end;
	uint64_t					inject_token_start;
	uint64_t					inject_token_end;
	int							debug_perturb_injection;
}	run_opts_t;

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-llama-run --model FILE.gguf --prompt-file FILE "
		"--mode MODE [options]\n"
		"\n"
		"MEMBRANE Product Phase 5/6: live shadow/injection runtime. Real\n"
		"llama.cpp inference; MEMBRANE quantizes/dequantizes live K/V\n"
		"projection values as they are computed. SHADOW modes never\n"
		"affect llama's own computation. INJECT modes write reconstructed\n"
		"values back into the tensor that feeds the KV-cache write. Never\n"
		"an actual process-memory reduction claim either way. See\n"
		"docs/live-runtime.md.\n"
		"\n"
		"  --model FILE       .gguf model path (required)\n"
		"  --prompt-file FILE prompt text file (required)\n"
		"  --mode MODE        baseline | shadow-q8 | shadow-adaptive |\n"
		"                     inject-q8 | inject-adaptive (required)\n"
		"                       baseline: native llama path, no\n"
		"                         MEMBRANE processing, zero overhead\n"
		"                       shadow-q8/shadow-adaptive: observe-only,\n"
		"                         native KV remains authoritative\n"
		"                       inject-q8/inject-adaptive: reconstructed\n"
		"                         values ARE written back into the\n"
		"                         tensor that feeds the KV-cache write\n"
		"                         (see --inject-* below to scope this)\n"
		"  --inject-layer N   restrict injection to this layer (may be\n"
		"                     repeated; default: every layer)\n"
		"  --inject-tensor T  k | v | both (default: both)\n"
		"  --inject-token-start N / --inject-token-end N\n"
		"                     restrict injection to this inclusive\n"
		"                     absolute token-position range (both\n"
		"                     required together; default: every token)\n"
		"  --gen-tokens N     greedy tokens to generate (default 32)\n"
		"  --json             machine-readable telemetry on stdout\n"
		"  --print-text       also print the decoded generated text to\n"
		"                     stderr, never stdout (so --json's stdout\n"
		"                     output stays parseable either way) -- off\n"
		"                     by default: no prompt/token text is ever\n"
		"                     emitted unless explicitly requested\n"
		"  --debug-runtime    per-step KV block counts to stderr --\n"
		"                     proof MEMBRANE processing happens\n"
		"                     interleaved with generation, not after\n"
		"  --debug-perturb-injection\n"
		"                     DEBUG ONLY, inject-* modes: deliberately\n"
		"                     corrupts every reconstructed value before\n"
		"                     write-back, to locally prove it is\n"
		"                     consumed -- never use for a reported run\n"
		"  --help             print this message and exit\n");
}

static int	parse_opts(int argc, char **argv, run_opts_t *o)
{
	int	i;

	o->model_path = NULL;
	o->prompt_path = NULL;
	o->mode = MEMBRANE_RUNTIME_MODE_BASELINE;
	o->gen_tokens = 32;
	o->want_json = 0;
	o->print_text = 0;
	o->debug_runtime = 0;
	o->want_help = 0;
	o->inject_tensor = MEMBRANE_RUNTIME_TENSOR_BOTH;
	o->have_token_start = 0;
	o->have_token_end = 0;
	o->inject_token_start = 0;
	o->inject_token_end = 0;
	o->debug_perturb_injection = 0;
	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--help") == 0)
			return (o->want_help = 1, 0);
		else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
			o->model_path = argv[++i];
		else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc)
			o->prompt_path = argv[++i];
		else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
		{
			++i;
			if (!membrane_runtime_mode_from_name(argv[i], &o->mode))
				return (fprintf(stderr, "unknown --mode '%s' (want "
						"baseline, shadow-q8, shadow-adaptive, inject-q8, "
						"or inject-adaptive)\n", argv[i]), -1);
		}
		else if (strcmp(argv[i], "--gen-tokens") == 0 && i + 1 < argc)
		{
			++i;
			o->gen_tokens = atoi(argv[i]);
			if (o->gen_tokens <= 0)
				return (fprintf(stderr,
						"--gen-tokens must be a positive integer\n"), -1);
		}
		else if (strcmp(argv[i], "--inject-layer") == 0 && i + 1 < argc)
		{
			long	val;
			char	*end;

			++i;
			errno = 0;
			val = strtol(argv[i], &end, 10);
			if (errno != 0 || end == argv[i] || *end != '\0' || val < 0
				|| (unsigned long)val >= MEMBRANE_RUNTIME_MAX_LAYERS)
				return (fprintf(stderr, "--inject-layer must be an "
						"integer in [0, %u)\n",
						(unsigned)MEMBRANE_RUNTIME_MAX_LAYERS), -1);
			o->inject_layers.push_back((uint32_t)val);
		}
		else if (strcmp(argv[i], "--inject-tensor") == 0 && i + 1 < argc)
		{
			++i;
			if (strcmp(argv[i], "k") == 0)
				o->inject_tensor = MEMBRANE_RUNTIME_TENSOR_K;
			else if (strcmp(argv[i], "v") == 0)
				o->inject_tensor = MEMBRANE_RUNTIME_TENSOR_V;
			else if (strcmp(argv[i], "both") == 0)
				o->inject_tensor = MEMBRANE_RUNTIME_TENSOR_BOTH;
			else
				return (fprintf(stderr,
						"--inject-tensor must be k, v, or both\n"), -1);
		}
		else if (strcmp(argv[i], "--inject-token-start") == 0
			&& i + 1 < argc)
		{
			char	*end;

			++i;
			errno = 0;
			o->inject_token_start = strtoull(argv[i], &end, 10);
			if (errno != 0 || end == argv[i] || *end != '\0')
				return (fprintf(stderr, "--inject-token-start must be a "
						"non-negative integer\n"), -1);
			o->have_token_start = 1;
		}
		else if (strcmp(argv[i], "--inject-token-end") == 0 && i + 1 < argc)
		{
			char	*end;

			++i;
			errno = 0;
			o->inject_token_end = strtoull(argv[i], &end, 10);
			if (errno != 0 || end == argv[i] || *end != '\0')
				return (fprintf(stderr, "--inject-token-end must be a "
						"non-negative integer\n"), -1);
			o->have_token_end = 1;
		}
		else if (strcmp(argv[i], "--debug-perturb-injection") == 0)
			o->debug_perturb_injection = 1;
		else if (strcmp(argv[i], "--json") == 0)
			o->want_json = 1;
		else if (strcmp(argv[i], "--print-text") == 0)
			o->print_text = 1;
		else if (strcmp(argv[i], "--debug-runtime") == 0)
			o->debug_runtime = 1;
		else
			return (fprintf(stderr, "unknown option: %s\n", argv[i]), -1);
		i++;
	}
	if (o->model_path == NULL || o->prompt_path == NULL)
		return (fprintf(stderr,
				"--model and --prompt-file are required\n"), -1);
	if (o->have_token_start != o->have_token_end)
		return (fprintf(stderr, "--inject-token-start and "
				"--inject-token-end must be given together\n"), -1);
	if (!membrane_runtime_mode_is_inject(o->mode)
		&& (!o->inject_layers.empty()
			|| o->inject_tensor != MEMBRANE_RUNTIME_TENSOR_BOTH
			|| o->have_token_start || o->debug_perturb_injection))
		return (fprintf(stderr, "--inject-* flags require --mode "
				"inject-q8 or inject-adaptive\n"), -1);
	return (0);
}

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
		if (exit_code != 0)
			t.injection_succeeded = 0;
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
