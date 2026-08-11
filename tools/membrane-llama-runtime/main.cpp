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
 * membrane-llama-run: MEMBRANE Product Phase 5, live llama.cpp shadow
 * runtime. Orchestration only -- model load, prompt tokenization, the
 * greedy decode loop, mode dispatch, and telemetry printing. All
 * llama-facing tensor extraction lives in llama_hook.cpp; all
 * MEMBRANE-side policy/quantization/telemetry lives in runtime_core.c.
 *
 * SHADOW MODE: native llama KV remains the authoritative store for
 * attention in every mode here. baseline installs no eval callback at
 * all (the true, zero-overhead native path); shadow-q8/shadow-adaptive
 * additionally observe and quantize/dequantize live K/V projection
 * values for measurement, never replacing what llama itself stores or
 * reads. See docs/live-runtime.md.
 */

typedef struct s_run_opts
{
	const char				*model_path;
	const char				*prompt_path;
	membrane_runtime_mode_t	mode;
	int						gen_tokens;
	int						want_json;
	int						print_text;
	int						debug_runtime;
	int						want_help;
}	run_opts_t;

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-llama-run --model FILE.gguf --prompt-file FILE "
		"--mode MODE [options]\n"
		"\n"
		"MEMBRANE Product Phase 5: live shadow runtime. Real llama.cpp\n"
		"inference; MEMBRANE quantizes/dequantizes live K/V projection\n"
		"values as they are computed, during generation. Native llama\n"
		"KV remains authoritative in every mode -- this never claims an\n"
		"actual process-memory reduction. See docs/live-runtime.md.\n"
		"\n"
		"  --model FILE       .gguf model path (required)\n"
		"  --prompt-file FILE prompt text file (required)\n"
		"  --mode MODE        baseline | shadow-q8 | shadow-adaptive\n"
		"                     (required)\n"
		"                       baseline: native llama path, no\n"
		"                         MEMBRANE processing, zero overhead\n"
		"                       shadow-q8: every observed live K/V\n"
		"                         block runs through Q8_0 encode/decode\n"
		"                       shadow-adaptive: the maintained content-\n"
		"                         driven Q4/Q8 selector (threshold\n"
		"                         unchanged from Phase 1-4)\n"
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
						"baseline, shadow-q8, or shadow-adaptive)\n",
						argv[i]), -1);
		}
		else if (strcmp(argv[i], "--gen-tokens") == 0 && i + 1 < argc)
		{
			++i;
			o->gen_tokens = atoi(argv[i]);
			if (o->gen_tokens <= 0)
				return (fprintf(stderr,
						"--gen-tokens must be a positive integer\n"), -1);
		}
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
 * extraction and the flush itself) is always a SUBSET of it, never
 * additive -- "overhead ratio" is the fraction of this step's wall
 * time MEMBRANE actually spent in, not extra time layered on top. */
static bool	timed_decode(llama_context *ctx, llama_batch batch,
				membrane_runtime_collector_t *collector,
				membrane_llama_hook_ctx_t *hook_ctx, int debug,
				const char *step_label)
{
	struct timespec	t0;
	int				rc;

	membrane_runtime_begin_step(collector);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	rc = llama_decode(ctx, batch);
	membrane_llama_hook_flush_step(hook_ctx);
	membrane_runtime_add_inference_seconds(collector, seconds_since(&t0));
	/* Live-interleaving proof (section 11): printed AFTER this exact
	 * llama_decode() call returns, using the block count MEMBRANE
	 * processed strictly within this same step -- not a post-run
	 * summary. In shadow modes this count is always > 0 for every step
	 * that produced at least one full block; in baseline it is always
	 * 0 (no callback was ever installed). */
	if (debug)
		fprintf(stderr, "%s: llama decode; membrane processed %llu KV "
			"blocks\n", step_label,
			(unsigned long long)membrane_runtime_step_block_count(
				collector));
	membrane_runtime_end_step(collector);
	return (rc == 0);
}

int	main(int argc, char **argv)
{
	run_opts_t					o;
	membrane_runtime_collector_t	*collector;
	membrane_llama_hook_ctx_t	*hook_ctx;
	llama_model					*model;
	llama_context				*ctx;
	const llama_vocab			*vocab;
	llama_context_params		cp;
	std::vector<llama_token>	prompt_tokens;
	std::vector<int32_t>		generated;
	std::string					prompt_text;
	std::string					generated_text;
	size_t						off;
	size_t						chunk;
	int							rc;
	int32_t						n_vocab;
	int							step;
	membrane_runtime_telemetry_t	t;

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
	/* Always created, regardless of mode: inference_steps/inference_
	 * seconds/generated_tokens are properties of the RUN itself, not of
	 * whether MEMBRANE processed anything. What makes baseline mode the
	 * true zero-overhead native path is that cp.cb_eval is left NULL
	 * below -- membrane_llama_eval_callback (and therefore
	 * membrane_runtime_observe_tensor) is never invoked at all, so no
	 * KV/precision/storage/accuracy counter ever moves off zero. */
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
			MEMBRANE_RUNTIME_MAX_LAYERS);
	if (hook_ctx == NULL)
	{
		fprintf(stderr, "membrane-llama-run: hook context allocation "
			"failed\n");
		membrane_runtime_collector_destroy(collector);
		return (llama_model_free(model), 1);
	}
	cp = llama_context_default_params();
	cp.n_ctx = (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
	cp.n_batch = 256;
	cp.n_threads = 4;
	cp.n_threads_batch = 4;
	cp.no_perf = true;
	if (o.mode != MEMBRANE_RUNTIME_MODE_BASELINE)
	{
		cp.cb_eval = membrane_llama_eval_callback;
		cp.cb_eval_user_data = hook_ctx;
	}
	ctx = llama_init_from_model(model, cp);
	if (ctx == NULL)
	{
		fprintf(stderr, "membrane-llama-run: context creation failed\n");
		membrane_llama_hook_destroy(hook_ctx);
		membrane_runtime_collector_destroy(collector);
		return (llama_model_free(model), 1);
	}
	off = 0;
	while (off < prompt_tokens.size())
	{
		chunk = prompt_tokens.size() - off;
		if (chunk > (size_t)cp.n_batch)
			chunk = (size_t)cp.n_batch;
		if (!timed_decode(ctx,
				llama_batch_get_one(prompt_tokens.data() + off,
					(int32_t)chunk), collector, hook_ctx, o.debug_runtime,
				"prompt decode"))
		{
			fprintf(stderr, "membrane-llama-run: prompt decode failed\n");
			membrane_llama_hook_destroy(hook_ctx);
			membrane_runtime_collector_destroy(collector);
			return (llama_free(ctx), llama_model_free(model), 1);
		}
		off += chunk;
	}
	n_vocab = llama_vocab_n_tokens(vocab);
	step = 0;
	while (step < o.gen_tokens)
	{
		const float	*logits;
		llama_token	tok;
		char		step_label[32];

		logits = llama_get_logits_ith(ctx, -1);
		if (logits == NULL)
		{
			fprintf(stderr, "membrane-llama-run: logits unavailable\n");
			break ;
		}
		tok = (llama_token)argmax(logits, n_vocab);
		if (llama_vocab_is_eog(vocab, tok))
			break ;
		generated.push_back((int32_t)tok);
		if (o.print_text)
		{
			char	piece[256];
			int		n;

			n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0,
					true);
			if (n > 0)
				generated_text.append(piece, n);
		}
		snprintf(step_label, sizeof(step_label), "gen step %d", step + 1);
		if (!timed_decode(ctx, llama_batch_get_one(&tok, 1), collector,
				hook_ctx, o.debug_runtime, step_label))
		{
			fprintf(stderr, "membrane-llama-run: generation decode "
				"failed\n");
			break ;
		}
		step++;
	}
	membrane_runtime_set_generated_tokens(collector,
		(uint64_t)generated.size());
	membrane_runtime_finalize(collector, &t);
	if (o.want_json)
		membrane_runtime_print_json(&t,
			membrane_runtime_safe_basename(o.model_path),
			membrane_runtime_safe_basename(o.prompt_path),
			generated.data(), generated.size(), stdout);
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
	membrane_llama_hook_destroy(hook_ctx);
	membrane_runtime_collector_destroy(collector);
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return (0);
}
