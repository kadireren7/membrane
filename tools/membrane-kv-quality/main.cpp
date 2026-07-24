/*
 * membrane-kv-quality: measures the model-quality impact of quantizing
 * the KV cache during real inference (Phase 3.1, item 7).
 *
 * MEMBRANE's own MEMBRANE_CODEC_F16_Q8_BLOCK codec (src/codecs/q8block.c)
 * is a standalone, offline transform -- it is not wired into any
 * inference runtime yet (that is future integration work, roadmap
 * Phase 3+). To measure what block-wise int8 KV quantization actually
 * does to a running model's outputs today, this tool uses llama.cpp's
 * OWN native ggml Q8_0 KV cache type (per-32-element-block symmetric int8
 * with an F16 scale) as the closest real analogue: same granularity and
 * quantization family as MEMBRANE's group_elems=32 symmetric config, just
 * implemented inside llama.cpp's attention kernels instead of MEMBRANE's
 * codec. Every number below is measured from actual llama.cpp inference,
 * not derived from the offline codec.
 *
 * Methodology: three decode passes over the same prompt and model.
 *   1. Baseline (F16 KV): greedy-decode gen_tokens steps, recording every
 *      chosen token and its full logit vector.
 *   2. Quantized (Q8_0 KV), free-running: greedy-decode independently
 *      from the same prompt, to see whether the *generated text* itself
 *      diverges over time.
 *   3. Quantized (Q8_0 KV), teacher-forced: fed the exact token sequence
 *      pass 1 chose, so pass 3's logits are directly comparable to pass
 *      1's logits at matched positions -- isolating the KV-quantization
 *      effect on logits from any confound of the two passes having
 *      walked different token sequences.
 * Peak RSS is a whole-process running maximum on Linux, so a clean
 * per-variant number needs separate process invocations: pass --variant
 * baseline or --variant q8 to measure one in isolation.
 */

#include <sys/resource.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "llama.h"
#include "membrane/q8block.h"

typedef struct s_quality_opts
{
	const char	*model_path;
	const char	*prompt_path;
	int			n_tokens;
	int			gen_tokens;
	const char	*variant;	/* "both", "baseline", or "q8" */
}	quality_opts_t;

typedef struct s_pass_result
{
	std::vector<llama_token>			tokens;
	std::vector<std::vector<float>>	logits;
	double								tokens_per_sec;
}	pass_result_t;

static int	die(const char *msg)
{
	fprintf(stderr, "membrane-kv-quality: %s\n", msg);
	return (-1);
}

static std::string	read_prompt(const char *path)
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

static std::vector<llama_token>	tokenize_prompt(const llama_vocab *vocab,
									const std::string &text)
{
	std::vector<llama_token>	out;
	int							n;

	out.resize(text.size() + 8);
	n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
			out.data(), (int32_t)out.size(), true, false);
	if (n < 0)
		out.clear();
	else
		out.resize(n);
	return (out);
}

static llama_context	*make_context(llama_model *model, int n_ctx,
						ggml_type type_k, ggml_type type_v)
{
	llama_context_params	cp;

	cp = llama_context_default_params();
	cp.n_ctx = (uint32_t)n_ctx;
	cp.n_batch = 256;
	cp.n_threads = 4;
	cp.n_threads_batch = 4;
	cp.type_k = type_k;
	cp.type_v = type_v;
	if (type_k != GGML_TYPE_F16 || type_v != GGML_TYPE_F16)
		cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
	return (llama_init_from_model(model, cp));
}

static int	decode_prompt(llama_context *ctx,
				const std::vector<llama_token> &tokens, int n_batch)
{
	size_t	off;
	size_t	n;

	off = 0;
	while (off < tokens.size())
	{
		n = tokens.size() - off;
		if (n > (size_t)n_batch)
			n = (size_t)n_batch;
		if (llama_decode(ctx,
				llama_batch_get_one((llama_token *)tokens.data() + off,
					(int32_t)n)) != 0)
			return (die("llama_decode (prompt) failed"));
		off += n;
	}
	return (0);
}

/*
 * Greedy-decodes gen_steps tokens. If `forced` is non-NULL, the token at
 * each step is taken from it instead of argmax-sampled (teacher forcing),
 * but the logits recorded are always this context's own -- this is what
 * makes pass 3 comparable to pass 1 at matched positions.
 */
static bool	run_pass(llama_context *ctx, const llama_vocab *vocab,
				const std::vector<llama_token> &prompt, int gen_steps,
				const std::vector<llama_token> *forced, pass_result_t *out)
{
	llama_token				tok;
	const float				*logits;
	int32_t					n_vocab;
	int						i;
	int						best;
	int						j;
	std::chrono::steady_clock::time_point	t0;
	std::chrono::steady_clock::time_point	t1;

	if (decode_prompt(ctx, prompt, 256) != 0)
		return (false);
	n_vocab = llama_vocab_n_tokens(vocab);
	t0 = std::chrono::steady_clock::now();
	i = 0;
	while (i < gen_steps)
	{
		logits = llama_get_logits_ith(ctx, -1);
		if (logits == NULL)
			return (die("llama_get_logits_ith failed"), false);
		out->logits.emplace_back(logits, logits + n_vocab);
		if (forced != NULL && i < (int)forced->size())
			tok = (*forced)[i];
		else
		{
			best = 0;
			j = 1;
			while (j < n_vocab)
			{
				if (logits[j] > logits[best])
					best = j;
				j++;
			}
			tok = best;
		}
		out->tokens.push_back(tok);
		if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0)
			return (die("llama_decode (gen) failed"), false);
		i++;
	}
	t1 = std::chrono::steady_clock::now();
	out->tokens_per_sec = (double)gen_steps
		/ std::chrono::duration<double>(t1 - t0).count();
	return (true);
}

static std::string	tokens_to_text(const llama_vocab *vocab,
						const std::vector<llama_token> &toks)
{
	std::string	out;
	char		buf[256];
	int			n;

	for (llama_token t : toks)
	{
		n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
		if (n > 0)
			out.append(buf, n);
	}
	return (out);
}

typedef struct s_logit_stats
{
	uint64_t	steps;
	uint64_t	top1_agree;
	uint64_t	top5_agree;
	double		kl_sum;
	double		max_logit_diff;
	double		mean_logit_diff;
}	logit_stats_t;

static void	softmax(const std::vector<float> &logits, std::vector<double> &p)
{
	double	mx;
	double	sum;
	size_t	i;

	mx = logits[0];
	for (float v : logits)
		if (v > mx)
			mx = v;
	sum = 0.0;
	p.resize(logits.size());
	for (i = 0; i < logits.size(); i++)
	{
		p[i] = exp((double)logits[i] - mx);
		sum += p[i];
	}
	for (i = 0; i < p.size(); i++)
		p[i] /= sum;
}

static int	argmax(const std::vector<float> &v)
{
	int	best;
	int	i;

	best = 0;
	for (i = 1; i < (int)v.size(); i++)
		if (v[i] > v[best])
			best = i;
	return (best);
}

/* True if `token` is among the 5 highest logits in `logits`. */
static bool	in_top5(const std::vector<float> &logits, int token)
{
	int	rank;
	int	i;

	rank = 0;
	for (i = 0; i < (int)logits.size(); i++)
		if (logits[i] > logits[token])
			rank++;
	return (rank < 5);
}

static void	compare_step(const std::vector<float> &base,
				const std::vector<float> &q8, logit_stats_t *st)
{
	std::vector<double>	pb;
	std::vector<double>	pq;
	int						base_top1;
	int						q8_top1;
	double					diff;
	double					sum_abs;
	size_t					i;

	softmax(base, pb);
	softmax(q8, pq);
	base_top1 = argmax(base);
	q8_top1 = argmax(q8);
	st->steps += 1;
	st->top1_agree += (base_top1 == q8_top1);
	st->top5_agree += in_top5(q8, base_top1);
	sum_abs = 0.0;
	for (i = 0; i < base.size(); i++)
	{
		diff = fabs((double)base[i] - (double)q8[i]);
		sum_abs += diff;
		if (diff > st->max_logit_diff)
			st->max_logit_diff = diff;
		if (pb[i] > 0.0)
			st->kl_sum += pb[i] * log(pb[i] / (pq[i] > 1e-300 ? pq[i] : 1e-300));
	}
	st->mean_logit_diff += sum_abs / (double)base.size();
}

static long	peak_rss_kb(void)
{
	struct rusage	ru;

	getrusage(RUSAGE_SELF, &ru);
	return (ru.ru_maxrss);
}

static void	report_variant_only(llama_context *ctx, const llama_vocab *vocab,
				const std::vector<llama_token> &prompt, int gen_tokens,
				const char *label)
{
	pass_result_t	r;

	if (!run_pass(ctx, vocab, prompt, gen_tokens, NULL, &r))
		return ;
	fprintf(stderr, "%s: %d tokens generated, %.1f tok/s, peak RSS %ld MB\n",
		label, gen_tokens, r.tokens_per_sec, peak_rss_kb() / 1024);
	fprintf(stderr, "%s text: %s\n", label,
		tokens_to_text(vocab, r.tokens).c_str());
}

static void	report_kv_footprint(void)
{
	membrane_q8_cfg_t	cfg;
	size_t				f16_bytes;
	size_t				q8_bytes;

	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 32;
	f16_bytes = 2 * 288;
	q8_bytes = membrane_q8_bound(f16_bytes, &cfg);
	fprintf(stderr,
		"\nKV cache footprint (analytic, from MEMBRANE_CODEC_F16_Q8_BLOCK's "
		"own ratio, NOT an OOM-probed max-context measurement):\n"
		"  per-token K or V row: F16 %zu B vs Q8(sym,32) %zu B -> %.3fx\n"
		"  at a fixed memory budget this implies roughly %.2fx more "
		"context could fit; not empirically probed to an actual limit.\n",
		f16_bytes, q8_bytes, (double)f16_bytes / (double)q8_bytes,
		(double)f16_bytes / (double)q8_bytes);
}

static void	report_comparison(llama_context *base_ctx,
				llama_context *q8_ctx, llama_context *q8_forced_ctx,
				const llama_vocab *vocab,
				const std::vector<llama_token> &prompt, int gen_tokens)
{
	pass_result_t	base;
	pass_result_t	q8_free;
	pass_result_t	q8_forced;
	logit_stats_t	st;
	size_t			first_diff;
	size_t			matched;
	size_t			i;

	memset(&st, 0, sizeof(st));
	if (!run_pass(base_ctx, vocab, prompt, gen_tokens, NULL, &base))
		return ;
	if (!run_pass(q8_ctx, vocab, prompt, gen_tokens, NULL, &q8_free))
		return ;
	if (!run_pass(q8_forced_ctx, vocab, prompt, gen_tokens, &base.tokens,
			&q8_forced))
		return ;
	for (i = 0; i < base.logits.size() && i < q8_forced.logits.size(); i++)
		compare_step(base.logits[i], q8_forced.logits[i], &st);
	first_diff = 0;
	matched = 0;
	while (first_diff < base.tokens.size() && first_diff < q8_free.tokens.size()
		&& base.tokens[first_diff] == q8_free.tokens[first_diff])
	{
		matched++;
		first_diff++;
	}
	fprintf(stderr, "\n=== Model quality: FP16 KV baseline vs llama.cpp "
		"native Q8_0 KV ===\n");
	fprintf(stderr, "gen_tokens=%d  n_vocab=%d\n", gen_tokens,
		llama_vocab_n_tokens(vocab));
	fprintf(stderr, "\nMatched-position logit comparison (Q8 teacher-forced "
		"on baseline's token choices):\n");
	fprintf(stderr, "  top-1 agreement: %llu/%llu (%.2f%%)\n",
		(unsigned long long)st.top1_agree, (unsigned long long)st.steps,
		st.steps ? 100.0 * (double)st.top1_agree / (double)st.steps : 0.0);
	fprintf(stderr, "  top-5 agreement: %llu/%llu (%.2f%%)\n",
		(unsigned long long)st.top5_agree, (unsigned long long)st.steps,
		st.steps ? 100.0 * (double)st.top5_agree / (double)st.steps : 0.0);
	fprintf(stderr, "  mean KL divergence (baseline || q8): %.6f\n",
		st.steps ? st.kl_sum / (double)st.steps : 0.0);
	fprintf(stderr, "  mean |logit diff|: %.6f   max |logit diff|: %.6f\n",
		st.steps ? st.mean_logit_diff / (double)st.steps : 0.0,
		st.max_logit_diff);
	fprintf(stderr, "\nFree-running generation (each variant picks its own "
		"tokens):\n");
	fprintf(stderr, "  tokens identical to baseline before first divergence: "
		"%zu/%zu\n", matched, base.tokens.size());
	fprintf(stderr, "  baseline (F16) tok/s: %.1f\n", base.tokens_per_sec);
	fprintf(stderr, "  q8 (Q8_0)      tok/s: %.1f\n", q8_free.tokens_per_sec);
	fprintf(stderr, "  baseline text: %s\n",
		tokens_to_text(vocab, base.tokens).c_str());
	fprintf(stderr, "  q8       text: %s\n",
		tokens_to_text(vocab, q8_free.tokens).c_str());
	fprintf(stderr, "\nPeak RSS after all three passes (whole-process "
		"running max, not per-variant -- rerun with --variant baseline "
		"or --variant q8 for an isolated number): %ld MB\n",
		peak_rss_kb() / 1024);
	report_kv_footprint();
}

static int	parse_args(int argc, char **argv, quality_opts_t *o)
{
	int	i;

	memset(o, 0, sizeof(*o));
	o->n_tokens = 512;
	o->gen_tokens = 64;
	o->variant = "both";
	i = 1;
	while (i + 1 < argc)
	{
		if (strcmp(argv[i], "--model") == 0)
			o->model_path = argv[i + 1];
		else if (strcmp(argv[i], "--prompt-file") == 0)
			o->prompt_path = argv[i + 1];
		else if (strcmp(argv[i], "--n-tokens") == 0)
			o->n_tokens = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "--gen-tokens") == 0)
			o->gen_tokens = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "--variant") == 0)
			o->variant = argv[i + 1];
		else
			return (die("unknown option"));
		i += 2;
	}
	if (o->model_path == NULL || o->prompt_path == NULL || o->gen_tokens < 1)
		return (die("usage: --model M --prompt-file P [--n-tokens N] "
				"[--gen-tokens G] [--variant both|baseline|q8]"));
	return (0);
}

int	main(int argc, char **argv)
{
	quality_opts_t				o;
	llama_model					*model;
	const llama_vocab			*vocab;
	std::vector<llama_token>	prompt;
	llama_context				*base_ctx;
	llama_context				*q8_ctx;
	llama_context				*q8_forced_ctx;

	if (parse_args(argc, argv, &o) != 0)
		return (2);
	llama_backend_init();
	model = llama_model_load_from_file(o.model_path,
			llama_model_default_params());
	if (model == NULL)
		return (die("model load failed"), 2);
	vocab = llama_model_get_vocab(model);
	prompt = tokenize_prompt(vocab, read_prompt(o.prompt_path));
	if (prompt.empty())
		return (die("tokenization failed"), llama_model_free(model), 2);
	if (strcmp(o.variant, "baseline") == 0)
	{
		base_ctx = make_context(model, o.n_tokens, GGML_TYPE_F16,
				GGML_TYPE_F16);
		if (base_ctx != NULL)
			report_variant_only(base_ctx, vocab, prompt, o.gen_tokens,
				"baseline (F16 KV)");
		llama_free(base_ctx);
	}
	else if (strcmp(o.variant, "q8") == 0)
	{
		q8_ctx = make_context(model, o.n_tokens, GGML_TYPE_Q8_0,
				GGML_TYPE_Q8_0);
		if (q8_ctx != NULL)
			report_variant_only(q8_ctx, vocab, prompt, o.gen_tokens,
				"quantized (Q8_0 KV)");
		llama_free(q8_ctx);
	}
	else
	{
		base_ctx = make_context(model, o.n_tokens, GGML_TYPE_F16,
				GGML_TYPE_F16);
		q8_ctx = make_context(model, o.n_tokens, GGML_TYPE_Q8_0,
				GGML_TYPE_Q8_0);
		q8_forced_ctx = make_context(model, o.n_tokens, GGML_TYPE_Q8_0,
				GGML_TYPE_Q8_0);
		if (base_ctx != NULL && q8_ctx != NULL && q8_forced_ctx != NULL)
			report_comparison(base_ctx, q8_ctx, q8_forced_ctx, vocab, prompt,
				o.gen_tokens);
		else
			die("context creation failed");
		llama_free(base_ctx);
		llama_free(q8_ctx);
		llama_free(q8_forced_ctx);
	}
	llama_model_free(model);
	llama_backend_free();
	return (0);
}
