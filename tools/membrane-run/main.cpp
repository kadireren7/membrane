#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "runtime_core.h"
#include "kv_store_telemetry.h"
#include "decode_loop.h"
#include "product_cli.h"
#include "compat_check.h"

/*
 * membrane-run: Product Phase 8, the user-facing MEMBRANE entry point.
 *
 * Normal mode (default): model load, ONE llama_context (native or
 * q8), prompt ingestion, generation, print, exit. No native reference
 * pass, no teacher-forced comparison pass, no per-step logit capture
 * -- run_kv_store_pass() is called with capture_logits=false, so
 * gen_run_result_t::logits never grows past empty. This is the single
 * most load-bearing property of this file; see run_normal_mode().
 *
 * Compare mode (--compare-kv, explicit only): reuses the exact Phase 7
 * 3-pass machinery (native reference, q8 canonical, q8 teacher-forced)
 * via the same run_kv_store_pass()/record_kv_store_behavior() shared
 * with membrane-llama-run (tools/membrane-llama-runtime/decode_loop.h)
 * -- one real implementation, two CLIs on top of it.
 */

/* llama.cpp/ggml log to stderr by default at INFO level -- hundreds of
 * lines of tensor/graph internals per run. Section 7: normal output
 * must be readable, internal diagnostics only with --verbose. Installed
 * before model load so it also covers load_tensors/print_info output,
 * not just decode-time messages. Errors always get through regardless
 * of --verbose -- a real failure should never be silently swallowed. */
static void	quiet_log_callback(enum ggml_log_level level, const char *text,
				void *user_data)
{
	(void)user_data;
	if (level == GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

static std::string	read_stdin(void)
{
	std::string	s;
	char		buf[4096];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0)
		s.append(buf, n);
	return (s);
}

static bool	resolve_prompt(const membrane_run_opts_t &o,
				std::string *prompt_text)
{
	if (o.prompt_mode == MEMBRANE_RUN_PROMPT_TEXT)
	{
		*prompt_text = o.prompt_text;
		return (true);
	}
	if (o.prompt_mode == MEMBRANE_RUN_PROMPT_STDIN)
	{
		*prompt_text = read_stdin();
		return (!prompt_text->empty());
	}
	*prompt_text = read_file(o.prompt_file);
	return (!prompt_text->empty());
}

typedef struct s_model_shape
{
	std::string	arch_name;
	int32_t		n_layer;
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	int64_t		n_embd_gqa;
}	model_shape_t;

static void	read_model_shape(llama_model *model, model_shape_t *s)
{
	char	buf[128];
	int32_t	n;

	n = llama_model_meta_val_str(model, "general.architecture", buf,
			sizeof(buf));
	s->arch_name = (n > 0) ? std::string(buf) : std::string();
	s->n_layer = llama_model_n_layer(model);
	s->n_embd = llama_model_n_embd(model);
	s->n_head = llama_model_n_head(model);
	s->n_head_kv = llama_model_n_head_kv(model);
	s->n_embd_gqa = (s->n_head > 0)
		? (int64_t)(s->n_embd / s->n_head) * s->n_head_kv : 0;
}

static uint64_t	native_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static uint64_t	q8_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static void	print_startup_summary(const membrane_run_opts_t &o,
				const char *model_label, uint32_t ctx_size,
				uint64_t kv_bytes)
{
	fprintf(stderr, "MEMBRANE 0.2.0-rc1\n");
	fprintf(stderr, "model      %s\n", model_label);
	fprintf(stderr, "context    %u\n", ctx_size);
	fprintf(stderr, "kv         %s\n",
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "Q8_0" : "native (F16)");
	fprintf(stderr, "flash attn %s\n",
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "enabled" : "auto");
	fprintf(stderr, "kv bytes   %.2f MiB (%s, from real model "
		"hparams -- not measured until context creation)\n",
		(double)kv_bytes / (1024.0 * 1024.0),
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "Q8_0" : "F16");
}

typedef struct s_token_print_ud
{
	bool	first;
}	token_print_ud_t;

static void	stream_token(const char *piece, size_t piece_len, int step,
				void *ud)
{
	(void)step;
	(void)ud;
	fwrite(piece, 1, piece_len, stdout);
	fflush(stdout);
}

static void	print_run_json(const membrane_run_opts_t &o,
				const char *model_label, const membrane_kv_store_telemetry_t &t,
				const std::string &text)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"0.2.0-rc1\","
		"\"mode\":\"run\",\"model_label\":\"%s\",\"kv_store\":\"%s\","
		"\"ctx_size\":%u,\"generated_tokens\":%llu,",
		model_label, o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "q8" : "native",
		t.ctx_size, (unsigned long long)t.generated_tokens);
	printf("\"storage\":{\"kv_allocated_bytes\":%llu},",
		(unsigned long long)t.compressed_kv_allocated_bytes);
	printf("\"memory\":{\"rss_after_model_load_kb\":%llu,"
		"\"rss_after_context_kb\":%llu,\"rss_after_prompt_kb\":%llu,"
		"\"rss_final_kb\":%llu,\"peak_rss_kb\":%llu},",
		(unsigned long long)t.rss_after_model_load.vm_rss_kb,
		(unsigned long long)t.rss_after_context.vm_rss_kb,
		(unsigned long long)t.rss_after_prompt.vm_rss_kb,
		(unsigned long long)t.rss_final.vm_rss_kb,
		(unsigned long long)t.rss_peak.vm_hwm_kb);
	printf("\"performance\":{\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		t.prompt_tok_per_s, t.generation_tok_per_s);
	printf("\"no_fallback_occurred\":%s",
		t.no_fallback_occurred ? "true" : "false");
	if (o.include_text)
	{
		printf(",\"text\":\"");
		for (char c : text)
		{
			if (c == '"' || c == '\\')
				putchar('\\');
			if ((unsigned char)c >= 0x20 || c == '\t')
				putchar(c);
		}
		printf("\"");
	}
	printf("}\n");
}

static void	print_run_human_stats(const membrane_kv_store_telemetry_t &t)
{
	fprintf(stderr, "\ngenerated  %llu tokens\n",
		(unsigned long long)t.generated_tokens);
	fprintf(stderr, "speed      %.1f tok/s\n", t.generation_tok_per_s);
	fprintf(stderr, "kv memory  %.2f MiB (real allocation)\n",
		(double)t.compressed_kv_allocated_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "rss final  %llu kB (peak %llu kB)\n",
		(unsigned long long)t.rss_final.vm_rss_kb,
		(unsigned long long)t.rss_peak.vm_hwm_kb);
}

/*
 * Normal single-pass mode. Loads the model, creates exactly ONE
 * llama_context (native or q8 per --kv), ingests the prompt, generates,
 * prints, exits. capture_logits is ALWAYS false here -- there is no
 * teacher_force pass, no second/third context, and therefore no
 * per-step logit buffer of any size, let alone a full-context one.
 */
static int	run_normal_mode(const membrane_run_opts_t &o, llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape)
{
	membrane_kv_store_telemetry_t	tel;
	gen_run_result_t				result;
	std::string						text;
	membrane_token_cb_t				cb;
	bool							stream;
	uint64_t						kv_bytes;

	memset(&tel, 0, sizeof(tel));
	tel.kv_store_mode_name = o.kv_mode == MEMBRANE_KV_STORE_Q8
		? "q8" : "native";
	/* This implementation never falls back: run_kv_store_pass() either
	 * succeeds with the requested storage type or fails the whole run
	 * (see the "generation failed" branch below) -- there is no retry-
	 * with-native path for a failed q8 context. */
	tel.no_fallback_occurred = 1;
	kv_bytes = o.kv_mode == MEMBRANE_KV_STORE_Q8
		? q8_kv_bytes(shape, ctx_size) : native_kv_bytes(shape, ctx_size);
	membrane_kv_store_read_rss(&tel.rss_after_model_load);
	if (!o.quiet)
		print_startup_summary(o, model_label, ctx_size, kv_bytes);
	stream = !o.want_json && !o.quiet;
	cb = stream ? stream_token : NULL;
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens, o.kv_mode,
			ctx_size, o.verbose, NULL, false, 0, &text, &tel, &result, cb,
			NULL))
	{
		fprintf(stderr, "membrane-run: generation failed\n");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	tel.generated_tokens = result.tokens.size();
	tel.ctx_size = ctx_size;
	/* kv_bytes is the real allocation size (real ggml_row_size() times
	 * real per-model constants, same arithmetic llama.cpp's own
	 * allocator uses -- see Phase 7) -- not re-derived from anything
	 * run_kv_store_pass() itself measured, since that function has no
	 * reason to know about byte accounting; it only manages the
	 * context/decode loop. */
	tel.compressed_kv_allocated_bytes = kv_bytes;
	membrane_kv_store_rss_max(&tel.rss_after_model_load,
		&tel.rss_after_context, &tel.rss_peak);
	membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_after_prompt,
		&tel.rss_peak);
	membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_final, &tel.rss_peak);
	if (o.want_json)
		print_run_json(o, model_label, tel, text);
	else
	{
		if (!stream)
			fwrite(text.data(), 1, text.size(), stdout);
		if (!o.quiet)
			print_run_human_stats(tel);
		else
			putchar('\n');
	}
	return (result.ok ? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

static void	print_compare_json(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_q8)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"0.2.0-rc1\","
		"\"mode\":\"compare\",\"model_label\":\"%s\",\"ctx_size\":%u,",
		model_label, ctx_size);
	printf("\"storage\":{\"native_kv_allocated_bytes\":%llu,"
		"\"q8_kv_allocated_bytes\":%llu},",
		(unsigned long long)native_bytes,
		(unsigned long long)tel_q8.compressed_kv_allocated_bytes);
	printf("\"memory_q8\":{\"rss_after_context_kb\":%llu,"
		"\"rss_final_kb\":%llu,\"peak_rss_kb\":%llu},",
		(unsigned long long)tel_q8.rss_after_context.vm_rss_kb,
		(unsigned long long)tel_q8.rss_final.vm_rss_kb,
		(unsigned long long)tel_q8.rss_peak.vm_hwm_kb);
	printf("\"performance\":{\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		tel_q8.prompt_tok_per_s, tel_q8.generation_tok_per_s);
	printf("\"quality\":{\"available\":%s,\"token_identity\":%s,"
		"\"first_divergence\":%d,\"logit_rel_l2\":%.6f,"
		"\"top1_preservation\":%.6f,\"delta_nll\":%.6f},",
		tel_q8.quality_available ? "true" : "false",
		tel_q8.token_identity ? "true" : "false", tel_q8.first_divergence,
		tel_q8.logit_rel_l2, tel_q8.top1_preservation, tel_q8.delta_nll);
	printf("\"no_fallback_occurred\":%s}\n",
		tel_q8.no_fallback_occurred ? "true" : "false");
}

static void	print_compare_human(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_q8)
{
	fprintf(stderr, "MEMBRANE 0.2.0-rc1 -- compare mode (native vs q8)\n");
	fprintf(stderr, "model            %s\n", model_label);
	fprintf(stderr, "context          %u\n", ctx_size);
	fprintf(stderr, "native kv bytes  %.2f MiB\n",
		(double)native_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "q8 kv bytes      %.2f MiB (%.3fx smaller)\n",
		(double)tel_q8.compressed_kv_allocated_bytes / (1024.0 * 1024.0),
		(double)native_bytes / (double)tel_q8.compressed_kv_allocated_bytes);
	fprintf(stderr, "q8 rss (ctx)     %llu kB\n",
		(unsigned long long)tel_q8.rss_after_context.vm_rss_kb);
	fprintf(stderr, "q8 tok/s (gen)   %.1f\n", tel_q8.generation_tok_per_s);
	if (tel_q8.quality_available)
		fprintf(stderr, "quality          token_identity=%s "
			"first_divergence=%d top1=%.4f logit_rel_l2=%.6f "
			"delta_nll=%.6f\n",
			tel_q8.token_identity ? "identical" : "diverged",
			tel_q8.first_divergence, tel_q8.top1_preservation,
			tel_q8.logit_rel_l2, tel_q8.delta_nll);
	else
		fprintf(stderr, "quality          unavailable (aligned "
			"comparison pass did not complete)\n");
}

/*
 * Explicit benchmark/compare mode (Section 6): reuses the exact Phase
 * 7 3-pass design (native reference, q8 canonical, q8 teacher-forced)
 * via the SAME run_kv_store_pass()/record_kv_store_behavior() as
 * membrane-llama-run's --kv-store q8 -- including the pass-ordering
 * fix from that phase's own review cycle (the canonical, memory-
 * reported pass runs FIRST, immediately after model load, so its
 * peak-RSS reading is never contaminated by an earlier pass in the
 * same process). Never runs during normal mode.
 */
static int	run_compare_mode(const membrane_run_opts_t &o, llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape)
{
	membrane_kv_store_telemetry_t	tel_q8;
	membrane_kv_store_telemetry_t	tel_scratch;
	gen_run_result_t				result_native;
	gen_run_result_t				result_q8;
	gen_run_result_t				result_q8_tf;
	membrane_runtime_divergence_t	divergence;
	uint64_t						native_bytes;
	int32_t							n_vocab;

	memset(&tel_q8, 0, sizeof(tel_q8));
	tel_q8.kv_store_mode_name = "q8";
	tel_q8.no_fallback_occurred = 1;
	tel_q8.ctx_size = ctx_size;
	tel_q8.compressed_kv_allocated_bytes = q8_kv_bytes(shape, ctx_size);
	native_bytes = native_kv_bytes(shape, ctx_size);
	n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
	membrane_kv_store_read_rss(&tel_q8.rss_after_model_load);
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
			MEMBRANE_KV_STORE_Q8, ctx_size, o.verbose, NULL, false, 0,
			NULL, &tel_q8, &result_q8))
	{
		fprintf(stderr, "membrane-run: compare-mode q8 pass failed\n");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	tel_q8.generated_tokens = result_q8.tokens.size();
	memset(&tel_scratch, 0, sizeof(tel_scratch));
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
			MEMBRANE_KV_STORE_NATIVE, ctx_size, o.verbose, NULL, true,
			n_vocab, NULL, &tel_scratch, &result_native))
	{
		fprintf(stderr, "membrane-run: compare-mode native reference "
			"pass failed\n");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	membrane_runtime_detect_divergence(result_native.tokens.data(),
		result_native.tokens.size(), result_q8.tokens.data(),
		result_q8.tokens.size(), &divergence);
	tel_q8.token_identity = divergence.identical;
	tel_q8.first_divergence = (int32_t)divergence.first_divergence_step;
	if (!result_native.tokens.empty())
	{
		memset(&tel_scratch, 0, sizeof(tel_scratch));
		if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
				MEMBRANE_KV_STORE_Q8, ctx_size, o.verbose,
				&result_native.tokens, true, n_vocab, NULL, &tel_scratch,
				&result_q8_tf))
			fprintf(stderr, "membrane-run: compare-mode aligned "
				"teacher-forced pass failed -- logit/NLL comparison "
				"unavailable, memory result otherwise still "
				"reported\n");
		else if (record_kv_store_behavior(result_native, result_q8_tf,
				n_vocab, &tel_q8))
			tel_q8.quality_available = 1;
	}
	membrane_kv_store_rss_max(&tel_q8.rss_after_model_load,
		&tel_q8.rss_after_context, &tel_q8.rss_peak);
	membrane_kv_store_rss_max(&tel_q8.rss_peak, &tel_q8.rss_after_prompt,
		&tel_q8.rss_peak);
	membrane_kv_store_rss_max(&tel_q8.rss_peak, &tel_q8.rss_final,
		&tel_q8.rss_peak);
	if (o.want_json)
		print_compare_json(model_label, ctx_size, native_bytes, tel_q8);
	else
		print_compare_human(model_label, ctx_size, native_bytes, tel_q8);
	return (result_q8.ok && result_native.ok
		? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

int	main(int argc, char **argv)
{
	membrane_run_opts_t			o;
	int								rc;
	llama_model						*model;
	const llama_vocab				*vocab;
	std::string						prompt_text;
	std::vector<llama_token>		prompt_tokens;
	model_shape_t					shape;
	uint32_t						ctx_size;
	membrane_compat_result_t		compat;
	const char						*model_label;

	rc = membrane_run_parse_opts(argc, argv, &o);
	if (o.want_help)
		return (membrane_run_usage(stdout), MEMBRANE_EXIT_SUCCESS);
	if (o.want_version)
		return (membrane_run_print_version(stdout), MEMBRANE_EXIT_SUCCESS);
	if (rc != MEMBRANE_EXIT_SUCCESS)
		return (membrane_run_usage(stderr), rc);
	if (!resolve_prompt(o, &prompt_text))
	{
		fprintf(stderr, "membrane-run: empty or unreadable prompt\n");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!o.verbose)
		llama_log_set(quiet_log_callback, NULL);
	llama_backend_init();
	model = llama_model_load_from_file(o.model_path,
			llama_model_default_params());
	if (model == NULL)
	{
		fprintf(stderr, "membrane-run: failed to load model '%s'\n",
			o.model_path);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	model_label = membrane_runtime_safe_basename(o.model_path);
	vocab = llama_model_get_vocab(model);
	prompt_tokens.resize(prompt_text.size() + 8);
	rc = llama_tokenize(vocab, prompt_text.c_str(),
			(int32_t)prompt_text.size(), prompt_tokens.data(),
			(int32_t)prompt_tokens.size(), true, false);
	if (rc < 0)
	{
		fprintf(stderr, "membrane-run: tokenization failed\n");
		return (llama_model_free(model), MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	prompt_tokens.resize(rc);
	ctx_size = o.ctx > 0 ? o.ctx
		: (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
	read_model_shape(model, &shape);
	if (o.kv_mode == MEMBRANE_KV_STORE_Q8 || o.compare_kv)
	{
		if (!membrane_check_kv_compat(shape.arch_name.c_str(),
				shape.n_embd, shape.n_head, shape.n_head_kv, ctx_size,
				MEMBRANE_KV_STORE_Q8, &compat))
		{
			fprintf(stderr, "MEMBRANE: Q8 KV storage unsupported for "
				"this model: %s.\n", compat.reason);
			return (llama_model_free(model), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	if (o.compare_kv)
		rc = run_compare_mode(o, model, prompt_tokens, model_label,
				ctx_size, shape);
	else
		rc = run_normal_mode(o, model, prompt_tokens, model_label, ctx_size,
				shape);
	llama_model_free(model);
	llama_backend_free();
	return (rc);
}
