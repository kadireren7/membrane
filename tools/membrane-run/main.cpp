#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
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

/* Phase 9B: what was requested vs what was actually resolved, kept as
 * two separate fields throughout (never collapsed into one) so JSON/
 * human telemetry can honestly distinguish "the user asked for this"
 * from "this is what public ggml_backend_dev_* enumeration found and
 * membrane-run explicitly selected" -- membrane-run never leaves this
 * to llama.cpp's own implicit upstream default (see resolve_gpu_config). */
typedef struct s_membrane_gpu_state
{
	bool		requested;				/* o.gpu_layers != 0 */
	bool		backend_gpu_capable;	/* llama_supports_gpu_offload() */
	int32_t		gpu_layers_requested;	/* o.gpu_layers, echoed (-1=all) */
	std::string	device_requested;		/* o.device, empty if not given */
	std::string	device_selected;		/* empty when CPU-only */
	std::string	backend_selected;		/* "CPU" or e.g. "Vulkan" */
}	membrane_gpu_state_t;

static std::string	ci_lower(const std::string &s)
{
	std::string	out(s);

	for (char &c : out)
		c = (char)std::tolower((unsigned char)c);
	return (out);
}

static std::string	gpu_layers_label(int32_t gpu_layers)
{
	if (gpu_layers < 0)
		return ("all");
	return (std::to_string(gpu_layers));
}

/* Resolves --gpu-layers/--device into an explicit llama_model_params
 * device list using only public ggml_backend_dev_*()/llama_supports_
 * gpu_offload() calls -- mp->devices is never left NULL for llama.cpp's
 * own upstream default to silently pick a GPU (Phase 9B: GPU use must
 * be explicit, not accidental). *device_storage must outlive mp's use
 * in llama_model_load_from_file(). Returns false (message already
 * printed to stderr) if the request cannot be satisfied -- callers
 * must fail closed (MEMBRANE_EXIT_UNSUPPORTED_KV), never silently
 * fall back to CPU. */
static bool	resolve_gpu_config(const membrane_run_opts_t &o,
				std::vector<ggml_backend_dev_t> *device_storage,
				llama_model_params *mp, membrane_gpu_state_t *gs)
{
	std::string	label;
	size_t		i;

	gs->requested = (o.gpu_layers != 0);
	gs->backend_gpu_capable = llama_supports_gpu_offload();
	gs->gpu_layers_requested = o.gpu_layers;
	gs->device_requested = o.want_device ? o.device : "";
	gs->device_selected.clear();
	gs->backend_selected = "CPU";
	if (!gs->requested)
	{
		/* explicit CPU-only: also clear main_gpu so a compiled-in GPU
		 * backend cannot be silently selected by llama.cpp's own
		 * default device list (belt-and-suspenders alongside
		 * n_gpu_layers=0 -- see llama_prepare_model_devices()). */
		mp->n_gpu_layers = 0;
		mp->main_gpu = -1;
		return (true);
	}
	label = gpu_layers_label(o.gpu_layers);
	if (!gs->backend_gpu_capable)
		return (fprintf(stderr, "membrane-run: --gpu-layers %s requested "
				"but this build has no GPU backend compiled in "
				"(rebuild with e.g. -DGGML_VULKAN=ON)\n", label.c_str()),
			false);

	std::vector<ggml_backend_dev_t>	gpu_devs;
	for (i = 0; i < ggml_backend_dev_count(); ++i)
	{
		ggml_backend_dev_t	dev = ggml_backend_dev_get(i);
		enum ggml_backend_dev_type	type = ggml_backend_dev_type(dev);

		if (type == GGML_BACKEND_DEVICE_TYPE_GPU
			|| type == GGML_BACKEND_DEVICE_TYPE_IGPU)
			gpu_devs.push_back(dev);
	}
	if (gpu_devs.empty())
		return (fprintf(stderr, "membrane-run: --gpu-layers %s requested "
				"but no GPU device was found on this host at runtime "
				"(driver/hardware not detected)\n", label.c_str()),
			false);

	ggml_backend_dev_t	chosen;

	if (o.want_device)
	{
		std::vector<ggml_backend_dev_t>	matches;

		/* Match against BOTH the short backend name ("Vulkan1") and
		 * the human-readable hardware description ("NVIDIA GeForce
		 * GTX 1650") -- --device Radeon or --device NVIDIA is what a
		 * user reasonably expects to work, not just the terse name a
		 * bare match against ggml_backend_dev_name() alone would
		 * require. Case-insensitive for the same reason. */
		for (ggml_backend_dev_t dev : gpu_devs)
		{
			std::string	hay = ci_lower(ggml_backend_dev_name(dev))
				+ " " + ci_lower(ggml_backend_dev_description(dev));
			if (hay.find(ci_lower(o.device)) != std::string::npos)
				matches.push_back(dev);
		}
		if (matches.empty())
		{
			fprintf(stderr, "membrane-run: --device \"%s\" matched no "
				"available GPU device. Available:\n", o.device.c_str());
			for (ggml_backend_dev_t dev : gpu_devs)
				fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(dev),
					ggml_backend_dev_description(dev));
			return (false);
		}
		if (matches.size() > 1)
		{
			fprintf(stderr, "membrane-run: --device \"%s\" matched %zu "
				"devices, expected exactly one:\n", o.device.c_str(),
				matches.size());
			for (ggml_backend_dev_t dev : matches)
				fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(dev),
					ggml_backend_dev_description(dev));
			return (false);
		}
		chosen = matches[0];
	}
	else
	{
		/* No --device given: prefer a discrete GPU over an integrated
		 * one, matching llama.cpp's own default-selection preference
		 * (llama_prepare_model_devices()) -- gpu_devs is in raw
		 * ggml_backend_dev_get() enumeration order, which is NOT
		 * discrete-first (confirmed by testing: index 0 was this
		 * host's integrated AMD GPU, index 1 the discrete NVIDIA
		 * one), so this must be sought explicitly. */
		chosen = gpu_devs[0];
		for (ggml_backend_dev_t dev : gpu_devs)
			if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU)
			{
				chosen = dev;
				break;
			}
	}
	device_storage->clear();
	device_storage->push_back(chosen);
	device_storage->push_back(NULL);
	mp->devices = device_storage->data();
	mp->n_gpu_layers = o.gpu_layers;
	mp->main_gpu = 0;
	gs->device_selected = ggml_backend_dev_name(chosen);
	gs->backend_selected = ggml_backend_reg_name(
			ggml_backend_dev_backend_reg(chosen));
	return (true);
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
				uint64_t kv_bytes, const membrane_gpu_state_t &gs)
{
	fprintf(stderr, "MEMBRANE %s\n", MEMBRANE_VERSION);
	fprintf(stderr, "model      %s\n", model_label);
	fprintf(stderr, "context    %u\n", ctx_size);
	fprintf(stderr, "kv         %s\n",
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "Q8_0" : "native (F16)");
	/* run_kv_store_pass() (decode_loop.cpp) forces flash attention
	 * ENABLED unconditionally for both native and q8 -- not just q8 --
	 * so a --compare-kv run never mixes "KV cache dtype" with "which
	 * attention kernel ran" as a second, uncontrolled variable. The
	 * summary must report what actually runs, not a mode-dependent
	 * guess. */
	fprintf(stderr, "flash attn enabled\n");
	fprintf(stderr, "kv bytes   %.2f MiB (%s, from real model "
		"hparams -- not measured until context creation)\n",
		(double)kv_bytes / (1024.0 * 1024.0),
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "Q8_0" : "F16");
	/* Phase 9B: backend/device is a REQUEST membrane-run made via
	 * public ggml_backend_dev_*() enumeration, not a confirmed
	 * post-load allocation (no public API exposes that without log-
	 * scraping) -- worded accordingly, never "confirmed"/"placed". */
	if (!gs.requested)
		fprintf(stderr, "backend    CPU (default -- pass --gpu-layers "
			"to use a GPU)\n");
	else
		fprintf(stderr, "backend    %s, device requested: %s "
			"(gpu-layers=%s)\n", gs.backend_selected.c_str(),
			gs.device_selected.c_str(),
			gpu_layers_label(gs.gpu_layers_requested).c_str());
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

/* RFC 8259 requires every control character (< 0x20) inside a JSON
 * string to be escaped -- a raw tab or newline in generated text
 * previously either passed through unescaped (invalid JSON) or was
 * silently dropped with no replacement (corrupting --include-text's
 * output relative to what the model actually generated). Both are
 * real, reachable bugs: models produce tabs and newlines routinely. */
static void	print_json_escaped(const std::string &text)
{
	for (unsigned char c : text)
	{
		if (c == '"' || c == '\\')
		{
			putchar('\\');
			putchar(c);
		}
		else if (c == '\n')
			fputs("\\n", stdout);
		else if (c == '\r')
			fputs("\\r", stdout);
		else if (c == '\t')
			fputs("\\t", stdout);
		else if (c < 0x20)
			printf("\\u%04x", c);
		else
			putchar(c);
	}
}

static void	print_gpu_json(const membrane_gpu_state_t &gs)
{
	printf("\"gpu\":{\"requested\":%s,\"backend_gpu_capable\":%s,",
		gs.requested ? "true" : "false",
		gs.backend_gpu_capable ? "true" : "false");
	printf("\"gpu_layers_requested\":%d,\"backend\":\"",
		gs.gpu_layers_requested);
	print_json_escaped(gs.backend_selected);
	printf("\",\"device_requested\":\"");
	/* device_requested/device_selected are escaped, unlike backend
	 * (always "CPU" or a ggml backend registry name membrane-run
	 * doesn't control) -- device_requested in particular is raw
	 * --device user input and must not be interpolated unescaped
	 * into JSON (same class of issue print_json_escaped's own
	 * "text" field use already guards against). */
	print_json_escaped(gs.device_requested);
	printf("\",\"device_selected\":\"");
	print_json_escaped(gs.device_selected);
	printf("\"},");
}

static void	print_run_json(const membrane_run_opts_t &o,
				const char *model_label, const membrane_kv_store_telemetry_t &t,
				const std::string &text, const membrane_gpu_state_t &gs)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"run\",\"model_label\":\"%s\",\"kv_store\":\"%s\","
		"\"ctx_size\":%u,\"generated_tokens\":%llu,",
		MEMBRANE_VERSION, model_label,
		o.kv_mode == MEMBRANE_KV_STORE_Q8 ? "q8" : "native",
		t.ctx_size, (unsigned long long)t.generated_tokens);
	printf("\"storage\":{\"kv_allocated_bytes\":%llu},",
		(unsigned long long)t.compressed_kv_allocated_bytes);
	print_gpu_json(gs);
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
		print_json_escaped(text);
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
				const model_shape_t &shape, const membrane_gpu_state_t &gs)
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
		print_startup_summary(o, model_label, ctx_size, kv_bytes, gs);
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
		print_run_json(o, model_label, tel, text, gs);
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
				const membrane_kv_store_telemetry_t &tel_q8,
				const membrane_gpu_state_t &gs)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"compare\",\"model_label\":\"%s\",\"ctx_size\":%u,",
		MEMBRANE_VERSION, model_label, ctx_size);
	printf("\"storage\":{\"native_kv_allocated_bytes\":%llu,"
		"\"q8_kv_allocated_bytes\":%llu},",
		(unsigned long long)native_bytes,
		(unsigned long long)tel_q8.compressed_kv_allocated_bytes);
	print_gpu_json(gs);
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
				const membrane_kv_store_telemetry_t &tel_q8,
				const membrane_gpu_state_t &gs)
{
	fprintf(stderr, "MEMBRANE %s -- compare mode (native vs q8)\n",
		MEMBRANE_VERSION);
	fprintf(stderr, "model            %s\n", model_label);
	fprintf(stderr, "context          %u\n", ctx_size);
	if (!gs.requested)
		fprintf(stderr, "backend          CPU (default)\n");
	else
		fprintf(stderr, "backend          %s, device: %s "
			"(gpu-layers=%s)\n", gs.backend_selected.c_str(),
			gs.device_selected.c_str(),
			gpu_layers_label(gs.gpu_layers_requested).c_str());
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
				const model_shape_t &shape, const membrane_gpu_state_t &gs)
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
		print_compare_json(model_label, ctx_size, native_bytes, tel_q8, gs);
	else
		print_compare_human(model_label, ctx_size, native_bytes, tel_q8, gs);
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
	membrane_gpu_state_t			gs;
	llama_model_params				mp;
	std::vector<ggml_backend_dev_t>	device_storage;

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
	/* GPU/device resolution needs no loaded model -- fail fast, before
	 * spending time on model load, if the request can't be satisfied. */
	mp = llama_model_default_params();
	if (!resolve_gpu_config(o, &device_storage, &mp, &gs))
		return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
	model = llama_model_load_from_file(o.model_path, mp);
	if (model == NULL)
	{
		fprintf(stderr, "membrane-run: failed to load model '%s'\n",
			o.model_path);
		return (llama_backend_free(), MEMBRANE_EXIT_MODEL_ERROR);
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
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	prompt_tokens.resize(rc);
	ctx_size = o.ctx > 0 ? o.ctx
		: (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
	/* An explicit --ctx too small to even hold the prompt previously
	 * surfaced as an opaque "kv-store context creation failed" or
	 * "kv-store prompt decode failed" from deep inside decode_loop.cpp
	 * -- this is the user-facing product CLI, so name the real cause
	 * (the requested --ctx value) instead of an unrelated-sounding
	 * internal failure. */
	if (o.ctx > 0 && (uint64_t)o.ctx <= prompt_tokens.size())
	{
		fprintf(stderr, "membrane-run: --ctx %u is too small for this "
			"prompt (%zu tokens) -- need at least %zu\n", o.ctx,
			prompt_tokens.size(), prompt_tokens.size() + 1);
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_CLI_ERROR);
	}
	read_model_shape(model, &shape);
	if (o.kv_mode == MEMBRANE_KV_STORE_Q8 || o.compare_kv)
	{
		if (!membrane_check_kv_compat(shape.arch_name.c_str(),
				shape.n_embd, shape.n_head, shape.n_head_kv, ctx_size,
				MEMBRANE_KV_STORE_Q8, &compat))
		{
			fprintf(stderr, "MEMBRANE: Q8 KV storage unsupported for "
				"this model: %s.\n", compat.reason);
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	if (o.compare_kv)
		rc = run_compare_mode(o, model, prompt_tokens, model_label,
				ctx_size, shape, gs);
	else
		rc = run_normal_mode(o, model, prompt_tokens, model_label, ctx_size,
				shape, gs);
	llama_model_free(model);
	llama_backend_free();
	return (rc);
}
