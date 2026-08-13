#include "product_cli.h"

#include <cstdio>
#include <cstring>

#include "cli_parse.h"			/* reuses the already-tested
								 * parse_u64_strict() rather than a
								 * second hand-copied strict-integer
								 * parser */
#include "kv_store_telemetry.h"	/* MEMBRANE_KV_STORE_NATIVE/Q8 */
#include "gpu_policy.h"			/* MEMBRANE_GPU_LAYERS_ALL/AUTO -- llama-
								 * free, pure constants, no ggml/llama
								 * dependency pulled in by this include */

void	membrane_run_print_version(FILE *out)
{
	fprintf(out, "MEMBRANE %s\n", MEMBRANE_VERSION);
}

void	membrane_run_usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-run --model MODEL.gguf "
		"(--prompt TEXT | --prompt-file FILE | --prompt -) [options]\n"
		"\n"
		"MEMBRANE %s -- local llama.cpp inference with an optional\n"
		"compressed (Q8_0) KV cache instead of the native full-\n"
		"precision one. Normal runs are a single decode pass: model\n"
		"load, one context, prompt, generation, exit -- no hidden\n"
		"comparison work.\n"
		"\n"
		"Example:\n"
		"  membrane-run --model model.gguf --prompt \"Hello\" \\\n"
		"      --ctx 4096 --kv q8\n"
		"\n"
		"  --model FILE       .gguf model path (required)\n"
		"  --prompt TEXT      prompt text\n"
		"  --prompt-file FILE prompt text file\n"
		"  --prompt -         read the prompt from stdin\n"
		"                     (--prompt/--prompt-file/--prompt - are\n"
		"                     mutually exclusive; exactly one required)\n"
		"  --ctx N            KV cache context size (default: prompt\n"
		"                     length + --gen-tokens + 8)\n"
		"  --kv native|q8     KV cache storage (default: native).\n"
		"                     native: unmodified llama.cpp behavior.\n"
		"                     q8: KV cache tensors are genuinely\n"
		"                     Q8_0-typed (not a shadow copy -- see\n"
		"                     docs/live-runtime.md), roughly half the\n"
		"                     memory of native at the same context\n"
		"                     size. This is real quantization: it can\n"
		"                     shift a greedy generation's exact token\n"
		"                     sequence on long enough runs, though\n"
		"                     aligned quality (top1/logit/NLL) stays\n"
		"                     close in local testing. Checked for\n"
		"                     compatibility before use; fails clearly\n"
		"                     (never silently falls back to native) if\n"
		"                     unsupported for this model.\n"
		"  --gen-tokens N     tokens to generate (default 128)\n"
		"  --threads N        decode thread count (default: let\n"
		"                     llama.cpp choose)\n"
		"  --json             machine-readable output on stdout\n"
		"                     instead of human-readable text\n"
		"  --include-text     JSON only: include the generated text\n"
		"                     (omitted by default)\n"
		"  --quiet            suppress the startup summary and stats\n"
		"                     (generated text still prints)\n"
		"  --verbose          print internal per-step diagnostics\n"
		"  --compare-kv       ADVANCED: run native AND q8 (plus an\n"
		"                     aligned quality comparison pass) and\n"
		"                     report memory/quality/performance\n"
		"                     side by side. Slower and more memory-\n"
		"                     hungry than a normal run by design --\n"
		"                     never implied by --kv q8 alone.\n"
		"  --gpu-layers all|auto|N\n"
		"                     GPU layer offload (default: 0, i.e. CPU-\n"
		"                     only -- explicit CPU-forcing option too;\n"
		"                     never implied by a GPU-capable build).\n"
		"                     \"all\" offloads every layer; \"auto\"\n"
		"                     (Phase 9B.1) picks a safe layer count\n"
		"                     from real device-free-memory and model-\n"
		"                     size information, leaving an explicit\n"
		"                     safety margin (NOT an OOM guarantee); N\n"
		"                     offloads N layers, clamped to the\n"
		"                     model's real layer count if N exceeds it\n"
		"                     (the common -ngl 99-style idiom for\n"
		"                     \"everything\"). all/auto/N alike require\n"
		"                     an explicit --ctx (the KV budget can't\n"
		"                     be estimated before context size is\n"
		"                     known) and, on a build with a GPU\n"
		"                     backend compiled in (e.g.\n"
		"                     -DGGML_VULKAN=ON), fail clearly (never\n"
		"                     silently smaller, never silently CPU) if\n"
		"                     the estimated requirement exceeds a safe\n"
		"                     budget.\n"
		"  --device NAME      select a specific GPU device by a\n"
		"                     case-insensitive substring of its name or\n"
		"                     description (shown if --device matches\n"
		"                     zero or more than one device). Requires\n"
		"                     --gpu-layers all|auto|N. Fails clearly if\n"
		"                     no device matches, or if more than one\n"
		"                     does.\n"
		"  --gpu-bench        explicit native-vs-q8 comparison under a\n"
		"                     GPU configuration (selected device, KV\n"
		"                     bytes, throughput, quality where\n"
		"                     available). Requires --gpu-layers\n"
		"                     all|auto|N. Mutually exclusive with\n"
		"                     --compare-kv.\n"
		"  --version          print version and exit\n"
		"  --help             print this message and exit\n"
		"\n"
		"Exit codes: 0 success, 2 CLI/config error, 3 model load "
		"error,\n"
		"            4 inference error, 5 unsupported KV or GPU/device\n"
		"            configuration (including a --gpu-layers request\n"
		"            that exceeds the estimated safe memory budget).\n"
		"\n"
		"Scope: verified against LLM_ARCH_LLAMA models. CPU backend\n"
		"always supported; GPU (Vulkan) offload is an explicit, opt-in\n"
		"path with measured VRAM/throughput/quality effects that\n"
		"depend on model, context, and host GPU -- see README.\n"
		"See tools/membrane-llama-runtime/ (membrane-llama-run) for\n"
		"the diagnostic shadow/injection research tooling this is\n"
		"built on top of.\n", MEMBRANE_VERSION);
}

static bool	parse_u32(const char *s, uint32_t *out, const char *flag_name)
{
	uint64_t	val;

	if (!parse_u64_strict(s, &val) || val < 1 || val > UINT32_MAX)
		return (fprintf(stderr, "membrane-run: %s must be an integer in "
				"[1, %u]\n", flag_name, UINT32_MAX), false);
	*out = (uint32_t)val;
	return (true);
}

/* "all" (every layer), "auto" (Phase 9B.1: MEMBRANE picks a safe
 * count from real device/model memory information), or a non-negative
 * layer count. 0 is a valid, meaningful value here (explicit
 * CPU-only) -- unlike parse_u32's flags, so this does not reuse it. */
static bool	parse_gpu_layers(const char *s, int32_t *out)
{
	uint64_t	val;

	if (strcmp(s, "all") == 0)
		return (*out = MEMBRANE_GPU_LAYERS_ALL, true);
	if (strcmp(s, "auto") == 0)
		return (*out = MEMBRANE_GPU_LAYERS_AUTO, true);
	if (!parse_u64_strict(s, &val) || val > (uint64_t)INT32_MAX)
		return (fprintf(stderr, "membrane-run: --gpu-layers must be "
				"\"all\", \"auto\", or an integer in [0, %d]\n",
				INT32_MAX), false);
	*out = (int32_t)val;
	return (true);
}

int	membrane_run_parse_opts(int argc, char **argv, membrane_run_opts_t *o)
{
	int	i;

	o->model_path = NULL;
	o->prompt_mode = MEMBRANE_RUN_PROMPT_NONE;
	o->prompt_text.clear();
	o->prompt_file = NULL;
	o->ctx = 0;
	o->kv_mode = MEMBRANE_KV_STORE_NATIVE;
	o->gen_tokens = 128;
	o->threads = 0;
	o->want_json = 0;
	o->quiet = 0;
	o->verbose = 0;
	o->include_text = 0;
	o->compare_kv = 0;
	o->gpu_layers = 0;
	o->device.clear();
	o->want_device = 0;
	o->gpu_bench = 0;
	o->want_version = 0;
	o->want_help = 0;
	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--help") == 0)
			return (o->want_help = 1, MEMBRANE_EXIT_SUCCESS);
		else if (strcmp(argv[i], "--version") == 0)
			return (o->want_version = 1, MEMBRANE_EXIT_SUCCESS);
		else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
			o->model_path = argv[++i];
		else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
		{
			++i;
			if (o->prompt_mode != MEMBRANE_RUN_PROMPT_NONE)
				return (fprintf(stderr, "membrane-run: --prompt and "
						"--prompt-file are mutually exclusive\n"),
					MEMBRANE_EXIT_CLI_ERROR);
			if (strcmp(argv[i], "-") == 0)
				o->prompt_mode = MEMBRANE_RUN_PROMPT_STDIN;
			else
			{
				o->prompt_mode = MEMBRANE_RUN_PROMPT_TEXT;
				o->prompt_text = argv[i];
			}
		}
		else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc)
		{
			++i;
			if (o->prompt_mode != MEMBRANE_RUN_PROMPT_NONE)
				return (fprintf(stderr, "membrane-run: --prompt and "
						"--prompt-file are mutually exclusive\n"),
					MEMBRANE_EXIT_CLI_ERROR);
			o->prompt_mode = MEMBRANE_RUN_PROMPT_FILE;
			o->prompt_file = argv[i];
		}
		else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc)
		{
			++i;
			if (!parse_u32(argv[i], &o->ctx, "--ctx"))
				return (MEMBRANE_EXIT_CLI_ERROR);
		}
		else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc)
		{
			++i;
			if (strcmp(argv[i], "native") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_NATIVE;
			else if (strcmp(argv[i], "q8") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_Q8;
			else
				return (fprintf(stderr,
						"membrane-run: --kv must be native or q8\n"),
					MEMBRANE_EXIT_CLI_ERROR);
		}
		else if (strcmp(argv[i], "--gen-tokens") == 0 && i + 1 < argc)
		{
			uint32_t	val;

			++i;
			if (!parse_u32(argv[i], &val, "--gen-tokens")
				|| val > (uint32_t)INT32_MAX)
				return (MEMBRANE_EXIT_CLI_ERROR);
			o->gen_tokens = (int)val;
		}
		else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
		{
			uint32_t	val;

			++i;
			if (!parse_u32(argv[i], &val, "--threads")
				|| val > (uint32_t)INT32_MAX)
				return (MEMBRANE_EXIT_CLI_ERROR);
			o->threads = (int)val;
		}
		else if (strcmp(argv[i], "--json") == 0)
			o->want_json = 1;
		else if (strcmp(argv[i], "--include-text") == 0)
			o->include_text = 1;
		else if (strcmp(argv[i], "--quiet") == 0)
			o->quiet = 1;
		else if (strcmp(argv[i], "--verbose") == 0)
			o->verbose = 1;
		else if (strcmp(argv[i], "--compare-kv") == 0)
			o->compare_kv = 1;
		else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc)
		{
			++i;
			if (!parse_gpu_layers(argv[i], &o->gpu_layers))
				return (MEMBRANE_EXIT_CLI_ERROR);
		}
		else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
		{
			++i;
			if (argv[i][0] == '\0')
				return (fprintf(stderr, "membrane-run: --device requires a "
						"non-empty device name substring\n"),
					MEMBRANE_EXIT_CLI_ERROR);
			o->want_device = 1;
			o->device = argv[i];
		}
		else if (strcmp(argv[i], "--gpu-bench") == 0)
			o->gpu_bench = 1;
		else
			return (fprintf(stderr, "membrane-run: unknown option: %s\n",
					argv[i]), MEMBRANE_EXIT_CLI_ERROR);
		i++;
	}
	if (o->model_path == NULL)
		return (fprintf(stderr, "membrane-run: --model is required\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->prompt_mode == MEMBRANE_RUN_PROMPT_NONE)
		return (fprintf(stderr, "membrane-run: one of --prompt, "
				"--prompt-file, or --prompt - is required\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->quiet && o->verbose)
		return (fprintf(stderr,
				"membrane-run: --quiet and --verbose are mutually "
				"exclusive\n"), MEMBRANE_EXIT_CLI_ERROR);
	if (o->include_text && !o->want_json)
		return (fprintf(stderr,
				"membrane-run: --include-text requires --json -- it has "
				"no defined effect on human-readable output (which "
				"already prints generated text by default)\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->want_device && o->gpu_layers == 0)
		return (fprintf(stderr,
				"membrane-run: --device requires --gpu-layers to be "
				"\"all\" or a positive count -- naming a device with "
				"zero GPU layers requested is ambiguous\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->gpu_layers != 0 && o->ctx == 0)
		return (fprintf(stderr,
				"membrane-run: --gpu-layers all|auto|N requires an "
				"explicit --ctx N -- the GPU memory guard can't check "
				"the real KV budget before the context size is known, "
				"and auto-sizing --ctx from the prompt requires the "
				"model already loaded (a decision GPU resolution needs "
				"to make first, before load)\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->gpu_bench && o->compare_kv)
		return (fprintf(stderr,
				"membrane-run: --gpu-bench and --compare-kv are both "
				"explicit benchmark modes -- mutually exclusive\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->gpu_bench && o->gpu_layers == 0)
		return (fprintf(stderr,
				"membrane-run: --gpu-bench requires --gpu-layers to be "
				"\"all\", \"auto\", or a positive count -- it compares "
				"native vs q8 under a GPU configuration, not CPU-only\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	return (MEMBRANE_EXIT_SUCCESS);
}
