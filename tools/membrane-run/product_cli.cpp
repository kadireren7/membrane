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
#include "kv_residency_policy.h"	/* MEMBRANE_KV_PLACEMENT_DEFAULT/GPU/
									 * CPU/AUTO -- also llama-free */

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
		"compressed (Q8_0 or Q5_1) KV cache instead of the native\n"
		"full-precision one. Normal runs are a single decode pass:\n"
		"model load, one context, prompt, generation, exit -- no\n"
		"hidden comparison work.\n"
		"\n"
		"Example:\n"
		"  membrane-run --model model.gguf --prompt \"Hello\" \\\n"
		"      --ctx 4096 --kv q8\n"
		"\n"
		"MODEL\n"
		"  --model FILE       .gguf model path (required)\n"
		"  --prompt TEXT      prompt text\n"
		"  --prompt-file FILE prompt text file\n"
		"  --prompt -         read the prompt from stdin\n"
		"                     (--prompt/--prompt-file/--prompt - are\n"
		"                     mutually exclusive; exactly one required)\n"
		"\n"
		"GENERATION\n"
		"  --ctx N            KV cache context size (default: prompt\n"
		"                     length + --gen-tokens + 8)\n"
		"  --gen-tokens N     tokens to generate (default 128)\n"
		"  --threads N        decode thread count (default: let\n"
		"                     llama.cpp choose)\n"
		"\n"
		"KV PRECISION -- controls KV cache storage TYPE (memory/quality),\n"
		"a SEPARATE dimension from KV PLACEMENT below (which controls\n"
		"DEVICE residency only -- see that section).\n"
		"  --kv native|q8|q5|adaptive\n"
		"                     KV cache storage (default: native).\n"
		"                     native: unmodified llama.cpp behavior.\n"
		"                     q8: KV cache tensors are genuinely\n"
		"                     Q8_0-typed (not a shadow copy -- see\n"
		"                     docs/live-runtime.md), roughly half the\n"
		"                     memory of native at the same context\n"
		"                     size. The safer, quality-first\n"
		"                     compressed mode.\n"
		"                     q5: KV cache tensors are genuinely\n"
		"                     Q5_1-typed (not a shadow copy), roughly\n"
		"                     37.5%% of native's memory (a computed\n"
		"                     additional ~29%% reduction beyond q8,\n"
		"                     from ggml block sizes).\n"
		"                     A memory/quality middle ground: saves\n"
		"                     more than q8 but carries more quality\n"
		"                     risk -- validated across a broad\n"
		"                     multi-category prompt set (see\n"
		"                     docs/live-runtime.md and\n"
		"                     results/v0.4/q5-validation.json) but not\n"
		"                     to the same depth as q8. Prefer q8 when\n"
		"                     quality matters most; prefer q5 when\n"
		"                     memory/VRAM is the binding constraint.\n"
		"                     adaptive: MEMBRANE picks exactly ONE of\n"
		"                     q8 or q5 for the entire context (never a\n"
		"                     per-layer/per-block mix) based on\n"
		"                     memory pressure / GPU budget / requested\n"
		"                     context -- q8 whenever it safely fits\n"
		"                     with the same practical residency as\n"
		"                     q5; q5 only when q8 would lose full GPU\n"
		"                     residency, fail the memory guard, or\n"
		"                     exceed --kv-budget-mib, while q5 still\n"
		"                     fits safely. Never picks q5 merely\n"
		"                     because it is smaller. The resolved\n"
		"                     mode and machine-readable reason are\n"
		"                     always reported (requested_kv/\n"
		"                     selected_kv/adaptive_reason in --json,\n"
		"                     shown in human output too) -- never a\n"
		"                     silent fallback to native. See README\n"
		"                     for the full policy and its limitations\n"
		"                     (point-in-time GPU memory snapshot, not\n"
		"                     an OOM guarantee).\n"
		"                     Both q8 and q5 are real quantization: it\n"
		"                     can shift a greedy generation's exact\n"
		"                     token sequence on long enough runs.\n"
		"                     All compressed modes are checked for\n"
		"                     compatibility before use; fail clearly\n"
		"                     (never silently fall back to native) if\n"
		"                     unsupported for this model, or if\n"
		"                     adaptive finds no mode that fits.\n"
		"  --kv-budget-mib N  ADVANCED: a hard memory-budget input to\n"
		"                     --kv adaptive's own policy (CPU and\n"
		"                     GPU) -- a candidate mode whose KV bytes\n"
		"                     exceed this is treated as not fitting.\n"
		"                     Requires --kv adaptive; never alters an\n"
		"                     explicit native/q8/q5 choice.\n"
		"\n"
		"GPU OFFLOAD\n"
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
		"\n"
		"KV PLACEMENT -- controls KV cache DEVICE residency, a SEPARATE\n"
		"dimension from KV PRECISION above (which controls storage TYPE\n"
		"only -- see that section).\n"
		"  --kv-placement default|gpu|cpu|auto\n"
		"                     STATIC KV cache device residency (default:\n"
		"                     default -- unmodified behavior, KV follows\n"
		"                     whatever device llama.cpp itself would\n"
		"                     already use; never changes unless this\n"
		"                     flag is explicitly given). A SEPARATE\n"
		"                     dimension from --kv: this never changes KV\n"
		"                     precision, --kv never changes KV device\n"
		"                     placement. The whole map is decided once,\n"
		"                     before context construction -- no\n"
		"                     movement during inference (see\n"
		"                     docs/kv-residency.md).\n"
		"                     gpu: every KV layer on the same GPU device\n"
		"                     as the offloaded model weights, or fail\n"
		"                     closed if it does not safely fit.\n"
		"                     cpu: every KV layer in system RAM, model\n"
		"                     weights unaffected (still governed by\n"
		"                     --gpu-layers).\n"
		"                     auto: MEMBRANE's planner keeps as many KV\n"
		"                     layers GPU-resident as safely fit (in\n"
		"                     ascending layer-index order) and places\n"
		"                     the rest in system RAM -- chosen for\n"
		"                     conservative compatibility with prior\n"
		"                     behavior, NOT because GPU-resident KV was\n"
		"                     measured faster (it was not -- see\n"
		"                     docs/kv-residency.md). Never fails: an\n"
		"                     all-CPU-KV plan is a valid auto outcome.\n"
		"                     gpu/auto require --gpu-layers all|auto|N\n"
		"                     (there is no GPU device to place KV on\n"
		"                     otherwise); cpu and default are always\n"
		"                     valid, including CPU-only builds.\n"
		"\n"
		"AUTOMATIC POLICY\n"
		"  --auto             MEMBRANE-managed convenience preset --\n"
		"                     equivalent to --gpu-layers auto --kv\n"
		"                     adaptive --kv-placement auto, applied\n"
		"                     ONLY to whichever of those three you do\n"
		"                     NOT also pass explicitly (explicit flags\n"
		"                     always win, e.g. \"--auto --kv q8\" lets\n"
		"                     --auto manage GPU layers and placement\n"
		"                     while precision stays explicitly q8).\n"
		"                     Not a second/parallel planner: resolves\n"
		"                     into the exact same policy pipeline the\n"
		"                     equivalent explicit flags already use --\n"
		"                     see PLAN OUTPUT below and --json's\n"
		"                     \"reason\" fields for what it actually\n"
		"                     chose and why. Still requires an\n"
		"                     explicit --ctx (same reason as\n"
		"                     --gpu-layers auto above). Gracefully\n"
		"                     behaves as CPU-only (never requires a\n"
		"                     GPU backend or device) when none is\n"
		"                     available -- unlike an explicit\n"
		"                     --gpu-layers auto/all/N, which still\n"
		"                     fails closed if a GPU was truly asked\n"
		"                     for and isn't there. Without --auto,\n"
		"                     nothing changes: the default remains\n"
		"                     --gpu-layers 0 --kv native --kv-placement\n"
		"                     default, exactly as before this flag\n"
		"                     existed.\n"
		"\n"
		"OUTPUT / DIAGNOSTICS\n"
		"  --json             machine-readable output on stdout\n"
		"                     instead of human-readable text\n"
		"  --include-text     JSON only: include the generated text\n"
		"                     (omitted by default)\n"
		"  --quiet            suppress the startup summary and stats\n"
		"                     (generated text still prints)\n"
		"  --verbose          print internal per-step diagnostics\n"
		"  --compare-kv       ADVANCED: run native AND a compressed mode\n"
		"                     (plus an aligned quality comparison\n"
		"                     pass) and report memory/quality/\n"
		"                     performance side by side. The compressed\n"
		"                     mode compared is q5 if --kv q5 was also\n"
		"                     given, q8 (the default, for backward\n"
		"                     compatibility) otherwise -- unless --kv\n"
		"                     adaptive was given, in which case\n"
		"                     adaptive resolves ONCE and that same\n"
		"                     resolved mode (q8 or q5) is what native\n"
		"                     is compared against. Never a 4-way\n"
		"                     comparison, always native vs exactly one\n"
		"                     selected compressed mode. Slower and\n"
		"                     more memory-hungry than a normal run by\n"
		"                     design -- never implied by --kv alone.\n"
		"  --gpu-bench        explicit native-vs-compressed comparison\n"
		"                     under a GPU configuration (selected\n"
		"                     device, KV bytes, throughput, quality\n"
		"                     where available). Compares native vs q5\n"
		"                     if --kv q5 was also given, q8 (the\n"
		"                     default, for backward compatibility)\n"
		"                     otherwise -- same one-selected-mode rule\n"
		"                     as --compare-kv, including --kv adaptive\n"
		"                     resolving once beforehand. Requires\n"
		"                     --gpu-layers all|auto|N. Mutually\n"
		"                     exclusive with --compare-kv.\n"
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
	o->want_kv_budget = 0;
	o->kv_budget_bytes = 0;
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
	o->kv_placement = MEMBRANE_KV_PLACEMENT_DEFAULT;
	o->auto_mode = 0;
	o->want_gpu_layers = 0;
	o->want_kv_mode = 0;
	o->want_kv_placement = 0;
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
			o->want_kv_mode = 1;
			if (strcmp(argv[i], "native") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_NATIVE;
			else if (strcmp(argv[i], "q8") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_Q8;
			else if (strcmp(argv[i], "q5") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_Q5;
			else if (strcmp(argv[i], "adaptive") == 0)
				o->kv_mode = MEMBRANE_KV_STORE_ADAPTIVE;
			else
				return (fprintf(stderr, "membrane-run: --kv must be "
						"native, q8, q5, or adaptive\n"),
					MEMBRANE_EXIT_CLI_ERROR);
		}
		else if (strcmp(argv[i], "--kv-budget-mib") == 0 && i + 1 < argc)
		{
			uint64_t	val;

			++i;
			if (!parse_u64_strict(argv[i], &val) || val < 1
				|| val > (UINT64_MAX / (1024 * 1024)))
				return (fprintf(stderr, "membrane-run: --kv-budget-mib "
						"must be a positive integer number of MiB\n"),
					MEMBRANE_EXIT_CLI_ERROR);
			o->want_kv_budget = 1;
			o->kv_budget_bytes = val * 1024 * 1024;
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
			o->want_gpu_layers = 1;
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
		else if (strcmp(argv[i], "--auto") == 0)
			o->auto_mode = 1;
		else if (strcmp(argv[i], "--kv-placement") == 0 && i + 1 < argc)
		{
			++i;
			o->want_kv_placement = 1;
			if (strcmp(argv[i], "default") == 0)
				o->kv_placement = MEMBRANE_KV_PLACEMENT_DEFAULT;
			else if (strcmp(argv[i], "gpu") == 0)
				o->kv_placement = MEMBRANE_KV_PLACEMENT_GPU;
			else if (strcmp(argv[i], "cpu") == 0)
				o->kv_placement = MEMBRANE_KV_PLACEMENT_CPU;
			else if (strcmp(argv[i], "auto") == 0)
				o->kv_placement = MEMBRANE_KV_PLACEMENT_AUTO;
			else
				return (fprintf(stderr, "membrane-run: --kv-placement "
						"must be default, gpu, cpu, or auto\n"),
					MEMBRANE_EXIT_CLI_ERROR);
		}
		else
			return (fprintf(stderr, "membrane-run: unknown option: %s\n",
					argv[i]), MEMBRANE_EXIT_CLI_ERROR);
		i++;
	}
	/* Phase 13.1, Section 7-9: --auto fills in ONLY the fields the user
	 * did not also explicitly pass, then falls straight through into
	 * every validation check below exactly as if the user had typed
	 * the equivalent explicit flags -- this is a preset applied once,
	 * before validation, never a separate decision path (Section 7:
	 * "Internally it should resolve into the same existing policy
	 * pipeline"). Placed here (not earlier, inline in the parse loop)
	 * so it runs after every explicit flag has been seen, regardless
	 * of argv order -- `--kv q8 --auto` and `--auto --kv q8` behave
	 * identically. */
	if (o->auto_mode)
	{
		if (!o->want_gpu_layers)
			o->gpu_layers = MEMBRANE_GPU_LAYERS_AUTO;
		if (!o->want_kv_mode)
			o->kv_mode = MEMBRANE_KV_STORE_ADAPTIVE;
		/* gpu_layers is already fully resolved (explicit override or
		 * the auto-fill two lines above) by the time this runs --
		 * Section 9/15: an explicit `--auto --gpu-layers 0` must not
		 * be rejected by the gpu|auto-requires-nonzero-gpu_layers check
		 * a few lines below just because --auto's OWN default would
		 * otherwise have picked "auto" placement; default (not auto)
		 * is the correct, sensible CPU-only outcome instead. */
		if (!o->want_kv_placement)
			o->kv_placement = o->gpu_layers != 0
				? MEMBRANE_KV_PLACEMENT_AUTO : MEMBRANE_KV_PLACEMENT_DEFAULT;
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
	if (o->want_kv_budget && o->kv_mode != MEMBRANE_KV_STORE_ADAPTIVE)
		return (fprintf(stderr,
				"membrane-run: --kv-budget-mib requires --kv adaptive -- "
				"it is a policy input to adaptive's own Q8-vs-Q5 "
				"decision and never silently alters an explicit "
				"native/q8/q5 choice\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if ((o->kv_placement == MEMBRANE_KV_PLACEMENT_GPU
			|| o->kv_placement == MEMBRANE_KV_PLACEMENT_AUTO)
		&& o->gpu_layers == 0)
		return (fprintf(stderr,
				"membrane-run: --kv-placement gpu|auto requires "
				"--gpu-layers all|auto|N -- there is no GPU device to "
				"place KV on with CPU-only weight placement\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	if (o->kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT
		&& (o->compare_kv || o->gpu_bench))
		return (fprintf(stderr,
				"membrane-run: --kv-placement is not yet supported "
				"together with --compare-kv/--gpu-bench (Phase 12H "
				"scope: a normal single-pass run only) -- rejected "
				"rather than silently ignored\n"),
			MEMBRANE_EXIT_CLI_ERROR);
	return (MEMBRANE_EXIT_SUCCESS);
}
