#include "cli_parse.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

/*
 * Strict base-10 uint64 parse: rejects anything strtoull() alone would
 * silently accept, notably a leading '-' (strtoull negates and wraps
 * mod 2^64 instead of failing -- e.g. "-1" becomes UINT64_MAX with
 * errno left at 0), a leading '+', empty input, trailing garbage, and
 * overflow. Assigns *out only once every check has passed.
 */
bool	parse_u64_strict(const char *s, uint64_t *out)
{
	char		*end;
	uint64_t	value;

	if (s == NULL || s[0] == '\0' || s[0] == '-' || s[0] == '+')
		return (false);
	errno = 0;
	value = strtoull(s, &end, 10);
	if (errno == ERANGE || end == s || *end != '\0')
		return (false);
	*out = value;
	return (true);
}

void	usage(FILE *out)
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
		"                         MEMBRANE K/V processing\n"
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
		"  --kv-store MODE    native | q8 (default native; requires\n"
		"                     --mode baseline). Product Phase 7: the KV\n"
		"                     CACHE TENSOR ITSELF is allocated at the\n"
		"                     given ggml type (q8: genuinely Q8_0,\n"
		"                     roughly half the bytes of native F16 --\n"
		"                     not a shadow/injected copy). q8 forces\n"
		"                     flash attention on (required by llama.cpp\n"
		"                     for quantized V cache). See\n"
		"                     docs/live-runtime.md.\n"
		"  --ctx N            explicit KV cache context size (default:\n"
		"                     auto-sized to prompt length + --gen-tokens\n"
		"                     + 8, matching prior behavior). Use an\n"
		"                     explicit value to compare memory at a\n"
		"                     fixed context size independent of prompt/\n"
		"                     generation length.\n"
		"  --help             print this message and exit\n");
}

int	parse_opts(int argc, char **argv, run_opts_t *o)
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
	o->have_kv_store = 0;
	o->kv_store_mode = MEMBRANE_KV_STORE_NATIVE;
	o->kv_store_ctx = 0;
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
			uint64_t	val;

			++i;
			if (!parse_u64_strict(argv[i], &val) || val < 1
				|| val > (uint64_t)INT_MAX)
				return (fprintf(stderr,
						"--gen-tokens must be an integer in [1, %d]\n",
						INT_MAX), -1);
			o->gen_tokens = (int)val;
		}
		else if (strcmp(argv[i], "--inject-layer") == 0 && i + 1 < argc)
		{
			uint64_t	val;

			++i;
			if (!parse_u64_strict(argv[i], &val)
				|| val >= MEMBRANE_RUNTIME_MAX_LAYERS)
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
			++i;
			if (!parse_u64_strict(argv[i], &o->inject_token_start))
				return (fprintf(stderr, "--inject-token-start must be a "
						"non-negative integer\n"), -1);
			o->have_token_start = 1;
		}
		else if (strcmp(argv[i], "--inject-token-end") == 0 && i + 1 < argc)
		{
			++i;
			if (!parse_u64_strict(argv[i], &o->inject_token_end))
				return (fprintf(stderr, "--inject-token-end must be a "
						"non-negative integer\n"), -1);
			o->have_token_end = 1;
		}
		else if (strcmp(argv[i], "--debug-perturb-injection") == 0)
			o->debug_perturb_injection = 1;
		else if (strcmp(argv[i], "--kv-store") == 0 && i + 1 < argc)
		{
			++i;
			o->have_kv_store = 1;
			if (strcmp(argv[i], "native") == 0)
				o->kv_store_mode = MEMBRANE_KV_STORE_NATIVE;
			else if (strcmp(argv[i], "q8") == 0)
				o->kv_store_mode = MEMBRANE_KV_STORE_Q8;
			else
				return (fprintf(stderr,
						"--kv-store must be native or q8\n"), -1);
		}
		else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc)
		{
			uint64_t	val;

			++i;
			if (!parse_u64_strict(argv[i], &val) || val < 1
				|| val > UINT32_MAX)
				return (fprintf(stderr,
						"--ctx must be an integer in [1, %u]\n",
						UINT32_MAX), -1);
			o->kv_store_ctx = (uint32_t)val;
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
	if (o->have_token_start != o->have_token_end)
		return (fprintf(stderr, "--inject-token-start and "
				"--inject-token-end must be given together\n"), -1);
	if (o->have_token_start && o->inject_token_start > o->inject_token_end)
		return (fprintf(stderr, "--inject-token-start must be <= "
				"--inject-token-end\n"), -1);
	if (!membrane_runtime_mode_is_inject(o->mode)
		&& (!o->inject_layers.empty()
			|| o->inject_tensor != MEMBRANE_RUNTIME_TENSOR_BOTH
			|| o->have_token_start || o->debug_perturb_injection))
		return (fprintf(stderr, "--inject-* flags require --mode "
				"inject-q8 or inject-adaptive\n"), -1);
	if (o->kv_store_mode != MEMBRANE_KV_STORE_NATIVE
		&& o->mode != MEMBRANE_RUNTIME_MODE_BASELINE)
		return (fprintf(stderr, "--kv-store q8 requires --mode baseline "
				"-- Phase 7 compressed storage is a distinct mechanism "
				"from Phase 6 injection, never combined\n"), -1);
	return (0);
}
