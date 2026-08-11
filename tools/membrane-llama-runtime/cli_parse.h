#ifndef MEMBRANE_LLAMA_RUNTIME_CLI_PARSE_H
# define MEMBRANE_LLAMA_RUNTIME_CLI_PARSE_H

#include <cstdio>
#include <cstdint>
#include <vector>

#include "runtime_core.h"

/*
 * membrane-llama-run's CLI parsing, split out from main.cpp so it can be
 * unit-tested (test_cli_parse.cpp) without linking llama.cpp or loading
 * a model -- this file and its .cpp only depend on runtime_core.h, the
 * llama-free MEMBRANE-side header, same as runtime_core.c itself.
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

bool	parse_u64_strict(const char *s, uint64_t *out);
void	usage(FILE *out);
int		parse_opts(int argc, char **argv, run_opts_t *o);

#endif
