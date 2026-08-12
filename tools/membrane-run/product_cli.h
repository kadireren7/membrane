#ifndef MEMBRANE_RUN_PRODUCT_CLI_H
# define MEMBRANE_RUN_PRODUCT_CLI_H

#include <cstdint>
#include <cstdio>
#include <string>

/*
 * membrane-run's product-facing CLI, llama-free and testable without
 * a model (test_product_cli.cpp) -- deliberately a SEPARATE parser
 * from tools/membrane-llama-runtime/cli_parse.h, not an extension of
 * it: that one's flag surface (shadow/inject modes, debug-perturbation
 * flags) is diagnostic/research, this one is the clean product surface
 * Section 3 asks for. Both link the same underlying decode loop
 * (tools/membrane-llama-runtime/decode_loop.h).
 */

/* Stable exit codes (Section 8) -- never change the meaning of an
 * already-shipped code, only add new ones. */
# define MEMBRANE_EXIT_SUCCESS			0
# define MEMBRANE_EXIT_CLI_ERROR		2
# define MEMBRANE_EXIT_MODEL_ERROR		3
# define MEMBRANE_EXIT_RUNTIME_ERROR	4
# define MEMBRANE_EXIT_UNSUPPORTED_KV	5

typedef enum e_membrane_run_prompt_mode
{
	MEMBRANE_RUN_PROMPT_NONE = 0,
	MEMBRANE_RUN_PROMPT_TEXT,		/* --prompt "..." */
	MEMBRANE_RUN_PROMPT_FILE,		/* --prompt-file FILE */
	MEMBRANE_RUN_PROMPT_STDIN		/* --prompt - */
}	membrane_run_prompt_mode_t;

typedef struct s_membrane_run_opts
{
	const char					*model_path;
	membrane_run_prompt_mode_t	prompt_mode;
	std::string					prompt_text;	/* PROMPT_TEXT only */
	const char					*prompt_file;	/* PROMPT_FILE only */

	uint32_t	ctx;			/* 0 = auto-size to prompt + gen_tokens + 8 */
	int			kv_mode;		/* MEMBRANE_KV_STORE_NATIVE/Q8 (default
								 * NATIVE -- Section 4: v0.2 never
								 * unexpectedly changes model behavior) */
	int			gen_tokens;
	int			threads;		/* 0 = let llama.cpp pick its own default */

	int			want_json;
	int			quiet;
	int			verbose;
	int			include_text;	/* JSON only: include generated text */
	int			compare_kv;		/* Section 6: explicit benchmark/compare
								 * mode, reuses the Phase 7 3-pass
								 * machinery -- never the default */

	int			want_version;
	int			want_help;
}	membrane_run_opts_t;

void	membrane_run_usage(FILE *out);
void	membrane_run_print_version(FILE *out);

/* Returns MEMBRANE_EXIT_SUCCESS on success, MEMBRANE_EXIT_CLI_ERROR on
 * any parse/validation failure (message already printed to stderr).
 * want_help/want_version are checked by the caller BEFORE treating a
 * nonzero return as fatal -- --help/--version short-circuit before
 * --model is required, matching ordinary CLI convention. */
int		membrane_run_parse_opts(int argc, char **argv,
			membrane_run_opts_t *o);

#endif
