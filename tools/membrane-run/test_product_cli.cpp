#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "product_cli.h"
#include "test_helpers.h"

/* membrane_run_opts_t::prompt_file is a raw `const char *` pointing
 * directly into argv -- correct and safe for the real CLI, where argv
 * strings live for the whole process (see product_cli.cpp). Tests that
 * check its string CONTENT after run_parse() returns need that backing
 * memory to outlive the call, unlike the numeric/enum/std::string
 * fields (which are copied by value at parse time). storage/argv are
 * function-static, not stack-local, specifically so a caller can still
 * safely dereference o->prompt_file afterward -- this is a single-
 * threaded test binary where each test checks its own `o` immediately,
 * so reuse across calls is never observed. A stack-local storage
 * vector here previously caused a real heap-use-after-free (caught by
 * ASan) in test_prompt_file_mode. */
static int	run_parse(const std::vector<std::string> &args,
	membrane_run_opts_t *o)
{
	static std::vector<char>	storage;
	std::vector<size_t>			offsets;
	std::vector<char *>			argv;
	size_t						i;

	storage.clear();
	offsets.push_back(0);
	for (i = 0; i < args.size(); i++)
	{
		storage.insert(storage.end(), args[i].begin(), args[i].end());
		storage.push_back('\0');
		offsets.push_back(storage.size());
	}
	for (i = 0; i < args.size(); i++)
		argv.push_back(storage.data() + offsets[i]);
	return (membrane_run_parse_opts((int)argv.size(), argv.data(), o));
}

static std::vector<std::string>	base_args(void)
{
	std::vector<std::string>	v;

	v.push_back("membrane-run");
	v.push_back("--model");
	v.push_back("/no/such/model.gguf");
	v.push_back("--prompt");
	v.push_back("hello");
	return (v);
}

static void	test_minimal_valid_invocation(void)
{
	membrane_run_opts_t	o;

	TEST_ASSERT(run_parse(base_args(), &o) == MEMBRANE_EXIT_SUCCESS,
		"--model + --prompt is a complete, valid invocation");
	TEST_ASSERT(o.kv_mode == 0 /* MEMBRANE_KV_STORE_NATIVE */,
		"default --kv is native -- Section 4: v0.2 never unexpectedly "
		"changes model behavior");
	TEST_ASSERT(o.gen_tokens == 128, "default --gen-tokens is 128");
	TEST_ASSERT(o.ctx == 0, "default --ctx is 0 (auto-size)");
	TEST_ASSERT(o.compare_kv == 0, "--compare-kv is off by default");
	TEST_ASSERT(o.want_json == 0, "--json is off by default");
	TEST_ASSERT(o.prompt_mode == MEMBRANE_RUN_PROMPT_TEXT,
		"--prompt sets TEXT mode");
	TEST_ASSERT(o.prompt_text == "hello",
		"the prompt text itself is stored");
}

static void	test_model_required(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--prompt");
	args.push_back("hi");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"missing --model is a CLI error");
}

static void	test_prompt_required(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"missing --prompt/--prompt-file is a CLI error");
}

static void	test_prompt_and_prompt_file_mutually_exclusive(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--prompt-file");
	args.push_back("/no/such/file.txt");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--prompt and --prompt-file together is a CLI error");
}

static void	test_prompt_dash_is_stdin_mode(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.clear();
	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--prompt");
	args.push_back("-");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--prompt - parses cleanly");
	TEST_ASSERT(o.prompt_mode == MEMBRANE_RUN_PROMPT_STDIN,
		"--prompt - selects stdin mode, not literal text \"-\"");
}

static void	test_prompt_file_mode(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.clear();
	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--prompt-file");
	args.push_back("/no/such/prompt.txt");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--prompt-file alone parses cleanly");
	TEST_ASSERT(o.prompt_mode == MEMBRANE_RUN_PROMPT_FILE,
		"--prompt-file selects FILE mode");
	TEST_ASSERT(std::string(o.prompt_file) == "/no/such/prompt.txt",
		"the prompt file path is stored");
}

static void	test_kv_flag(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv");
	args.push_back("q8");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv q8 accepted");
	TEST_ASSERT(o.kv_mode == 1 /* MEMBRANE_KV_STORE_Q8 */,
		"--kv q8 sets q8 mode");

	args = base_args();
	args.push_back("--kv");
	args.push_back("bogus");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv with an invalid value is a CLI error");
}

static void	test_compare_kv_flag(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--compare-kv accepted");
	TEST_ASSERT(o.compare_kv == 1, "--compare-kv flag is stored");
}

static void	test_ctx_and_gen_tokens_malformed_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("-1");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"negative --ctx rejected (strtoull leading '-' wrap bug class, "
		"guarded via the shared parse_u64_strict helper)");

	args = base_args();
	args.push_back("--gen-tokens");
	args.push_back("abc");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"non-numeric --gen-tokens rejected");

	args = base_args();
	args.push_back("--gen-tokens");
	args.push_back("0");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"zero --gen-tokens rejected");
}

static void	test_ctx_and_gen_tokens_valid_stored(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("4096");
	args.push_back("--gen-tokens");
	args.push_back("256");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"valid --ctx/--gen-tokens accepted");
	TEST_ASSERT(o.ctx == 4096, "--ctx value stored");
	TEST_ASSERT(o.gen_tokens == 256, "--gen-tokens value stored");
}

static void	test_quiet_and_verbose_mutually_exclusive(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--quiet");
	args.push_back("--verbose");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--quiet and --verbose together is a CLI error");
}

static void	test_help_and_version_short_circuit(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.clear();
	args.push_back("membrane-run");
	args.push_back("--help");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--help alone (no --model/--prompt) still parses cleanly");
	TEST_ASSERT(o.want_help == 1, "want_help is set");

	args.clear();
	args.push_back("membrane-run");
	args.push_back("--version");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--version alone still parses cleanly");
	TEST_ASSERT(o.want_version == 1, "want_version is set");
}

static void	test_version_output_format(void)
{
	FILE	*tmp;
	char	buf[256];
	size_t	n;

	tmp = tmpfile();
	TEST_ASSERT(tmp != NULL, "tmpfile() for capturing version output");
	membrane_run_print_version(tmp);
	rewind(tmp);
	n = fread(buf, 1, sizeof(buf) - 1, tmp);
	buf[n] = '\0';
	fclose(tmp);
	TEST_ASSERT(strncmp(buf, "MEMBRANE ", 9) == 0,
		"version output starts with \"MEMBRANE \"");
	TEST_ASSERT(strstr(buf, "0.2.0-rc1") != NULL,
		"version output contains the current version string");
}

static void	test_help_mentions_key_concepts(void)
{
	FILE	*tmp;
	char	buf[8192];
	size_t	n;

	tmp = tmpfile();
	TEST_ASSERT(tmp != NULL, "tmpfile() for capturing help output");
	membrane_run_usage(tmp);
	rewind(tmp);
	n = fread(buf, 1, sizeof(buf) - 1, tmp);
	buf[n] = '\0';
	fclose(tmp);
	TEST_ASSERT(strstr(buf, "--kv") != NULL, "help documents --kv");
	TEST_ASSERT(strstr(buf, "native") != NULL
		&& strstr(buf, "q8") != NULL,
		"help explains both native and q8");
	TEST_ASSERT(strstr(buf, "--compare-kv") != NULL,
		"help documents --compare-kv");
	TEST_ASSERT(strstr(buf, "--ctx") != NULL, "help documents --ctx");
	TEST_ASSERT(strstr(buf, "--json") != NULL, "help documents --json");
	TEST_ASSERT(strstr(buf, "Exit codes") != NULL,
		"help documents exit codes");
	TEST_ASSERT(strstr(buf, "LLM_ARCH_LLAMA") != NULL,
		"help states the architecture-scope limitation");
}

int	main(void)
{
	test_minimal_valid_invocation();
	test_model_required();
	test_prompt_required();
	test_prompt_and_prompt_file_mutually_exclusive();
	test_prompt_dash_is_stdin_mode();
	test_prompt_file_mode();
	test_kv_flag();
	test_compare_kv_flag();
	test_ctx_and_gen_tokens_malformed_rejected();
	test_ctx_and_gen_tokens_valid_stored();
	test_quiet_and_verbose_mutually_exclusive();
	test_help_and_version_short_circuit();
	test_version_output_format();
	test_help_mentions_key_concepts();
	printf("test_product_cli: all tests passed\n");
	return (0);
}
