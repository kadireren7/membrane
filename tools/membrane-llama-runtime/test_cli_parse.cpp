#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cli_parse.h"
#include "test_helpers.h"

/*
 * Exercises the real parse_opts()/argv boundary used by membrane-llama-
 * run's main() -- not just parse_u64_strict() in isolation -- so these
 * tests fail exactly the way the compiled binary would. No model is
 * ever loaded: every case here is decided by parse_opts() alone, before
 * main() would reach llama_model_load_from_file.
 */

static int	run_parse(const std::vector<std::string> &args, run_opts_t *o)
{
	std::vector<char>	storage;
	std::vector<size_t>	offsets;
	std::vector<char *>	argv;
	size_t				i;

	offsets.push_back(0);
	for (i = 0; i < args.size(); i++)
	{
		storage.insert(storage.end(), args[i].begin(), args[i].end());
		storage.push_back('\0');
		offsets.push_back(storage.size());
	}
	/* Re-derive pointers only after all inserts, so reallocation of
	 * `storage` never invalidates an already-taken pointer. */
	for (i = 0; i < args.size(); i++)
		argv.push_back(storage.data() + offsets[i]);
	return (parse_opts((int)argv.size(), argv.data(), o));
}

static std::vector<std::string>	base_args(const char *extra_key = NULL,
	const char *extra_val = NULL)
{
	std::vector<std::string>	v;

	v.push_back("membrane-llama-run");
	v.push_back("--model");
	v.push_back("/no/such/model.gguf");
	v.push_back("--prompt-file");
	v.push_back("/no/such/prompt.txt");
	v.push_back("--mode");
	v.push_back("inject-q8");
	if (extra_key != NULL)
		v.push_back(extra_key);
	if (extra_val != NULL)
		v.push_back(extra_val);
	return (v);
}

static void	expect_rejected(const char *key, const char *val,
	const char *what)
{
	run_opts_t					o;
	std::vector<std::string>	args;
	int							rc;

	args = base_args(key, val);
	rc = run_parse(args, &o);
	TEST_ASSERT(rc != 0, what);
}

static void	expect_accepted(const char *key, const char *val,
	const char *what)
{
	run_opts_t					o;
	std::vector<std::string>	args;
	int							rc;

	args = base_args(key, val);
	rc = run_parse(args, &o);
	TEST_ASSERT(rc == 0, what);
}

/* Sentinel test: a rejected parse_u64_strict() call must not have
 * written *out at all -- it should be left exactly as the caller set
 * it, never a partial/garbage strtoull() result. */
static void	test_parse_u64_strict_leaves_out_unchanged_on_rejection(void)
{
	uint64_t	out;

	out = 0xdeadbeefULL;
	TEST_ASSERT(!parse_u64_strict("-1", &out), "leading '-' rejected");
	TEST_ASSERT(out == 0xdeadbeefULL,
		"*out untouched after a rejected parse (leading '-')");

	out = 0xdeadbeefULL;
	TEST_ASSERT(!parse_u64_strict("abc", &out), "non-numeric rejected");
	TEST_ASSERT(out == 0xdeadbeefULL,
		"*out untouched after a rejected parse (non-numeric)");

	out = 0xdeadbeefULL;
	TEST_ASSERT(!parse_u64_strict(
			"999999999999999999999999999999999999999999", &out),
		"overflow rejected");
	TEST_ASSERT(out == 0xdeadbeefULL,
		"*out untouched after a rejected parse (overflow)");

	out = 0;
	TEST_ASSERT(parse_u64_strict("42", &out), "valid input accepted");
	TEST_ASSERT(out == 42, "*out set to the parsed value on success");
}

static void	test_inject_layer_malformed_rejected(void)
{
	expect_rejected("--inject-layer", "abc", "non-numeric layer rejected");
	expect_rejected("--inject-layer", "3x",
		"trailing garbage on layer rejected");
	expect_rejected("--inject-layer", "-1", "negative layer rejected");
	expect_rejected("--inject-layer", "999999999999999999999",
		"overflowing layer rejected");
	expect_rejected("--inject-layer", "", "empty layer value rejected");
}

static void	test_inject_layer_valid_accepted(void)
{
	run_opts_t	o;

	expect_accepted("--inject-layer", "0", "layer 0 accepted");
	expect_accepted("--inject-layer", "29", "layer 29 accepted");
	expect_accepted("--inject-layer", "511", "layer 511 (max-1) accepted");
	expect_rejected("--inject-layer", "512",
		"layer 512 (== MEMBRANE_RUNTIME_MAX_LAYERS) rejected");
	TEST_ASSERT(run_parse(base_args("--inject-layer", "29"), &o) == 0
		&& o.inject_layers.size() == 1 && o.inject_layers[0] == 29,
		"parsed layer value is actually stored");
}

static void	test_inject_token_start_malformed_rejected(void)
{
	expect_rejected("--inject-token-start", "abc",
		"non-numeric token-start rejected");
	expect_rejected("--inject-token-start", "3x",
		"trailing garbage on token-start rejected");
	expect_rejected("--inject-token-start", "-1",
		"negative token-start rejected (strtoull would otherwise wrap "
		"this to UINT64_MAX)");
	expect_rejected("--inject-token-start", "999999999999999999999999999",
		"overflowing token-start rejected");
}

static void	test_inject_token_end_malformed_rejected(void)
{
	expect_rejected("--inject-token-end", "abc",
		"non-numeric token-end rejected");
	expect_rejected("--inject-token-end", "12foo",
		"trailing garbage on token-end rejected");
	expect_rejected("--inject-token-end", "-1",
		"negative token-end rejected");
	expect_rejected("--inject-token-end", "999999999999999999999999999",
		"overflowing token-end rejected");
}

static void	test_inject_token_range_valid_accepted(void)
{
	std::vector<std::string>	args;
	run_opts_t					o;
	int							rc;

	args = base_args("--inject-token-start", "0");
	args.push_back("--inject-token-end");
	args.push_back("5");
	rc = run_parse(args, &o);
	TEST_ASSERT(rc == 0, "token-start 0 / token-end 5 accepted");
	TEST_ASSERT(o.inject_token_start == 0 && o.inject_token_end == 5,
		"parsed token range is actually stored");

	args = base_args("--inject-token-start", "3");
	args.push_back("--inject-token-end");
	args.push_back("9");
	rc = run_parse(args, &o);
	TEST_ASSERT(rc == 0, "token-start 3 / token-end 9 accepted");
}

static void	test_inject_token_range_relationship_enforced(void)
{
	std::vector<std::string>	args;
	run_opts_t					o;

	args = base_args("--inject-token-start", "9");
	args.push_back("--inject-token-end");
	args.push_back("3");
	TEST_ASSERT(run_parse(args, &o) != 0,
		"token-start > token-end rejected");

	args = base_args("--inject-token-start", "5");
	TEST_ASSERT(run_parse(args, &o) != 0,
		"token-start without token-end rejected");
}

static void	test_gen_tokens_malformed_rejected(void)
{
	expect_rejected("--gen-tokens", "12junk",
		"trailing garbage on --gen-tokens rejected");
	expect_rejected("--gen-tokens", "+12",
		"leading '+' on --gen-tokens rejected");
	expect_rejected("--gen-tokens", "-1",
		"negative --gen-tokens rejected");
	expect_rejected("--gen-tokens", "0",
		"zero --gen-tokens rejected (must be >= 1)");
	expect_rejected("--gen-tokens", "999999999999999999999999",
		"overflowing --gen-tokens rejected");
	expect_rejected("--gen-tokens", "abc",
		"non-numeric --gen-tokens rejected");
}

static void	test_gen_tokens_valid_accepted(void)
{
	run_opts_t	o;

	expect_accepted("--gen-tokens", "1", "--gen-tokens 1 accepted");
	expect_accepted("--gen-tokens", "32", "--gen-tokens 32 (default) "
		"accepted");
	TEST_ASSERT(run_parse(base_args("--gen-tokens", "17"), &o) == 0
		&& o.gen_tokens == 17, "parsed --gen-tokens value is stored");
}

static void	test_debug_perturb_flag_documented(void)
{
	FILE	*tmp;
	char	buf[8192];
	size_t	n;

	tmp = tmpfile();
	TEST_ASSERT(tmp != NULL, "tmpfile() for capturing usage() text");
	usage(tmp);
	rewind(tmp);
	n = fread(buf, 1, sizeof(buf) - 1, tmp);
	buf[n] = '\0';
	fclose(tmp);
	TEST_ASSERT(strstr(buf, "--debug-perturb-injection") != NULL,
		"--debug-perturb-injection appears in usage()/--help output");
	TEST_ASSERT(strstr(buf, "DEBUG ONLY") != NULL,
		"--debug-perturb-injection is labeled DEBUG ONLY, not a normal "
		"production mode");
}

static void	test_valid_baseline_parses_without_model_load(void)
{
	run_opts_t					o;
	std::vector<std::string>	args;

	args.push_back("membrane-llama-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--prompt-file");
	args.push_back("/no/such/prompt.txt");
	args.push_back("--mode");
	args.push_back("baseline");
	TEST_ASSERT(run_parse(args, &o) == 0,
		"a syntactically valid baseline invocation parses cleanly even "
		"though the paths don't exist -- parse_opts() never touches the "
		"filesystem");
}

int	main(void)
{
	test_parse_u64_strict_leaves_out_unchanged_on_rejection();
	test_inject_layer_malformed_rejected();
	test_inject_layer_valid_accepted();
	test_inject_token_start_malformed_rejected();
	test_inject_token_end_malformed_rejected();
	test_inject_token_range_valid_accepted();
	test_inject_token_range_relationship_enforced();
	test_gen_tokens_malformed_rejected();
	test_gen_tokens_valid_accepted();
	test_debug_perturb_flag_documented();
	test_valid_baseline_parses_without_model_load();
	printf("test_cli_parse: all tests passed\n");
	return (0);
}
