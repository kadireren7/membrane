#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "product_cli.h"
#include "gpu_policy.h"
#include "kv_residency_policy.h"
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
	args.push_back("q5");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv q5 accepted");
	TEST_ASSERT(o.kv_mode == 2 /* MEMBRANE_KV_STORE_Q5 */,
		"--kv q5 sets q5 mode (maps to Q5_1 internally, see decode_loop.cpp)");

	args = base_args();
	args.push_back("--kv");
	args.push_back("bogus");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv with an invalid value is a CLI error");
}

/* Phase 10C Section 18: the experimental Q5_0/Q4 aliases that existed
 * on the research branches (experiment/q5-kv-evaluation,
 * experiment/q4-kv-storage) must NOT be carried over into the product
 * CLI -- only native/q8/q5 (Q5_1) are exposed here. */
static void	test_no_q5_0_or_q4_public_alias(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv");
	args.push_back("q5_0");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv q5_0 is rejected -- not a public product alias");

	args = base_args();
	args.push_back("--kv");
	args.push_back("q5_1");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv q5_1 (the experimental spelling) is rejected -- \"q5\" is "
		"the only public spelling, and it must not silently succeed by "
		"accident");

	args = base_args();
	args.push_back("--kv");
	args.push_back("q4");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv q4 is rejected -- Q4 stays a research-only comparison "
		"baseline, never a public product mode");
}

/* Phase 11A Section 1/18: --kv adaptive is a new, explicit opt-in
 * request value (MEMBRANE_KV_STORE_ADAPTIVE), distinct from the three
 * real storage modes -- default stays native (Section 1: "Adaptive is
 * explicit opt-in"). */
static void	test_kv_adaptive_flag(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv adaptive accepted");
	TEST_ASSERT(o.kv_mode == 3 /* MEMBRANE_KV_STORE_ADAPTIVE */,
		"--kv adaptive sets the adaptive request value");

	TEST_ASSERT(run_parse(base_args(), &o) == MEMBRANE_EXIT_SUCCESS,
		"omitting --kv still parses cleanly");
	TEST_ASSERT(o.kv_mode == 0 /* MEMBRANE_KV_STORE_NATIVE */,
		"default --kv remains native even though adaptive now exists -- "
		"adaptive is opt-in only, never a new default");
}

static void	test_kv_budget_mib_flag(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	args.push_back("--kv-budget-mib");
	args.push_back("512");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv-budget-mib with --kv adaptive is accepted");
	TEST_ASSERT(o.want_kv_budget == 1, "want_kv_budget is set");
	TEST_ASSERT(o.kv_budget_bytes == (uint64_t)512 * 1024 * 1024,
		"--kv-budget-mib is stored as bytes (MiB * 1024 * 1024)");

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	args.push_back("--kv-budget-mib");
	args.push_back("0");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib 0 is rejected -- must be a positive MiB count");

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	args.push_back("--kv-budget-mib");
	args.push_back("nope");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib non-numeric is rejected");

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	args.push_back("--kv-budget-mib");
	args.push_back("-1");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib -1 is rejected (parse_u64_strict's leading '-' "
		"guard, same class as --ctx's)");

	args = base_args();
	args.push_back("--kv");
	args.push_back("adaptive");
	args.push_back("--kv-budget-mib");
	args.push_back("18446744073709551615");	/* UINT64_MAX */
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib is rejected when MiB * 1024 * 1024 would "
		"overflow uint64_t");
}

/* Section 5: "budget does NOT silently alter explicit q8/q5 choices"
 * -- enforced here as an outright parse-time rejection of the
 * combination, not a silent no-op, for native/q8/q5 and for the
 * flag's absence-of-adaptive default (no --kv at all) alike. */
static void	test_kv_budget_mib_requires_adaptive(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv-budget-mib");
	args.push_back("512");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib without --kv adaptive (--kv omitted, so native) "
		"is a CLI error");

	args = base_args();
	args.push_back("--kv");
	args.push_back("q8");
	args.push_back("--kv-budget-mib");
	args.push_back("512");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib with an explicit --kv q8 is a CLI error -- it "
		"must never silently alter an explicit compressed-mode choice");

	args = base_args();
	args.push_back("--kv");
	args.push_back("q5");
	args.push_back("--kv-budget-mib");
	args.push_back("512");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-budget-mib with an explicit --kv q5 is a CLI error, same "
		"as q8");
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

static void	test_include_text_requires_json(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--include-text");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--include-text without --json is a CLI error -- it has no "
		"defined effect on human-readable output");

	args = base_args();
	args.push_back("--json");
	args.push_back("--include-text");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--include-text with --json is accepted");
	TEST_ASSERT(o.include_text == 1, "include_text flag is stored");
}

static void	test_gpu_layers_default_is_cpu_only(void)
{
	membrane_run_opts_t	o;

	TEST_ASSERT(run_parse(base_args(), &o) == MEMBRANE_EXIT_SUCCESS,
		"no --gpu-layers is a valid, complete invocation");
	TEST_ASSERT(o.gpu_layers == 0,
		"default gpu_layers is 0 (CPU-only) -- explicit, not implicit,"
		" even on a GPU-capable build");
	TEST_ASSERT(o.want_device == 0, "default want_device is unset");
	TEST_ASSERT(o.device.empty(), "default device is empty");
}

static void	test_gpu_layers_all_and_explicit_zero(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers all (with the now-required --ctx) is accepted");
	TEST_ASSERT(o.gpu_layers == -1,
		"\"all\" is stored as -1, matching llama.cpp's own "
		"n_gpu_layers<0-means-all convention");

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("0");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers 0 is accepted -- the explicit CPU-forcing form");
	TEST_ASSERT(o.gpu_layers == 0,
		"--gpu-layers 0 stores the same value as the unset default");
}

static void	test_gpu_layers_positive_count_stored(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("12");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers 12 (with the now-required --ctx) is accepted");
	TEST_ASSERT(o.gpu_layers == 12, "--gpu-layers 12 value stored exactly");
}

static void	test_gpu_layers_malformed_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("nope");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-layers non-numeric, non-\"all\" value is a CLI error");

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("-1");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-layers -1 (the literal, not \"all\") is rejected -- "
		"\"all\" is the only accepted sentinel, avoiding a silent "
		"typo turning into \"every layer\"");
}

static void	test_device_flag_stored(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	args.push_back("--device");
	args.push_back("nvidia");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--device with --gpu-layers all is accepted");
	TEST_ASSERT(o.want_device == 1, "want_device is set");
	TEST_ASSERT(o.device == "nvidia", "--device value stored exactly");
}

static void	test_device_requires_nonzero_gpu_layers(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--device");
	args.push_back("nvidia");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--device without --gpu-layers is a CLI error -- naming a "
		"device with no GPU layers requested is ambiguous");

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("0");
	args.push_back("--device");
	args.push_back("nvidia");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--device with --gpu-layers 0 (explicit CPU-only) is still a "
		"CLI error, not just --device with the flag omitted entirely");
}

static void	test_device_empty_value_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("all");
	args.push_back("--device");
	args.push_back("");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--device \"\" is rejected at parse time -- an empty needle "
		"would otherwise match every device via std::string::find");
}

static void	test_gpu_layers_auto_parses(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers auto with an explicit --ctx is accepted");
	TEST_ASSERT(o.gpu_layers == -2,
		"\"auto\" is stored as -2 (MEMBRANE_GPU_LAYERS_AUTO)");
}

static void	test_gpu_layers_auto_requires_explicit_ctx(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-layers auto without --ctx is a CLI error -- the KV "
		"budget can't be estimated before context size is known");
}

static void	test_gpu_layers_all_and_n_require_explicit_ctx(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("all");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-layers all without --ctx is a CLI error -- the GPU "
		"memory guard can't check the real KV budget before an "
		"auto-sized context is known");

	args = base_args();
	args.push_back("--gpu-layers");
	args.push_back("10");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-layers N without --ctx is a CLI error, same as all/auto");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers all with an explicit --ctx is accepted");

	args = base_args();
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-layers 0 (CPU-only default) without --ctx is unaffected");
}

static void	test_gpu_bench_requires_gpu_layers(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--gpu-bench");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-bench without --gpu-layers is a CLI error");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("auto");
	args.push_back("--gpu-bench");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--gpu-bench with --gpu-layers auto is accepted");
	TEST_ASSERT(o.gpu_bench == 1, "gpu_bench flag is stored");
}

static void	test_gpu_bench_compare_kv_mutually_exclusive(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	args.push_back("--gpu-bench");
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--gpu-bench and --compare-kv are mutually exclusive");
}

static void	test_plan_only_accepted_alone(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--plan-only alone is accepted");
	TEST_ASSERT(o.plan_only == 1, "plan_only flag is stored");
}

static void	test_plan_only_off_by_default(void)
{
	membrane_run_opts_t	o;

	TEST_ASSERT(run_parse(base_args(), &o) == MEMBRANE_EXIT_SUCCESS,
		"base invocation still parses");
	TEST_ASSERT(o.plan_only == 0, "--plan-only is off by default");
}

/* Phase 30, Section 1/11/20: a real, pre-existing friction point this
 * phase's own first-run audit found -- --plan-only never generates, so
 * prompt content is never used by it, yet the CLI previously rejected
 * `--plan-only --ctx N --auto` outright (even this project's own
 * --help EXAMPLES text shipped exactly that command). --ctx explicit
 * means auto-sizing (the only real use of prompt content on this
 * path) never applies, so an empty placeholder prompt is injected
 * instead of requiring one. */
static void	test_plan_only_no_prompt_ok_with_explicit_ctx(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--plan-only --ctx N --auto parses without any --prompt");
	TEST_ASSERT(o.prompt_mode == MEMBRANE_RUN_PROMPT_TEXT,
		"an empty placeholder TEXT prompt is injected, not left NONE");
	TEST_ASSERT(o.prompt_text.empty(),
		"the injected placeholder prompt is genuinely empty, not a "
		"fabricated string");
}

static void	test_plan_only_no_prompt_no_ctx_still_requires_prompt(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--plan-only with neither --ctx nor --prompt still requires a "
		"prompt -- ctx auto-sizing genuinely needs one");
}

/* A real --prompt is never silently discarded/overridden by the
 * placeholder-injection logic above -- it only ever fires when
 * prompt_mode is still NONE. */
static void	test_plan_only_explicit_prompt_not_overridden(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();	/* --prompt "hello" already present */
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--plan-only with both --ctx and an explicit --prompt parses");
	TEST_ASSERT(o.prompt_text == "hello",
		"the real --prompt value is preserved, not replaced with an "
		"empty placeholder");
}

static void	test_plan_only_with_auto(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --plan-only composes cleanly");
	TEST_ASSERT(o.plan_only == 1 && o.auto_mode == 1,
		"both flags are stored independently");
}

static void	test_plan_only_rejects_compare_kv(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--plan-only");
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--plan-only with --compare-kv is a CLI error -- neither mode "
		"has a single \"plan\" of its own");
}

static void	test_plan_only_rejects_gpu_bench(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("auto");
	args.push_back("--plan-only");
	args.push_back("--gpu-bench");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--plan-only with --gpu-bench is a CLI error");
}

/* Phase 13.2, Section 14: the deferred docs follow-up from PR #22 --
 * --auto's help text must mention the --compare-kv/--gpu-bench
 * placement exception (product_cli.cpp:496-499's actual behavior). A
 * substring check against the real membrane_run_usage() output, not a
 * hand-copied duplicate of the text, so this test only stays green
 * while the real help text still says it. */
static void	test_help_text_documents_auto_compare_bench_exception(void)
{
	/* Review fix (CodeRabbit, PR #23): zero-initialized so strstr()
	 * below stays safely NUL-terminated even in the (currently
	 * unreachable, but not guaranteed to stay that way as this help
	 * text keeps growing phase over phase) case where the real --help
	 * output fills the entire buffer and fmemopen("w")'s own trailing-
	 * NUL-on-close has no room left to write. */
	char	buf[32768] = {0};
	FILE	*f = fmemopen(buf, sizeof(buf), "w");

	TEST_ASSERT(f != NULL, "fmemopen for capturing --help output");
	membrane_run_usage(f);
	fclose(f);
	TEST_ASSERT(strstr(buf, "--compare-kv") != NULL
			&& strstr(buf, "--gpu-bench") != NULL
			&& strstr(buf, "AUTOMATIC MODE") != NULL,
		"--help mentions --compare-kv/--gpu-bench in the AUTOMATIC "
		"MODE section");
	/* The specific sentence added for this follow-up -- "leaves --kv-
	 * placement at default instead" is the actual behavioral claim,
	 * not just an incidental mention of the flag names above. */
	TEST_ASSERT(strstr(buf, "leaves --kv-placement") != NULL,
		"--help states the placement exception itself, not just the "
		"flag names");
}

static void	test_kv_placement_default_unchanged(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"no --kv-placement given is accepted");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"kv_placement defaults to MEMBRANE_KV_PLACEMENT_DEFAULT -- "
		"Section 4: zero behavior change unless explicitly requested");
}

static void	test_kv_placement_values_parsed(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	args.push_back("--kv-placement");
	args.push_back("gpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv-placement gpu with --gpu-layers all is accepted");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_GPU,
		"gpu value stored exactly");

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv-placement cpu needs no --gpu-layers");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_CPU,
		"cpu value stored exactly");

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("default");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv-placement default needs no --gpu-layers");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"default value stored exactly");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("auto");
	args.push_back("--kv-placement");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv-placement auto with --gpu-layers auto is accepted");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO,
		"auto value stored exactly");
}

static void	test_kv_placement_invalid_value_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("bogus");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-placement with an unrecognized value is a CLI error");
}

static void	test_kv_placement_gpu_requires_gpu_layers(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("gpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-placement gpu without --gpu-layers is a CLI error");

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-placement auto without --gpu-layers is a CLI error");
}

static void	test_kv_placement_orthogonal_to_kv_precision(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	/* Section 5: precision and placement are independent dimensions --
	 * --kv q5 with --kv-placement cpu must set BOTH fields exactly as
	 * requested, neither overriding the other. */
	args = base_args();
	args.push_back("--kv");
	args.push_back("q5");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--kv q5 with --kv-placement cpu composes cleanly");
	TEST_ASSERT(o.kv_mode == 2 /* MEMBRANE_KV_STORE_Q5 */,
		"precision (q5) unaffected by placement");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_CPU,
		"placement (cpu) unaffected by precision");
}

static void	test_kv_placement_rejects_compare_and_bench_modes(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--kv-placement");
	args.push_back("cpu");
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-placement with --compare-kv is rejected (Phase 12H scope), "
		"not silently ignored");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--gpu-layers");
	args.push_back("all");
	args.push_back("--kv-placement");
	args.push_back("gpu");
	args.push_back("--gpu-bench");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--kv-placement with --gpu-bench is rejected");
}

/* Phase 13.1, Section 7-9: --auto is a preset applied once, after the
 * parse loop, to whichever of gpu_layers/kv_mode/kv_placement the user
 * did not also explicitly pass. Every case below is checked at the CLI-
 * parse level only -- whether the runtime GPU backend/device is
 * actually available is a main.cpp/resolve_gpu_config() concern (not
 * reachable from this llama-free parser), covered separately by a
 * Vulkan smoke test and code review, not a unit test here. */
static void	test_auto_alone_fills_all_three_fields(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --ctx N alone is a complete, valid invocation");
	TEST_ASSERT(o.auto_mode == 1, "auto_mode is set");
	TEST_ASSERT(o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
		"--auto fills gpu_layers with the auto sentinel");
	TEST_ASSERT(o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE,
		"--auto fills kv_mode with adaptive");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO,
		"--auto fills kv_placement with auto (gpu_layers ended up "
		"nonzero, so auto placement is valid)");
}

static void	test_auto_requires_explicit_ctx(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	/* Same underlying rule as --gpu-layers auto/all/N: the GPU memory
	 * guard can't check a KV budget before ctx size is known. --auto
	 * does not weaken this. */
	args = base_args();
	args.push_back("--auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--auto without --ctx is a CLI error, same as --gpu-layers auto");
}

static void	test_auto_explicit_kv_mode_overrides(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv");
	args.push_back("q8");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --kv q8 composes cleanly");
	TEST_ASSERT(o.kv_mode == 1 /* MEMBRANE_KV_STORE_Q8 */,
		"explicit --kv q8 overrides --auto's own adaptive default");
	TEST_ASSERT(o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
		"gpu_layers is still auto-managed (not explicitly given)");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO,
		"kv_placement is still auto-managed (not explicitly given)");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv");
	args.push_back("q5");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --kv q5 composes cleanly");
	TEST_ASSERT(o.kv_mode == 2 /* MEMBRANE_KV_STORE_Q5 */,
		"explicit --kv q5 overrides --auto's own adaptive default");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv");
	args.push_back("native");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --kv native composes cleanly");
	TEST_ASSERT(o.kv_mode == 0 /* MEMBRANE_KV_STORE_NATIVE */,
		"explicit --kv native overrides --auto's own adaptive default");
}

static void	test_auto_explicit_kv_placement_cpu_overrides(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --kv-placement cpu composes cleanly");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_CPU,
		"explicit --kv-placement cpu overrides --auto's own auto "
		"default -- Section 8's own worked example");
	TEST_ASSERT(o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
		"gpu_layers is still auto-managed by --auto");
	TEST_ASSERT(o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE,
		"kv_mode is still auto-managed by --auto");
}

static void	test_auto_explicit_kv_placement_gpu_overrides(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv-placement");
	args.push_back("gpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --kv-placement gpu composes cleanly (gpu_layers is "
		"auto, so the gpu|auto-requires-nonzero-gpu_layers check "
		"already passes)");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_GPU,
		"explicit --kv-placement gpu overrides --auto's own default");
}

static void	test_auto_explicit_gpu_layers_zero_falls_back_to_cpu(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	/* Section 9/15: an explicit CPU-only override must not be rejected
	 * by the gpu|auto-requires-nonzero-gpu_layers check just because
	 * --auto's OWN kv_placement default would otherwise have picked
	 * "auto" -- --auto must notice gpu_layers resolved to 0 and pick
	 * "default" placement instead, the sensible CPU-only outcome. */
	args = base_args();
	args.push_back("--auto");
	args.push_back("--gpu-layers");
	args.push_back("0");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --gpu-layers 0 composes cleanly, not a CLI error");
	TEST_ASSERT(o.gpu_layers == 0,
		"explicit --gpu-layers 0 overrides --auto's own auto default");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"kv_placement falls back to default (not auto) once gpu_layers "
		"is known to be 0 -- the sensible CPU-only outcome");
	TEST_ASSERT(o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE,
		"kv_mode is still auto-managed (CPU adaptive is still useful "
		"with no GPU at all)");
}

static void	test_auto_explicit_gpu_layers_positive_stays(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--gpu-layers");
	args.push_back("16");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --gpu-layers 16 composes cleanly");
	TEST_ASSERT(o.gpu_layers == 16,
		"explicit --gpu-layers N overrides --auto's own auto default");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO,
		"kv_placement still auto-manages (gpu_layers ended up nonzero)");
}

/* Review fix (CodeRabbit, PR #22): --auto's own kv_placement=auto
 * default silently made `--auto --compare-kv`/`--gpu-bench` fail with
 * "membrane-run: --kv-placement is not yet supported together with
 * --compare-kv/--gpu-bench" -- an error naming a flag the user never
 * typed. --compare-kv/--gpu-bench never had KV placement control to
 * begin with (Phase 12H scope, unrelated to --auto); --auto must not
 * inject a placement value the current mode can't accept. */
static void	test_auto_with_compare_kv_or_gpu_bench_stays_default_placement(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --compare-kv composes cleanly -- no longer rejected for "
		"a --kv-placement value the user never asked for");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"kv_placement stays default under --compare-kv even though "
		"gpu_layers ended up nonzero (auto-managed)");
	TEST_ASSERT(o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
		"gpu_layers is still auto-managed under --compare-kv");
	TEST_ASSERT(o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE,
		"kv_mode is still auto-managed under --compare-kv");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--gpu-bench");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--auto --gpu-bench composes cleanly -- same fix applies");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"kv_placement stays default under --gpu-bench too");
}

static void	test_default_behavior_unaffected_by_auto_fields(void)
{
	membrane_run_opts_t	o;

	/* Section 9: without --auto, nothing about the new fields changes
	 * pre-existing default behavior at all. */
	TEST_ASSERT(run_parse(base_args(), &o) == MEMBRANE_EXIT_SUCCESS,
		"the plain base invocation is still valid with no new flags");
	TEST_ASSERT(o.auto_mode == 0, "auto_mode is off by default");
	TEST_ASSERT(o.want_gpu_layers == 0, "want_gpu_layers is 0 by default");
	TEST_ASSERT(o.want_kv_mode == 0, "want_kv_mode is 0 by default");
	TEST_ASSERT(o.want_kv_placement == 0,
		"want_kv_placement is 0 by default");
	TEST_ASSERT(o.gpu_layers == 0,
		"default remains --gpu-layers 0 (CPU-only), unchanged");
	TEST_ASSERT(o.kv_mode == 0 /* MEMBRANE_KV_STORE_NATIVE */,
		"default remains --kv native, unchanged");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT,
		"default remains --kv-placement default, unchanged");
}

/* Phase 13.1 review follow-up (Section G, "does --auto's position in
 * argv unexpectedly alter precedence"): the defaulting pass runs once,
 * AFTER the full parse loop, using each field's want_* flag -- not
 * inline during parsing -- so argv order must never matter. This was
 * previously exercised only with --auto BEFORE the overriding flags;
 * this test additionally checks --auto AFTER them, and a flag on each
 * side of --auto in the same invocation. */
static void	test_auto_position_in_argv_does_not_affect_precedence(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o_before;
	membrane_run_opts_t			o_after;
	membrane_run_opts_t			o_both_sides;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--auto");
	args.push_back("--kv");
	args.push_back("q8");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o_before) == MEMBRANE_EXIT_SUCCESS,
		"--auto before overriding flags composes cleanly");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--kv");
	args.push_back("q8");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	args.push_back("--auto");
	TEST_ASSERT(run_parse(args, &o_after) == MEMBRANE_EXIT_SUCCESS,
		"--auto after overriding flags composes cleanly");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	args.push_back("--kv");
	args.push_back("q8");
	args.push_back("--auto");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o_both_sides) == MEMBRANE_EXIT_SUCCESS,
		"--auto between overriding flags composes cleanly");

	for (const membrane_run_opts_t *o : {&o_before, &o_after, &o_both_sides})
	{
		TEST_ASSERT(o->auto_mode == 1, "auto_mode set regardless of position");
		TEST_ASSERT(o->kv_mode == 1 /* MEMBRANE_KV_STORE_Q8 */,
			"explicit --kv q8 wins regardless of --auto's position in argv");
		TEST_ASSERT(o->kv_placement == MEMBRANE_KV_PLACEMENT_CPU,
			"explicit --kv-placement cpu wins regardless of --auto's "
			"position in argv");
		TEST_ASSERT(o->gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
			"gpu_layers is auto-managed identically in every position");
	}
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
	TEST_ASSERT(strcmp(buf, "MEMBRANE " MEMBRANE_VERSION "\n") == 0,
		"version output is exactly \"MEMBRANE \" + MEMBRANE_VERSION "
		"+ \"\\n\" -- an exact match so a stray suffix (e.g. a "
		"leftover \"-rc1\") can't slip through a substring check");
}

static void	test_help_mentions_key_concepts(void)
{
	FILE	*tmp;
	char	buf[32768];	/* Phase 12H: grown from 8192; Phase 30: grown
						 * again from 16384 -- the real help text keeps
						 * growing (most recently past 16384 with the
						 * --inspect-model additions), which silently
						 * truncated "Exit codes" out of the captured
						 * tail rather than testing anything meaningful
						 * about its absence. */
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
	TEST_ASSERT(strstr(buf, "adaptive") != NULL,
		"help documents --kv adaptive");
	TEST_ASSERT(strstr(buf, "--kv-budget-mib") != NULL,
		"help documents --kv-budget-mib");
	TEST_ASSERT(strstr(buf, "--compare-kv") != NULL,
		"help documents --compare-kv");
	TEST_ASSERT(strstr(buf, "--ctx") != NULL, "help documents --ctx");
	TEST_ASSERT(strstr(buf, "--json") != NULL, "help documents --json");
	TEST_ASSERT(strstr(buf, "Exit codes") != NULL,
		"help documents exit codes");
	TEST_ASSERT(strstr(buf, "--kv-placement") != NULL,
		"help documents --kv-placement");
	TEST_ASSERT(strstr(buf, "LLM_ARCH_LLAMA") != NULL,
		"help states the architecture-scope limitation");
	TEST_ASSERT(strstr(buf, "--auto") != NULL,
		"help documents --auto");
	TEST_ASSERT(strstr(buf, "AUTOMATIC MODE") != NULL,
		"help is organized into labeled sections, including "
		"AUTOMATIC MODE for --auto");
}

/* Phase 28, Section 10/15: --list-devices/--doctor need no --model/
 * --prompt at all -- same short-circuit shape as --help/--version
 * (test_help_and_version_short_circuit below), just resolved one step
 * later (after the parse loop, not mid-loop) since main.cpp -- not
 * this llama-free parser -- is what actually implements them (both
 * need live device enumeration). */
static void	test_list_devices_flag_short_circuits(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--list-devices");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--list-devices alone (no --model/--prompt) parses successfully");
	TEST_ASSERT(o.want_list_devices == 1, "--list-devices sets want_list_devices");
	TEST_ASSERT(o.want_doctor == 0, "--list-devices does not set want_doctor");
}

static void	test_doctor_flag_short_circuits(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--doctor");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--doctor alone (no --model/--prompt) parses successfully");
	TEST_ASSERT(o.want_doctor == 1, "--doctor sets want_doctor");
	TEST_ASSERT(o.want_list_devices == 0, "--doctor does not set want_list_devices");
}

/* --doctor still accepts --model (checked for readability, not
 * required) without falling through to the normal --model/--prompt
 * validation below it. */
static void	test_doctor_flag_with_model_still_short_circuits(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--doctor");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--doctor --model ... still short-circuits (no --prompt required)");
	TEST_ASSERT(o.want_doctor == 1, "--doctor sets want_doctor");
}

/* Section 3: reorganized --help structure -- QUICK START and EXAMPLES
 * are new sections this phase adds; --list-devices/--doctor are new
 * flags this phase adds and must be documented. Substring checks
 * against the real membrane_run_usage() output (Section 20: no
 * brittle whole-output snapshot), same pattern as the existing
 * test_help_text_documents_auto_compare_bench_exception /
 * test_help_mentions_key_concepts above. */
static void	test_help_documents_phase28_additions(void)
{
	char	buf[32768] = {0};
	FILE	*f = fmemopen(buf, sizeof(buf), "w");

	TEST_ASSERT(f != NULL, "fmemopen for capturing --help output");
	membrane_run_usage(f);
	fclose(f);
	TEST_ASSERT(strstr(buf, "QUICK START") != NULL,
		"help has a QUICK START section");
	TEST_ASSERT(strstr(buf, "EXAMPLES") != NULL,
		"help has an EXAMPLES section");
	TEST_ASSERT(strstr(buf, "--list-devices") != NULL,
		"help documents --list-devices");
	TEST_ASSERT(strstr(buf, "--doctor") != NULL,
		"help documents --doctor");
	TEST_ASSERT(strstr(buf, "MODEL / PROMPT") != NULL,
		"help has a MODEL / PROMPT section");
	TEST_ASSERT(strstr(buf, "ADVANCED MEMORY CONTROL") != NULL,
		"help has an ADVANCED MEMORY CONTROL section");
}

/* Phase 30, Section 5-8/29: --inspect-model CLI-parse-level coverage --
 * the actual GGUF-metadata read and compatibility check (main.cpp's
 * run_inspect_model_mode(), compat_check.c) are exercised by real
 * models in results/first-run/validation.json instead, the same
 * llama-free/testable-without-a-model split every other main.cpp mode
 * in this project already uses (see test_list_devices_flag_short_
 * circuits/test_doctor_flag_short_circuits above). */
static void	test_inspect_model_flag_requires_model(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--inspect-model");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--inspect-model without --model is a CLI error, same as an "
		"ordinary run");
	TEST_ASSERT(o.want_inspect_model == 1, "--inspect-model sets want_inspect_model");
}

static void	test_inspect_model_flag_no_prompt_required(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--inspect-model");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--model + --inspect-model parses successfully with no --prompt");
	TEST_ASSERT(o.want_inspect_model == 1, "--inspect-model sets want_inspect_model");
	TEST_ASSERT(o.prompt_mode == MEMBRANE_RUN_PROMPT_NONE,
		"no prompt mode is set -- --inspect-model never required one");
}

static void	test_inspect_model_flag_accepts_optional_ctx(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();	/* --model + --prompt already present */
	args.push_back("--inspect-model");
	args.push_back("--ctx");
	args.push_back("4096");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--inspect-model with --ctx (and even an unused --prompt) still "
		"parses -- --ctx is optional, not required, for --inspect-model");
	TEST_ASSERT(o.ctx == 4096, "--ctx value is stored normally");
}

static void	test_inspect_model_rejects_other_modes(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--inspect-model");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--inspect-model + --plan-only is rejected -- pick one mode");

	args.clear();
	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--inspect-model");
	args.push_back("--compare-kv");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--inspect-model + --compare-kv is rejected -- pick one mode");
}

static void	test_help_documents_inspect_model(void)
{
	char	buf[32768] = {0};
	FILE	*f = fmemopen(buf, sizeof(buf), "w");

	TEST_ASSERT(f != NULL, "fmemopen for capturing --help output");
	membrane_run_usage(f);
	fclose(f);
	TEST_ASSERT(strstr(buf, "--inspect-model") != NULL,
		"help documents --inspect-model");
}

/* ------------------------------------------------------------------ */
/* Phase 35: --ctx auto -- Section 38 test matrix, items 1-13          */
/* ------------------------------------------------------------------ */

/* 1. --ctx numeric unchanged. */
static void	test_ctx_numeric_still_explicit(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("2048");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS, "parses ok");
	TEST_ASSERT(o.ctx == 2048, "numeric value stored exactly as before");
	TEST_ASSERT(o.ctx_mode == MEMBRANE_RUN_CTX_EXPLICIT,
		"ctx_mode reflects EXPLICIT, a new field, no behavior change");
}

/* 2. --ctx auto parses. */
static void	test_ctx_auto_parses(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS, "parses ok");
	TEST_ASSERT(o.ctx_mode == MEMBRANE_RUN_CTX_AUTO, "ctx_mode is AUTO");
	TEST_ASSERT(o.ctx == 0, "numeric ctx stays 0 (unresolved placeholder, "
		"never the auto-size sentinel's own meaning)");
}

/* 3. Invalid --ctx string rejected. */
static void	test_ctx_invalid_string_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("bogus");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"a non-numeric, non-\"auto\" --ctx value is rejected");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("Auto");	/* case-sensitive: not the same as "auto" */
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"\"Auto\" (wrong case) is rejected, not silently treated as auto");
}

/* 4. Auto mode represented explicitly, not via a ctx==0 sentinel --
 * UNSPECIFIED (no --ctx at all) and AUTO (--ctx auto) are distinct
 * ctx_mode values even though both leave numeric ctx at 0. */
static void	test_ctx_unspecified_vs_auto_distinct(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o_unspecified;
	membrane_run_opts_t			o_auto;

	TEST_ASSERT(run_parse(base_args(), &o_unspecified) == MEMBRANE_EXIT_SUCCESS,
		"no --ctx at all parses ok");
	TEST_ASSERT(o_unspecified.ctx_mode == MEMBRANE_RUN_CTX_UNSPECIFIED,
		"omitted --ctx is UNSPECIFIED, not AUTO");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o_auto) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto parses ok");
	TEST_ASSERT(o_auto.ctx_mode != o_unspecified.ctx_mode,
		"AUTO and UNSPECIFIED are genuinely distinct values, never "
		"conflated via a shared ctx==0 sentinel");
}

/* 5. ctx auto + prompt accepted. */
static void	test_ctx_auto_with_prompt_accepted(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();	/* already has --prompt hello */
	args.push_back("--ctx");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto with a real prompt parses ok");
}

/* 6. ctx auto without prompt rejected. */
static void	test_ctx_auto_without_prompt_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--ctx");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--ctx auto without any --prompt/--prompt-file/--prompt - is a "
		"CLI error -- recommendation cannot invent a prompt length");
}

/* 7. ctx auto plan-only + prompt. */
static void	test_ctx_auto_plan_only_with_prompt(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto --plan-only with a real prompt parses ok");
	TEST_ASSERT(o.plan_only == 1, "plan_only set");
	TEST_ASSERT(o.ctx_mode == MEMBRANE_RUN_CTX_AUTO, "ctx_mode is AUTO");
}

/* 8. ctx auto plan-only without prompt rejected -- unlike EXPLICIT
 * --ctx N --plan-only (which legitimately never needs a prompt, tested
 * by test_plan_only_no_prompt_ok_with_explicit_ctx above), AUTO always
 * needs one (Section 8 of the Phase 35 task). */
static void	test_ctx_auto_plan_only_without_prompt_rejected(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args.push_back("membrane-run");
	args.push_back("--model");
	args.push_back("/no/such/model.gguf");
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--plan-only");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_CLI_ERROR,
		"--ctx auto --plan-only without a prompt is still a CLI error");
}

/* 9/10. explicit q8/q5 preserved with --ctx auto (never overridden by
 * auto-fill just because context is being recommended). */
static void	test_ctx_auto_explicit_kv_q8_preserved(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--kv");
	args.push_back("q8");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS, "parses ok");
	TEST_ASSERT(o.kv_mode == 1 /* MEMBRANE_KV_STORE_Q8 */,
		"explicit q8 preserved");
	TEST_ASSERT(o.want_kv_mode == 1, "want_kv_mode reflects explicit request");
}

static void	test_ctx_auto_explicit_kv_q5_preserved(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--kv");
	args.push_back("q5");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS, "parses ok");
	TEST_ASSERT(o.kv_mode == 2 /* MEMBRANE_KV_STORE_Q5 */,
		"explicit q5 preserved");
}

/* 11. explicit placement preserved with --ctx auto. */
static void	test_ctx_auto_explicit_placement_preserved(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--gpu-layers");
	args.push_back("4");
	args.push_back("--kv-placement");
	args.push_back("cpu");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS, "parses ok");
	TEST_ASSERT(o.kv_placement == MEMBRANE_KV_PLACEMENT_CPU,
		"explicit --kv-placement cpu preserved");
	TEST_ASSERT(o.want_kv_placement == 1,
		"want_kv_placement reflects explicit request");
}

/* 12. explicit gpu layers preserved with --ctx auto -- and, critically,
 * the pre-existing "--gpu-layers N requires --ctx" rejection does NOT
 * fire for AUTO (Section 2/5: exempted so recommendation, not the
 * user, supplies the context size before resolve_gpu_config() runs). */
static void	test_ctx_auto_explicit_gpu_layers_preserved(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--gpu-layers");
	args.push_back("12");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto --gpu-layers 12 parses ok -- NOT rejected by the "
		"gpu-layers-requires-explicit-ctx check");
	TEST_ASSERT(o.gpu_layers == 12, "explicit gpu-layers count preserved");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--gpu-layers");
	args.push_back("all");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto --gpu-layers all also parses ok (was previously "
		"rejected without an explicit numeric --ctx)");

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--gpu-layers");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto --gpu-layers auto also parses ok");
}

/* --ctx auto alone (no --auto) is legal (Section 9, option B: the core
 * already supports recommendation under fully-explicit/default
 * constraints) -- recommends context for whatever gpu_layers/kv/
 * kv_placement defaults or explicit flags are already in force. */
static void	test_ctx_auto_without_auto_flag_is_legal(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto without --auto parses ok -- not forced to require it");
	TEST_ASSERT(o.auto_mode == 0, "--auto itself was not implied");
	TEST_ASSERT(o.gpu_layers == 0, "gpu_layers keeps its own ordinary "
		"CPU-only default, unaffected by --ctx auto alone");
}

/* --ctx auto + --auto together is the primary documented path. */
static void	test_ctx_auto_with_auto_flag(void)
{
	std::vector<std::string>	args;
	membrane_run_opts_t			o;

	args = base_args();
	args.push_back("--ctx");
	args.push_back("auto");
	args.push_back("--auto");
	TEST_ASSERT(run_parse(args, &o) == MEMBRANE_EXIT_SUCCESS,
		"--ctx auto --auto parses ok");
	TEST_ASSERT(o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO,
		"--auto's own gpu_layers auto-fill still applies normally");
	TEST_ASSERT(o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE,
		"--auto's own kv auto-fill still applies normally");
}

/* Help documents --ctx N|auto. */
static void	test_help_documents_ctx_auto(void)
{
	char	buf[32768] = {0};
	FILE	*f = fmemopen(buf, sizeof(buf), "w");

	TEST_ASSERT(f != NULL, "fmemopen for capturing --help output");
	membrane_run_usage(f);
	fclose(f);
	TEST_ASSERT(strstr(buf, "--ctx N|auto") != NULL
		|| strstr(buf, "--ctx N | auto") != NULL,
		"help documents --ctx N|auto");
	TEST_ASSERT(strstr(buf, "OOM") == NULL
		|| strstr(buf, "not") != NULL || strstr(buf, "never") != NULL,
		"any OOM mention in --help is qualified (never an unqualified "
		"OOM-proof claim)");
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
	test_no_q5_0_or_q4_public_alias();
	test_kv_adaptive_flag();
	test_kv_budget_mib_flag();
	test_kv_budget_mib_requires_adaptive();
	test_compare_kv_flag();
	test_ctx_and_gen_tokens_malformed_rejected();
	test_ctx_and_gen_tokens_valid_stored();
	test_quiet_and_verbose_mutually_exclusive();
	test_include_text_requires_json();
	test_gpu_layers_default_is_cpu_only();
	test_gpu_layers_all_and_explicit_zero();
	test_gpu_layers_positive_count_stored();
	test_gpu_layers_malformed_rejected();
	test_device_flag_stored();
	test_device_requires_nonzero_gpu_layers();
	test_device_empty_value_rejected();
	test_gpu_layers_auto_parses();
	test_gpu_layers_auto_requires_explicit_ctx();
	test_gpu_layers_all_and_n_require_explicit_ctx();
	test_gpu_bench_requires_gpu_layers();
	test_gpu_bench_compare_kv_mutually_exclusive();
	test_plan_only_accepted_alone();
	test_plan_only_off_by_default();
	test_plan_only_no_prompt_ok_with_explicit_ctx();
	test_plan_only_no_prompt_no_ctx_still_requires_prompt();
	test_plan_only_explicit_prompt_not_overridden();
	test_plan_only_with_auto();
	test_plan_only_rejects_compare_kv();
	test_plan_only_rejects_gpu_bench();
	test_help_text_documents_auto_compare_bench_exception();
	test_kv_placement_default_unchanged();
	test_kv_placement_values_parsed();
	test_kv_placement_invalid_value_rejected();
	test_kv_placement_gpu_requires_gpu_layers();
	test_kv_placement_orthogonal_to_kv_precision();
	test_kv_placement_rejects_compare_and_bench_modes();
	test_auto_alone_fills_all_three_fields();
	test_auto_requires_explicit_ctx();
	test_auto_explicit_kv_mode_overrides();
	test_auto_explicit_kv_placement_cpu_overrides();
	test_auto_explicit_kv_placement_gpu_overrides();
	test_auto_explicit_gpu_layers_zero_falls_back_to_cpu();
	test_auto_explicit_gpu_layers_positive_stays();
	test_auto_with_compare_kv_or_gpu_bench_stays_default_placement();
	test_default_behavior_unaffected_by_auto_fields();
	test_auto_position_in_argv_does_not_affect_precedence();
	test_help_and_version_short_circuit();
	test_version_output_format();
	test_help_mentions_key_concepts();
	test_list_devices_flag_short_circuits();
	test_doctor_flag_short_circuits();
	test_doctor_flag_with_model_still_short_circuits();
	test_help_documents_phase28_additions();
	test_inspect_model_flag_requires_model();
	test_inspect_model_flag_no_prompt_required();
	test_inspect_model_flag_accepts_optional_ctx();
	test_inspect_model_rejects_other_modes();
	test_help_documents_inspect_model();
	test_ctx_numeric_still_explicit();
	test_ctx_auto_parses();
	test_ctx_invalid_string_rejected();
	test_ctx_unspecified_vs_auto_distinct();
	test_ctx_auto_with_prompt_accepted();
	test_ctx_auto_without_prompt_rejected();
	test_ctx_auto_plan_only_with_prompt();
	test_ctx_auto_plan_only_without_prompt_rejected();
	test_ctx_auto_explicit_kv_q8_preserved();
	test_ctx_auto_explicit_kv_q5_preserved();
	test_ctx_auto_explicit_placement_preserved();
	test_ctx_auto_explicit_gpu_layers_preserved();
	test_ctx_auto_without_auto_flag_is_legal();
	test_ctx_auto_with_auto_flag();
	test_help_documents_ctx_auto();
	printf("test_product_cli: all tests passed\n");
	return (0);
}
