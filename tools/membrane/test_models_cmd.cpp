#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "model_views.h"
#include "models_cmd.h"
#include "product_cli.h"
#include "registry_core.h"
#include "server_config.h"
#include "test_helpers.h"

using json = nlohmann::json;

/* P2/P6/P7: `membrane models` against an isolated registry that holds one
 * real fixture file, plus the pure helpers. */

static std::string	g_dir;

static std::string	capture(const std::vector<std::string> &args, bool want_json,
						int *rc)
{
	std::string	tmpl = g_dir + "/cap-XXXXXX";
	int			fd = mkstemp(&tmpl[0]);

	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_models_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	FILE		*f = fopen(tmpl.c_str(), "rb");
	std::string	out;
	char		buf[4096];
	size_t		n;

	while (f != NULL && (n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	if (f != NULL)
		fclose(f);
	unlink(tmpl.c_str());
	return (out);
}

static void	test_pure_helpers(void)
{
	TEST_ASSERT(membrane_human_size(1100000000ull) == "1.1 GB"
		&& membrane_human_size(725553760ull) == "726 MB"
		&& membrane_human_size(2000) == "2 KB", "sizes");
	TEST_ASSERT(membrane_quant_from_filename("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
			== "Q4_K_M" && membrane_quant_from_filename("SmolLM2-360M-"
				"Instruct-F16.gguf") == "F16" && membrane_quant_from_filename(
				"llama-3.2-1b-instruct-q8_0.gguf") == "Q8_0"
		&& membrane_quant_from_filename("model.gguf") == ""
		&& membrane_quant_from_filename("stories15M.gguf") == "",
		"quant from file name (no false positive on 'stories15M')");
}

static void	test_empty_registry(void)
{
	int			rc;
	std::string	out = capture({}, false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("No models are "
			"installed yet") != std::string::npos && out.find("membrane "
			"models recommend") != std::string::npos, "next steps when empty");
	json	j = json::parse(capture({}, true, &rc));

	TEST_ASSERT(j["models"].is_array() && j["models"].empty(), "JSON empty");
}

static void	test_installed_listing(void)
{
	int							rc;
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	e;
	std::string					path = membrane_registry_resolve_path();

	e.name = "smollm2-360m-instruct";
	e.path = std::string(MEMBRANE_TEST_SOURCE_DIR) + "/models/stories15M.gguf";
	e.basename = "SmolLM2-360M-Instruct-Q8_0.gguf";
	e.file_size_bytes = 725553760;
	e.file_mtime_ns = 1;
	e.added_at_unix = 1;
	membrane_registry_load(path, &reg, &err);
	membrane_registry_add(&reg, e, &err);
	e.name = "my-custom";
	e.basename = "custom-Q4_K_M.gguf";
	e.file_size_bytes = 1100000000;
	membrane_registry_add(&reg, e, &err);
	membrane_registry_save(path, reg, &err);
	{
		membrane_server_config_t		cfg = membrane_server_config_defaults();
		membrane_server_config_error_t	cerr;

		cfg.default_model = "smollm2-360m-instruct";
		cfg.port = 1;			/* nothing listens: "loaded" stays unknown */
		membrane_server_config_save(g_dir + "/server.json", cfg, &cerr);
	}
	std::string	out = capture({}, false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("Installed models")
		!= std::string::npos, "header");
	TEST_ASSERT(out.find("SmolLM2-360M-Instruct") != std::string::npos
		&& out.find("smollm2-360m-instruct") != std::string::npos
		&& out.find("726 MB") != std::string::npos && out.find("Q8_0")
			!= std::string::npos, "display name, id, size, quant from the "
				"catalog variant that matches the file");
	TEST_ASSERT(out.find("my-custom") != std::string::npos
		&& out.find("Q4_K_M") != std::string::npos, "a model the catalog "
			"does not know still lists, quant read from its file name");
	TEST_ASSERT(out.find("default") != std::string::npos
		&& out.find("Next:") != std::string::npos
		&& out.find('\033') == std::string::npos, "default marker, next "
			"steps, no ANSI in redirected output");
	size_t	a = out.find("my-custom");
	size_t	b = out.find("smollm2-360m-instruct");

	TEST_ASSERT(a != std::string::npos && b != std::string::npos
		&& a < b, "deterministic order (by display name: 'my-custom' < "
			"'SmolLM2...')");
	json	j = json::parse(capture({}, true, &rc));

	TEST_ASSERT(j["models"].size() == 2 && j["models"][0]["id"] == "my-custom"
		&& j["models"][1]["default"] == true
		&& j["models"][1]["loaded"].is_null(), "JSON: ids, default flag, "
			"'loaded' is null (unknown) when no server answers, not false");
	TEST_ASSERT(j["models"][1]["tags"][0] == "tiny"
		&& j["models"][1]["tested_status"] == "validated", "catalog labels");
}

static void	test_available_catalog(void)
{
	int			rc;
	std::string	out = capture({"available"}, false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("Models MEMBRANE "
			"can install") != std::string::npos, "header");
	for (const char *n : {"Qwen2.5-0.5B-Instruct", "Qwen2.5-Coder-1.5B-"
			"Instruct", "Llama-3.2-1B-Instruct", "DeepSeek-R1-Distill-Qwen-"
			"1.5B", "Gemma-2-2B-it", "Phi-3.5-mini-instruct"})
		TEST_ASSERT(out.find(n) != std::string::npos, "catalog family listed");
	TEST_ASSERT(out.find("metadata-only") != std::string::npos
		&& out.find("validated") != std::string::npos
		&& out.find("MEMBRANE manages the runtime, not model quality")
			!= std::string::npos, "evidence labels explained");
	json	j = json::parse(capture({"available"}, true, &rc));

	TEST_ASSERT(j["models"].size() == 9, "nine catalog families");
	bool	coding = false;
	bool	reasoning = false;

	for (const auto &m : j["models"])
	{
		TEST_ASSERT(m["license"].is_string() && m["purpose"].is_string()
			&& m["tested_status"].is_string(), "every family has license, "
				"purpose, evidence label");
		for (const auto &t : m["tags"])
		{
			coding = coding || t == "coding";
			reasoning = reasoning || t == "reasoning";
		}
	}
	TEST_ASSERT(coding && reasoning, "diverse categories");
	bool	sorted = true;

	for (size_t i = 1; i < j["models"].size(); ++i)
		if (j["models"][i]["size_bytes"].get<uint64_t>() == 0)
			sorted = false;
	TEST_ASSERT(sorted, "sizes known");
}

static void	test_recommend(void)
{
	int			rc;
	std::string	out = capture({"recommend"}, false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("Suggested for this "
			"computer") != std::string::npos && out.find("Measured:")
		!= std::string::npos, "header + measured hardware");
	TEST_ASSERT(out.find("not a quality ranking") != std::string::npos
		&& out.find("Expected fit") != std::string::npos,
		"states the rule and that fit is an estimate");
	TEST_ASSERT(out.find("best") == std::string::npos
		&& out.find("Best") == std::string::npos, "never says 'best'");
	json	j = json::parse(capture({"recommend"}, true, &rc));

	TEST_ASSERT(j["hardware"]["ram_total_bytes"].is_number(), "hardware");
	for (const auto &r : j["recommendations"])
		TEST_ASSERT(r["expected_fit"].is_string() && r["download_bytes"]
				.is_number() && r["fit_basis"].get<std::string>().find(
					"weights only") != std::string::npos
			&& r["tested_status"].is_string(), "each carries its evidence "
				"and the basis of the fit estimate");
	std::string	again = capture({"recommend"}, true, &rc);
	json		j2 = json::parse(again);

	TEST_ASSERT(j["recommendations"] == j2["recommendations"],
		"deterministic for identical hardware");
}

static void	test_usage(void)
{
	int	rc;

	capture({"bogus"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown subcommand");
	capture({"available", "x"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "stray argument");
}

int	main(void)
{
	char	tmpl[] = "/tmp/membrane-models-test-XXXXXX";

	TEST_ASSERT(mkdtemp(tmpl) != NULL, "temp dir");
	g_dir = tmpl;
	setenv("MEMBRANE_MODELS_PATH", (g_dir + "/models.json").c_str(), 1);
	setenv("MEMBRANE_SERVER_CONFIG_PATH", (g_dir + "/server.json").c_str(), 1);
	test_pure_helpers();
	test_empty_registry();
	test_installed_listing();
	test_available_catalog();
	test_recommend();
	test_usage();
	{
		std::string	cmd = "rm -rf '" + g_dir + "'";

		if (system(cmd.c_str()) != 0)
			fprintf(stderr, "warning: cleanup failed\n");
	}
	printf("test_models_cmd: all tests passed\n");
	return (0);
}
