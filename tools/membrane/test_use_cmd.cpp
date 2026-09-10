#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <httplib.h>

#include "use_cmd.h"
#include "server.h"
#include "registry_core.h"
#include "server_config.h"
#include "product_cli.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D6: unit/integration tests for `membrane use` --
 * driven directly (in-process, no subprocess), matching test_setup_
 * cmd.cpp's own isolated-temp-directory-env-var convention exactly.
 * Never a real network download here (Section 35 of the task: "avoid
 * launching real multi-GB downloads in ctest" -- this project's own
 * existing precedent, test_download_manager.cpp's own top comment,
 * already established that even a real HTTPS transfer stays OUT of
 * ctest, verified manually instead; the one real not-installed-model
 * consent/install/download/activate end-to-end pass against the small
 * real `smollm2-135m-instruct` catalog entry is likewise run manually,
 * recorded in results/model-lifecycle-ux/validation.json). This test
 * binary has no TTY under ctest (same as test_setup_cmd.cpp), so
 * cli_shared.h's own is_interactive() is always false here regardless
 * of --yes -- exercising the real non-interactive/automation path.
 *
 * One test below (test_switch_to_invalid_gguf_reports_failure) DOES
 * launch a real membrane_server_run() instance and drives a real HTTP
 * round trip through the new POST /membrane/v1/models/activate endpoint
 * against a real (if intentionally non-GGUF) file -- CI-safe (no real
 * multi-hundred-MB model, matching test_server.cpp's own "no real GGUF
 * model anywhere in this file" precedent) while still exercising
 * acquire_model_slot()'s own real failure path end to end, not a mock.
 */

# define TEST_PORT	18943

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-use-cmd-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	rmdir_recursive(const std::string &dir)
{
	std::string	cmd = "rm -rf '" + dir + "'";
	int			rc = system(cmd.c_str());

	if (rc != 0)
		fprintf(stderr, "warning: cleanup of %s may have failed (rc=%d)\n",
			dir.c_str(), rc);
}

static void	write_file(const std::string &path, const std::string &content)
{
	FILE	*f = fopen(path.c_str(), "w");

	TEST_ASSERT(f != NULL, "could open a test file for writing");
	fwrite(content.data(), 1, content.size(), f);
	fclose(f);
}

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		dir = make_temp_dir();
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_MODELS_INSTALL_DIR", (dir + "/models").c_str(), 1);
		setenv("MEMBRANE_SERVER_CONFIG_PATH",
			(dir + "/server.json").c_str(), 1);
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_MODELS_INSTALL_DIR");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		rmdir_recursive(dir);
	}
};

static void	register_entry(const std::string &registry_path,
				const std::string &name, const std::string &path)
{
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	membrane_registry_load(registry_path, &reg, &err);
	entry.name = name;
	entry.path = path;
	entry.basename = "fake.gguf";
	entry.arch_name = "llama";
	entry.model_max_context = 2048;
	entry.file_size_bytes = 6;
	entry.file_mtime_ns = 1;
	entry.added_at_unix = 1700000000;
	membrane_registry_add(&reg, entry, &err);
	membrane_registry_save(registry_path, reg, &err);
}

static void	test_unknown_model_returns_not_found(void)
{
	s_isolated_env	env;
	int	rc = membrane_use_cmd_dispatch({"this-model-does-not-exist-anywhere"},
			false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR,
		"a name that is neither registered nor a known catalog id/alias "
		"is a clean CLI_ERROR, never a crash or a silent no-op");
}

static void	test_installed_but_service_stopped(void)
{
	s_isolated_env	env;
	std::string		fake_path = env.dir + "/fake.gguf";

	write_file(fake_path, "not a real gguf");
	register_entry(env.dir + "/models.json", "my-model", fake_path);

	int	rc = membrane_use_cmd_dispatch({"my-model"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS,
		"selecting an already-installed model while no server is running "
		"still succeeds (Section 13 of the task) -- never a failure just "
		"because the service is stopped");

	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;

	TEST_ASSERT(membrane_server_config_load(env.dir + "/server.json", &cfg,
		&cfg_err) == true, "reloading the config afterward succeeds");
	TEST_ASSERT(cfg.default_model == "my-model",
		"default_model was set even though the service is stopped -- "
		"Section 11/13: default changes, active stays none");
}

static void	test_installed_but_file_missing(void)
{
	s_isolated_env	env;

	register_entry(env.dir + "/models.json", "ghost-model",
		env.dir + "/does-not-exist.gguf");

	int	rc = membrane_use_cmd_dispatch({"ghost-model"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_MODEL_ERROR,
		"a registered model whose file no longer exists on disk is "
		"refused (MODEL_FILE_MISSING), never silently selected as if it "
		"were still installed (Section 4 of the task)");

	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;

	membrane_server_config_load(env.dir + "/server.json", &cfg, &cfg_err);
	TEST_ASSERT(cfg.default_model.empty(),
		"default_model was never set for a model whose file is missing");
}

static void	test_noninteractive_without_yes_requires_consent(void)
{
	s_isolated_env	env;
	/* A REAL catalog entry (see model_catalog.cpp) -- resolution alone
	 * (no download) is exercised here; the consent gate fails BEFORE any
	 * network call would ever happen. */
	int	rc = membrane_use_cmd_dispatch({"smollm2-135m-instruct"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR,
		"a catalog model that is not installed, with no --yes and no TTY "
		"(ctest has none), fails clearly (NONINTERACTIVE_CONSENT_REQUIRED) "
		"instead of hanging or silently downloading (Section 6)");

	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;

	membrane_registry_load(env.dir + "/models.json", &reg, &reg_err);
	TEST_ASSERT(reg.entries.empty(),
		"nothing was downloaded/registered when consent was never given");
}

static void	test_invalid_variant_override_is_refused(void)
{
	s_isolated_env	env;
	int	rc = membrane_use_cmd_dispatch({"smollm2-135m-instruct", "--quant",
			"NOT_A_REAL_QUANT", "--yes"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR,
		"an explicit --quant naming a variant the family does not have is "
		"refused before any download is attempted");
}

/*
 * Real membrane_server_run() instance (background thread), same pattern
 * as test_server.cpp's own start_test_server()/stop_test_server() -- no
 * real GGUF model, just a real HTTP round trip exercising the new admin
 * endpoint's own real failure/recovery path.
 */
static std::thread	*g_server_thread = NULL;

static void	start_test_server(const std::string &registry_path)
{
	membrane_server_options_t	opts;

	opts.bind_address = "127.0.0.1";
	opts.port = TEST_PORT;
	opts.allow_non_loopback = false;
	opts.registry_path = registry_path;
	g_server_thread = new std::thread([opts]()
		{ membrane_server_run(opts); });
	httplib::Client	probe("127.0.0.1", TEST_PORT);
	int				attempts = 0;

	probe.set_connection_timeout(0, 50000);
	while (attempts < 100)
	{
		auto	res = probe.Get("/health");

		if (res && res->status == 200)
			return ;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		attempts++;
	}
	TEST_ASSERT(false, "server did not become ready within 5s");
}

static void	stop_test_server(void)
{
	membrane_server_request_stop();
	if (g_server_thread != NULL)
	{
		g_server_thread->join();
		delete g_server_thread;
		g_server_thread = NULL;
	}
}

static void	test_switch_to_invalid_gguf_reports_failure(void)
{
	s_isolated_env	env;
	std::string		fake_path = env.dir + "/fake.gguf";

	write_file(fake_path, "not a real gguf -- a real load attempt against "
		"this file must fail honestly");
	register_entry(env.dir + "/models.json", "unloadable", fake_path);

	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	cfg.listen_address = "127.0.0.1";
	cfg.port = TEST_PORT;
	membrane_server_config_save(env.dir + "/server.json", cfg, &cfg_err);

	start_test_server(env.dir + "/models.json");

	int	rc = membrane_use_cmd_dispatch({"unloadable"}, false);

	stop_test_server();
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR,
		"a real server that is reachable but fails to load the requested "
		"(invalid) model reports a real switch failure, never a false "
		"success (Section 16 of the task)");

	membrane_server_config_t	reloaded;

	membrane_server_config_load(env.dir + "/server.json", &reloaded,
		&cfg_err);
	TEST_ASSERT(reloaded.default_model == "unloadable",
		"default_model is still set even though the LIVE switch failed -- "
		"Section 11: default and active are distinct promises");
}

int	main(void)
{
	test_unknown_model_returns_not_found();
	test_installed_but_service_stopped();
	test_installed_but_file_missing();
	test_noninteractive_without_yes_requires_consent();
	test_invalid_variant_override_is_refused();
	test_switch_to_invalid_gguf_reports_failure();
	printf("test_use_cmd: all tests passed\n");
	return (0);
}
