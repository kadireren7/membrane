#include <cstdio>
#include <string>

#ifndef _WIN32
# include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "runtime_cmd.h"
#include "runtime_capabilities.h"
#include "product_cli.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone H1: `membrane runtime list|inspect`'s own tests -- driven
 * directly (in-process, no subprocess), same real-dispatch-function
 * convention as test_plan_cmd.cpp/test_doctor_cmd.cpp. Unlike those,
 * this command touches no registry/config/service file at all (runtime_
 * cmd.h's own top comment), so no per-test isolated-env fixture is
 * needed for most tests -- only the read-only-guarantee test below
 * sets one up, specifically to PROVE the absence of any filesystem
 * footprint rather than to isolate real state.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-runtime-cmd-test-XXXXXX";
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

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		dir = make_temp_dir();
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_SERVER_CONFIG_PATH",
			(dir + "/server.json").c_str(), 1);
		setenv("MEMBRANE_SYSTEMD_USER_DIR", dir.c_str(), 1);
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
		rmdir_recursive(dir);
	}
};

static bool	dir_is_empty(const std::string &dir)
{
	std::string	cmd = "[ -z \"$(ls -A '" + dir + "' 2>/dev/null)\" ]";

	return (system(cmd.c_str()) == 0);
}

static std::string	capture_stdout_of_runtime_dispatch(
				const std::vector<std::string> &args, bool want_json,
				int *out_rc)
{
	fflush(stdout);
	char	tmpl[] = "/tmp/membrane-runtime-cmd-test-capture-XXXXXX";
	int		fd = mkstemp(tmpl);

	TEST_ASSERT(fd >= 0, "could create a temp file to capture stdout into");
	int	saved_fd = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*out_rc = membrane_runtime_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved_fd, STDOUT_FILENO);
	close(saved_fd);

	FILE		*f = fopen(tmpl, "r");
	std::string	captured;
	char		buf[4096];
	size_t		n;

	TEST_ASSERT(f != NULL, "could reopen the captured-stdout temp file");
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		captured.append(buf, n);
	fclose(f);
	remove(tmpl);
	return (captured);
}

/* ------------------------------------------------------------------ */
/* E: runtime list -- native runtime appears; deterministic JSON      */
/* ------------------------------------------------------------------ */

static void	test_list_json_contains_native(void)
{
	json	j = membrane_runtime_list_json();

	TEST_ASSERT(j["schema_version"] == MEMBRANE_RUNTIME_SCHEMA_VERSION,
		"list JSON carries the runtime schema version");
	TEST_ASSERT(j["membrane_version"].get<std::string>() == MEMBRANE_VERSION,
		"list JSON carries MEMBRANE_VERSION");
	TEST_ASSERT(j["runtimes"].is_array() && j["runtimes"].size() == 1,
		"exactly one runtime is listed in H1");
	TEST_ASSERT(j["runtimes"][0]["id"] == MEMBRANE_RUNTIME_ID_NATIVE,
		"the one listed runtime is membrane-native");
	TEST_ASSERT(j["runtimes"][0]["type"] == "native",
		"membrane-native's JSON type is 'native'");
	TEST_ASSERT(j["runtimes"][0]["status"] == "available",
		"membrane-native's JSON status is 'available'");
}

static void	test_list_json_deterministic(void)
{
	std::string	a = membrane_runtime_list_json().dump();
	std::string	b = membrane_runtime_list_json().dump();

	TEST_ASSERT(a == b, "repeated calls to membrane_runtime_list_json() "
		"produce byte-identical output");
}

static void	test_dispatch_list_matches_library_json(void)
{
	int			rc;
	std::string	captured = capture_stdout_of_runtime_dispatch({"list"},
			true, &rc);
	json		from_cli = json::parse(captured);
	json		from_lib = membrane_runtime_list_json();

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "`membrane runtime list "
		"--json` exits 0");
	TEST_ASSERT(from_cli == from_lib, "the CLI's printed JSON is exactly "
		"membrane_runtime_list_json()'s own output");
}

static void	test_dispatch_list_human_exit_success(void)
{
	int			rc;
	std::string	captured = capture_stdout_of_runtime_dispatch({"list"},
			false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "`membrane runtime list` "
		"exits 0");
	TEST_ASSERT(captured.find(MEMBRANE_RUNTIME_ID_NATIVE) != std::string::npos,
		"the human table lists membrane-native");
	TEST_ASSERT(captured.find("available") != std::string::npos,
		"the human table shows an available status");
}

/* ------------------------------------------------------------------ */
/* F: runtime inspect -- correct capability fields; stable schema     */
/* ------------------------------------------------------------------ */

static void	test_inspect_json_native_fields(void)
{
	json		j;
	std::string	err;
	bool		ok = membrane_runtime_inspect_json(MEMBRANE_RUNTIME_ID_NATIVE,
			&j, &err);

	TEST_ASSERT(ok, "inspecting membrane-native succeeds");
	TEST_ASSERT(j.contains("schema_version") && j.contains("membrane_version")
		&& j.contains("runtime"), "the inspect schema has its 3 top-level "
		"keys");

	const json	&r = j["runtime"];

	TEST_ASSERT(r["id"] == MEMBRANE_RUNTIME_ID_NATIVE, "runtime.id");
	TEST_ASSERT(r["type"] == "native", "runtime.type");
	TEST_ASSERT(r["execution_mode"] == "embedded_native",
		"runtime.execution_mode");
	TEST_ASSERT(r["status"] == "available", "runtime.status");
	TEST_ASSERT(r["endpoint"].is_null(), "an embedded runtime has a null "
		"endpoint");
	TEST_ASSERT(r["capability_provenance"] == "static_contract",
		"runtime.capability_provenance");

	const json	&c = r["capabilities"];

	TEST_ASSERT(c["chat_completions"] == "supported",
		"capabilities.chat_completions");
	TEST_ASSERT(c["streaming"] == "supported", "capabilities.streaming");
	TEST_ASSERT(c["cancellation"] == "supported",
		"capabilities.cancellation");
	TEST_ASSERT(c["context_control"] == "supported",
		"capabilities.context_control");
	TEST_ASSERT(c["gpu_layer_control"] == "supported",
		"capabilities.gpu_layer_control");
	TEST_ASSERT(c["kv_precision_control"] == "supported",
		"capabilities.kv_precision_control");
	TEST_ASSERT(c["kv_placement_control"] == "supported",
		"capabilities.kv_placement_control");
	TEST_ASSERT(c["concurrency_control"] == "partial",
		"capabilities.concurrency_control");
	TEST_ASSERT(c["ram_usage"] == "partial", "capabilities.ram_usage");
	TEST_ASSERT(c["vram_usage"] == "partial", "capabilities.vram_usage");
	TEST_ASSERT(c["kv_cache_usage"] == "partial",
		"capabilities.kv_cache_usage");
	TEST_ASSERT(c["loaded_model_memory"] == "partial",
		"capabilities.loaded_model_memory");
	TEST_ASSERT(c["live_kv_migration"] == "unsupported",
		"capabilities.live_kv_migration");
	TEST_ASSERT(c["dynamic_reconfiguration"] == "unsupported",
		"capabilities.dynamic_reconfiguration");
}

static void	test_inspect_unknown_runtime_fails_clearly(void)
{
	json		j;
	std::string	err;
	bool		ok = membrane_runtime_inspect_json("does-not-exist", &j, &err);

	TEST_ASSERT(!ok, "an unknown runtime id is not describable");
	TEST_ASSERT(err.find("unknown runtime") != std::string::npos,
		"the error message says 'unknown runtime'");
}

static void	test_inspect_reserved_runtime_distinguishes_from_unknown(void)
{
	json		j;
	std::string	err;
	bool		ok = membrane_runtime_inspect_json(MEMBRANE_RUNTIME_ID_OLLAMA,
			&j, &err);

	TEST_ASSERT(!ok, "the reserved 'ollama' id has no adapter in H1");
	TEST_ASSERT(err.find("reserved") != std::string::npos
		&& err.find("adapter") != std::string::npos,
		"the error text distinguishes 'reserved, no adapter yet' from a "
		"plain unknown id");
	TEST_ASSERT(err.find("unknown runtime") == std::string::npos,
		"a reserved id is never reported with the generic 'unknown "
		"runtime' wording");
}

static void	test_dispatch_inspect_unknown_id_is_cli_error(void)
{
	int	rc;

	capture_stdout_of_runtime_dispatch({"inspect", "nope"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "--json inspect of an "
		"unknown id exits CLI_ERROR");
	capture_stdout_of_runtime_dispatch({"inspect", "nope"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "human inspect of an "
		"unknown id exits CLI_ERROR");
}

static void	test_dispatch_inspect_human_shows_capabilities(void)
{
	int			rc;
	std::string	captured = capture_stdout_of_runtime_dispatch(
			{"inspect", MEMBRANE_RUNTIME_ID_NATIVE}, false, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "human inspect of "
		"membrane-native exits 0");
	TEST_ASSERT(captured.find("Chat completions: supported")
		!= std::string::npos, "human output shows chat completions");
	TEST_ASSERT(captured.find("Live KV migration: unsupported")
		!= std::string::npos, "human output discloses what is NOT "
		"supported, not only what is");
}

static void	test_dispatch_no_args_and_unknown_subcommand(void)
{
	int	rc;

	capture_stdout_of_runtime_dispatch({}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "no subcommand is a "
		"CLI_ERROR");
	capture_stdout_of_runtime_dispatch({"frobnicate"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "an unknown subcommand is "
		"a CLI_ERROR");
}

/* ------------------------------------------------------------------ */
/* G: no service dependency                                           */
/* ------------------------------------------------------------------ */

static void	test_available_without_any_service_running(void)
{
	int			rc;
	std::string	captured = capture_stdout_of_runtime_dispatch(
			{"inspect", MEMBRANE_RUNTIME_ID_NATIVE}, true, &rc);
	json		j = json::parse(captured);

	/* No `membrane serve`/service process is started anywhere in this
	 * test binary -- membrane-native still reports available, because
	 * this module never checks service state at all (runtime_
	 * capabilities.h's own "availability != service running" contract). */
	TEST_ASSERT(j["runtime"]["status"] == "available",
		"membrane-native is available with no service running");
}

/* ------------------------------------------------------------------ */
/* H: read-only behavior                                              */
/* ------------------------------------------------------------------ */

static void	test_dispatch_creates_no_files(void)
{
	s_isolated_env	env;
	int				rc;

	capture_stdout_of_runtime_dispatch({"list"}, false, &rc);
	capture_stdout_of_runtime_dispatch({"list"}, true, &rc);
	capture_stdout_of_runtime_dispatch(
		{"inspect", MEMBRANE_RUNTIME_ID_NATIVE}, false, &rc);
	capture_stdout_of_runtime_dispatch(
		{"inspect", MEMBRANE_RUNTIME_ID_NATIVE}, true, &rc);
	capture_stdout_of_runtime_dispatch({"inspect", "nope"}, true, &rc);

	TEST_ASSERT(dir_is_empty(env.dir), "`membrane runtime list`/`inspect` "
		"never create any file -- no registry, no config, no service "
		"unit, nothing at all, even with MEMBRANE_MODELS_PATH/"
		"MEMBRANE_SERVER_CONFIG_PATH/MEMBRANE_SYSTEMD_USER_DIR pointed "
		"at an isolated, otherwise-empty directory");
}

int	main(void)
{
	test_list_json_contains_native();
	test_list_json_deterministic();
	test_dispatch_list_matches_library_json();
	test_dispatch_list_human_exit_success();
	test_inspect_json_native_fields();
	test_inspect_unknown_runtime_fails_clearly();
	test_inspect_reserved_runtime_distinguishes_from_unknown();
	test_dispatch_inspect_unknown_id_is_cli_error();
	test_dispatch_inspect_human_shows_capabilities();
	test_dispatch_no_args_and_unknown_subcommand();
	test_available_without_any_service_running();
	test_dispatch_creates_no_files();
	printf("test_runtime_cmd: all tests passed\n");
	return (0);
}
