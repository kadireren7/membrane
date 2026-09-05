#include <cstdio>
#include <cstdlib>
#include <string>

#include <nlohmann/json.hpp>

#include "doctor_cmd.h"
#include "registry_core.h"
#include "server_config.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Mega Phase C, PR C1: unit tests for doctor_cmd.h's membrane_doctor_
 * collect() -- the same real registry/config/service checks `membrane
 * doctor` itself runs, driven directly (in-process, no subprocess)
 * against isolated temp registry/config/systemd-unit-dir locations via
 * this project's established MEMBRANE_MODELS_PATH/MEMBRANE_SERVER_
 * CONFIG_PATH/MEMBRANE_SYSTEMD_USER_DIR test-hook env overrides --
 * never the real user's own state. MEMBRANE_RUN_EXEC_PATH is left
 * UNSET here on purpose (this test binary has no membrane-run binary
 * sitting next to it) -- exercising the real "hardware check gracefully
 * degrades to WARN" path rather than the happy path, which the real
 * local dev-build smoke test already covers with a real override.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-doctor-cmd-test-XXXXXX";
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
		unsetenv("MEMBRANE_RUN_EXEC_PATH");
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
		rmdir_recursive(dir);
	}
};

static json	find_check(const json &root, const std::string &name)
{
	for (const auto &c : root["checks"])
		if (c["name"] == name)
			return (c);
	TEST_ASSERT(false, ("check '" + name + "' not found").c_str());
	return (json());
}

static void	test_empty_environment_is_ok_except_hardware(void)
{
	s_isolated_env	env;
	json			root;
	std::string		overall = membrane_doctor_collect(&root);

	TEST_ASSERT(root["schema_version"] == 1, "schema_version is 1");
	TEST_ASSERT(root.contains("membrane_version"), "membrane_version present");
	/* hardware WARNs (no membrane-run binary reachable in this test
	 * binary's own directory, and MEMBRANE_RUN_EXEC_PATH is
	 * deliberately unset) -- every OTHER check on a genuinely empty,
	 * isolated environment is OK. */
	TEST_ASSERT(overall == "WARN", "overall is WARN (only hardware)");
	TEST_ASSERT(find_check(root, "hardware")["status"] == "WARN",
		"hardware check WARNs when membrane-run can't be located");
	TEST_ASSERT(find_check(root, "registry")["status"] == "OK",
		"registry check OK on an empty, isolated registry");
	TEST_ASSERT(find_check(root, "registry")["detail"]["count"] == 0,
		"registry count is 0");
	TEST_ASSERT(find_check(root, "default_model")["status"] == "OK",
		"default_model check OK with none configured");
	TEST_ASSERT(find_check(root, "config")["status"] == "OK",
		"config check OK on a missing (defaults-to) config file");
}

static void	test_stale_model_warns(void)
{
	s_isolated_env				env;
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	entry.name = "gone";
	entry.path = "/tmp/membrane-doctor-test-nonexistent-file.gguf";
	entry.basename = "nonexistent-file.gguf";
	entry.arch_name = "llama";
	entry.model_max_context = 2048;
	entry.file_size_bytes = 123;
	entry.file_mtime_ns = 456;
	entry.added_at_unix = 1700000000;
	membrane_registry_add(&reg, entry, &err);
	membrane_registry_save(env.dir + "/models.json", reg, &err);

	json		root;
	std::string	overall = membrane_doctor_collect(&root);

	TEST_ASSERT(overall == "WARN", "overall is WARN (a stale model exists)");
	json	registry_check = find_check(root, "registry");

	TEST_ASSERT(registry_check["status"] == "WARN",
		"registry check itself WARNs");
	TEST_ASSERT(registry_check["detail"]["models"][0]["status"] == "MISSING",
		"the specific model is reported MISSING");
}

static void	test_dangling_default_model_warns(void)
{
	s_isolated_env					env;
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	err;

	cfg.default_model = "never-registered";
	membrane_server_config_save(env.dir + "/server.json", cfg, &err);

	json		root;
	std::string	overall = membrane_doctor_collect(&root);

	TEST_ASSERT(overall == "WARN", "overall is WARN (dangling default_model)");
	json	default_model_check = find_check(root, "default_model");

	TEST_ASSERT(default_model_check["status"] == "WARN",
		"default_model check itself WARNs");
	TEST_ASSERT(default_model_check["detail"]["registered"] == false,
		"registered is reported false");
}

static void	test_malformed_config_is_error(void)
{
	s_isolated_env	env;
	FILE			*f = fopen((env.dir + "/server.json").c_str(), "w");

	TEST_ASSERT(f != NULL, "could open the test config file for writing");
	fprintf(f, "{ not valid json");
	fclose(f);

	json		root;
	std::string	overall = membrane_doctor_collect(&root);

	TEST_ASSERT(overall == "ERROR",
		"overall is ERROR (a malformed config fails closed, never "
		"silently ignored)");
	TEST_ASSERT(find_check(root, "config")["status"] == "ERROR",
		"config check itself is ERROR");
}

/* Mega Phase C's own first product audit (Section 2 of the task) found
 * this exact real condition in a bare `ubuntu:24.04` container:
 * systemctl is not installed at all, and `membrane service start` used
 * to fail with an unhelpful "(no output)" message with no proactive
 * warning anywhere. Simulated here deterministically by pointing PATH
 * at a directory with nothing in it -- execvp("systemctl", ...) then
 * fails exactly the same way it would on a real systemd-less host,
 * without needing an actual container for this one specific check.
 * membrane-run itself is still resolved via MEMBRANE_RUN_EXEC_PATH's
 * own ABSOLUTE path override (execvp with an absolute argv[0] never
 * consults PATH at all), so only the systemctl-specific behavior is
 * exercised here, not a real hardware-check regression. */
static void	test_no_systemctl_warns_never_errors(void)
{
	s_isolated_env	env;
	const char		*old_path = getenv("PATH");
	std::string		saved_path = old_path != NULL ? old_path : "";
	std::string		empty_path_dir = env.dir + "/empty-path";

	int	mkdir_rc = system(("mkdir -p '" + empty_path_dir + "'").c_str());

	TEST_ASSERT(mkdir_rc == 0, "could create the empty PATH directory");
	setenv("PATH", empty_path_dir.c_str(), 1);

	json		root;
	std::string	overall = membrane_doctor_collect(&root);

	setenv("PATH", saved_path.c_str(), 1);
	TEST_ASSERT(overall == "WARN",
		"overall is WARN, never ERROR -- a missing systemctl is a real, "
		"survivable condition (`membrane serve` still works), never "
		"treated as fatal");
	json	service_check = find_check(root, "service");

	TEST_ASSERT(service_check["status"] == "WARN",
		"service check itself WARNs");
	TEST_ASSERT(service_check["detail"]["systemctl_available"] == false,
		"systemctl_available is honestly reported false");
	TEST_ASSERT(service_check["detail"].contains("message"),
		"a clear, actionable message is present -- never the old "
		"unhelpful '(no output)'");
}

int	main(void)
{
	test_empty_environment_is_ok_except_hardware();
	test_stale_model_warns();
	test_dangling_default_model_warns();
	test_malformed_config_is_error();
	test_no_systemctl_warns_never_errors();
	printf("test_doctor_cmd: all tests passed\n");
	return (0);
}
