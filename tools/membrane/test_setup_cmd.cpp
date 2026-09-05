#include <cstdio>
#include <cstdlib>
#include <string>

#include "setup_cmd.h"
#include "registry_core.h"
#include "server_config.h"
#include "product_cli.h"
#include "test_helpers.h"

/*
 * Mega Phase C, PR C1: unit tests for setup_cmd.h's membrane_setup_cmd_
 * dispatch() -- driven directly (in-process, no subprocess) against
 * isolated temp registry/config/systemd-unit-dir locations, always
 * with --no-service (this project's own established precedent: the
 * real systemd --user service lifecycle is validated locally, not in
 * CI, since a CI runner's own --user session availability is
 * uncertain) and never a real GGUF model (CI-safe by construction --
 * models/ is gitignored). This test binary is not interactive (no
 * TTY under ctest), so membrane_setup_cmd_dispatch()'s own is_
 * interactive() check is always false here regardless of --yes,
 * exercising the exact same "non-interactive automation" path real
 * CI/scripted use would take.
 *
 * The model-registration idempotence/conflict tests below never
 * actually need a valid GGUF file: setup_cmd.cpp's own pre-check
 * (does a registry entry with this NAME already exist, and does its
 * path match?) runs BEFORE ever calling into the real `membrane model
 * add` GGUF-validating path -- so a same-path "already registered"
 * case and a different-path "conflict" case are both resolved (and
 * membrane_model_cmd_dispatch("add", ...) is never reached) using
 * nothing more than a real, arbitrary temp file on disk.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-setup-cmd-test-XXXXXX";
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

static void	test_no_model_path_skips_registration_cleanly(void)
{
	s_isolated_env	env;
	int	rc = membrane_setup_cmd_dispatch({"--yes", "--no-service"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS,
		"setup with no --model and no TTY succeeds, skipping model "
		"registration entirely rather than hanging on a prompt");

	membrane_registry_t			reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_load(env.dir + "/models.json", &reg, &err)
		== true, "loading the registry afterward succeeds");
	TEST_ASSERT(reg.entries.empty(),
		"nothing was registered when no model path was given");
}

static void	test_model_registration_is_idempotent(void)
{
	s_isolated_env	env;
	std::string		fake_model_path = env.dir + "/fake.gguf";

	write_file(fake_model_path, "not a real gguf, only used for the "
		"same-path idempotence check, never actually GGUF-validated");

	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	entry.name = "existing";
	entry.path = fake_model_path;	/* setup_cmd.cpp canonicalizes via
									 * realpath() before comparing --
									 * already absolute here, matches */
	entry.basename = "fake.gguf";
	entry.arch_name = "llama";
	entry.model_max_context = 2048;
	entry.file_size_bytes = 10;
	entry.file_mtime_ns = 1;
	entry.added_at_unix = 1700000000;
	membrane_registry_add(&reg, entry, &err);
	membrane_registry_save(env.dir + "/models.json", reg, &err);

	int	rc = membrane_setup_cmd_dispatch({"--model", fake_model_path,
			"--model-name", "existing", "--yes", "--no-service"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS,
		"re-registering the exact same name+path succeeds (idempotent) "
		"-- never a raw DUPLICATE_NAME failure, and the real GGUF-"
		"validating add path is never even reached since the pre-check "
		"already recognized it as already-correct");

	membrane_registry_t	reloaded;

	TEST_ASSERT(membrane_registry_load(env.dir + "/models.json", &reloaded,
		&err) == true, "reloading the registry succeeds");
	TEST_ASSERT(reloaded.entries.size() == 1,
		"still exactly one entry -- no duplicate was created");
}

static void	test_model_registration_conflict_is_refused(void)
{
	s_isolated_env	env;
	std::string		path_a = env.dir + "/a.gguf";
	std::string		path_b = env.dir + "/b.gguf";

	write_file(path_a, "file a");
	write_file(path_b, "file b");

	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	entry.name = "conflict";
	entry.path = path_a;
	entry.basename = "a.gguf";
	entry.arch_name = "llama";
	entry.model_max_context = 2048;
	entry.file_size_bytes = 6;
	entry.file_mtime_ns = 1;
	entry.added_at_unix = 1700000000;
	membrane_registry_add(&reg, entry, &err);
	membrane_registry_save(env.dir + "/models.json", reg, &err);

	int	rc = membrane_setup_cmd_dispatch({"--model", path_b,
			"--model-name", "conflict", "--yes", "--no-service"}, false);

	TEST_ASSERT(rc != MEMBRANE_EXIT_SUCCESS,
		"registering the SAME name against a DIFFERENT path is refused, "
		"never silently overwritten (Section 8: never destructively "
		"undo/replace existing configuration)");

	membrane_registry_t	reloaded;

	TEST_ASSERT(membrane_registry_load(env.dir + "/models.json", &reloaded,
		&err) == true, "reloading the registry succeeds");
	TEST_ASSERT(reloaded.entries.size() == 1
		&& membrane_registry_find(reloaded, "conflict")->path == path_a,
		"the ORIGINAL entry is completely untouched after the refused "
		"conflict");
}

static void	test_no_service_flag_skips_service_steps(void)
{
	s_isolated_env	env;
	int	rc = membrane_setup_cmd_dispatch({"--yes", "--no-service"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS,
		"--no-service succeeds without ever attempting a real systemctl "
		"call");

	std::string	unit_path = env.dir + "/membrane.service";
	FILE		*f = fopen(unit_path.c_str(), "r");

	TEST_ASSERT(f == NULL,
		"no unit file was written -- --no-service really skipped the "
		"service step entirely, not just the start half of it");
	if (f != NULL)
		fclose(f);
}

int	main(void)
{
	test_no_model_path_skips_registration_cleanly();
	test_model_registration_is_idempotent();
	test_model_registration_conflict_is_refused();
	test_no_service_flag_skips_service_steps();
	printf("test_setup_cmd: all tests passed\n");
	return (0);
}
