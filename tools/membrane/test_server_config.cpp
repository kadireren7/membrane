#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "server_config.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B1: unit tests for server_config.h's persistent
 * `membrane serve` config -- path resolution, defaults, load()'s
 * "missing file is not an error / malformed or wrong schema fails
 * closed" contract, and save()'s atomic-write round trip. Same
 * temp-dir-per-test discipline as test_registry_core.cpp -- never the
 * user's own real config.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-server-config-test-XXXXXX";
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

static void	test_defaults(void)
{
	membrane_server_config_t	cfg = membrane_server_config_defaults();

	TEST_ASSERT(cfg.schema_version == MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION,
		"default schema_version matches the current build");
	TEST_ASSERT(cfg.listen_address == "127.0.0.1",
		"default listen_address is loopback-only");
	TEST_ASSERT(cfg.port == 8642, "default port is 8642");
	TEST_ASSERT(cfg.default_model.empty(),
		"default_model is empty -- Section 9: never force one");
	TEST_ASSERT(cfg.log_level == "info", "default log_level is info");
}

static void	test_load_nonexistent_is_defaults_not_error(void)
{
	std::string					dir = make_temp_dir();
	membrane_server_config_t		cfg;
	membrane_server_config_error_t	err;

	TEST_ASSERT(membrane_server_config_load(dir + "/server.json", &cfg, &err)
		== true, "loading a nonexistent config file succeeds");
	TEST_ASSERT(!err.set, "no error populated");
	TEST_ASSERT(cfg.port == 8642, "defaults are returned");
	rmdir_recursive(dir);
}

static void	test_save_load_round_trip_and_atomic(void)
{
	std::string					dir = make_temp_dir();
	std::string					path = dir + "/server.json";
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	err;

	cfg.listen_address = "127.0.0.1";
	cfg.port = 9000;
	cfg.default_model = "qwen";
	cfg.log_level = "debug";
	TEST_ASSERT(membrane_server_config_save(path, cfg, &err) == true,
		"save succeeds");
	TEST_ASSERT(!err.set, "no error on successful save");

	struct stat	st;

	TEST_ASSERT(stat(path.c_str(), &st) == 0,
		"the real config file exists after save");

	DIR				*dh = opendir(dir.c_str());
	int				tmp_count = 0;
	struct dirent	*de;

	TEST_ASSERT(dh != NULL, "could open the temp dir to scan it");
	if (dh != NULL)
	{
		while ((de = readdir(dh)) != NULL)
			if (strstr(de->d_name, ".tmp.") != NULL)
				tmp_count++;
		closedir(dh);
	}
	TEST_ASSERT(tmp_count == 0, "no leftover .tmp.* file after a "
		"successful atomic save");

	membrane_server_config_t		reloaded;
	membrane_server_config_error_t	reload_err;

	TEST_ASSERT(membrane_server_config_load(path, &reloaded, &reload_err)
		== true, "reloading the saved file succeeds");
	TEST_ASSERT(reloaded.port == 9000, "port survives the round trip");
	TEST_ASSERT(reloaded.default_model == "qwen",
		"default_model survives the round trip");
	TEST_ASSERT(reloaded.log_level == "debug",
		"log_level survives the round trip");
	rmdir_recursive(dir);
}

static void	test_load_malformed_json_fails_closed(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/server.json";
	FILE		*f = fopen(path.c_str(), "w");

	TEST_ASSERT(f != NULL, "could open the test file for writing");
	fprintf(f, "{ not valid json");
	fclose(f);

	membrane_server_config_t		cfg;
	membrane_server_config_error_t	err;

	TEST_ASSERT(membrane_server_config_load(path, &cfg, &err) == false,
		"loading malformed JSON fails");
	TEST_ASSERT(err.set && err.code == "PARSE_ERROR",
		"error code is PARSE_ERROR");
	rmdir_recursive(dir);
}

static void	test_load_missing_schema_version_fails_closed(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/server.json";
	FILE		*f = fopen(path.c_str(), "w");

	fprintf(f, "{\"listen_address\": \"127.0.0.1\"}");
	fclose(f);

	membrane_server_config_t		cfg;
	membrane_server_config_error_t	err;

	TEST_ASSERT(membrane_server_config_load(path, &cfg, &err) == false,
		"loading valid JSON with no schema_version field fails");
	TEST_ASSERT(err.set && err.code == "PARSE_ERROR",
		"error code is PARSE_ERROR");
	rmdir_recursive(dir);
}

static void	test_load_unsupported_schema_version_fails_closed(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/server.json";
	FILE		*f = fopen(path.c_str(), "w");

	fprintf(f, "{\"schema_version\": 999}");
	fclose(f);

	membrane_server_config_t		cfg;
	membrane_server_config_error_t	err;

	TEST_ASSERT(membrane_server_config_load(path, &cfg, &err) == false,
		"a schema_version this build does not understand fails closed "
		"rather than silently reinterpreting the file");
	TEST_ASSERT(err.set && err.code == "UNSUPPORTED_SCHEMA",
		"error code is UNSUPPORTED_SCHEMA");
	rmdir_recursive(dir);
}

/* Milestone J: the optional overrides block. */
static void	test_overrides_round_trip_and_validation(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/server.json";

	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	err;

	TEST_ASSERT(membrane_server_overrides_empty(cfg.overrides),
		"defaults carry no overrides (pre-J behavior)");
	TEST_ASSERT(membrane_server_config_save(path, cfg, &err), "save plain");
	{
		FILE	*f = fopen(path.c_str(), "r");
		char	buf[512] = {0};

		TEST_ASSERT(f != NULL && fread(buf, 1, sizeof(buf) - 1, f) > 0,
			"read back");
		fclose(f);
		TEST_ASSERT(strstr(buf, "overrides") == NULL, "an empty overrides "
			"block is not written (file shape unchanged)");
	}
	cfg.overrides.has_context = true;
	cfg.overrides.context = 4096;
	cfg.overrides.has_kv_precision = true;
	cfg.overrides.kv_precision = "q8";
	cfg.overrides.has_kv_placement = true;
	cfg.overrides.kv_placement = "cpu";
	cfg.overrides.has_gpu_layers = true;
	cfg.overrides.gpu_layers = -1;
	TEST_ASSERT(membrane_server_config_save(path, cfg, &err), "save overrides");

	membrane_server_config_t	back;

	TEST_ASSERT(membrane_server_config_load(path, &back, &err), "load");
	TEST_ASSERT(membrane_server_overrides_equal(cfg.overrides, back.overrides),
		"overrides round-trip");
	cfg.overrides.kv_precision = "q9";
	TEST_ASSERT(!membrane_server_config_save(path, cfg, &err)
		&& err.code == "INVALID_OVERRIDES", "invalid overrides are not saved");
	for (const char *bad : {
			"{\"schema_version\":1,\"overrides\":{\"context\":\"big\"}}",
			"{\"schema_version\":1,\"overrides\":{\"context\":4}}",
			"{\"schema_version\":1,\"overrides\":{\"kv_precision\":\"q9\"}}",
			"{\"schema_version\":1,\"overrides\":{\"surprise\":1}}",
			"{\"schema_version\":1,\"overrides\":[1]}",
			"{\"schema_version\":1,\"overrides\":{\"gpu_layers\":\"most\"}}"})
	{
		FILE	*f = fopen(path.c_str(), "w");

		fputs(bad, f);
		fclose(f);
		TEST_ASSERT(!membrane_server_config_load(path, &back, &err)
			&& err.code == "INVALID_OVERRIDES",
			"an invalid overrides block fails the whole load (fail closed)");
	}
	rmdir_recursive(dir);
}

static void	test_resolve_path_env_override(void)
{
	const char	*old_override = getenv("MEMBRANE_SERVER_CONFIG_PATH");
	std::string	saved_override = old_override != NULL ? old_override : "";
	bool		had_override = old_override != NULL;
	const char	*old_xdg = getenv("XDG_CONFIG_HOME");
	std::string	saved_xdg = old_xdg != NULL ? old_xdg : "";
	bool		had_xdg = old_xdg != NULL;

	setenv("MEMBRANE_SERVER_CONFIG_PATH", "/tmp/override-server.json", 1);
	TEST_ASSERT(membrane_server_config_resolve_path()
		== "/tmp/override-server.json",
		"MEMBRANE_SERVER_CONFIG_PATH overrides everything else -- tests/CI "
		"never touch a real user config");
	unsetenv("MEMBRANE_SERVER_CONFIG_PATH");

	setenv("XDG_CONFIG_HOME", "/tmp/xdgtest", 1);
	TEST_ASSERT(membrane_server_config_resolve_path()
		== "/tmp/xdgtest/membrane/server.json",
		"XDG_CONFIG_HOME is honored when set");
	unsetenv("XDG_CONFIG_HOME");

	std::string	fallback = membrane_server_config_resolve_path();

	TEST_ASSERT(fallback.find("/.config/membrane/server.json")
		!= std::string::npos,
		"falls back to $HOME/.config/membrane/server.json");

	if (had_override)
		setenv("MEMBRANE_SERVER_CONFIG_PATH", saved_override.c_str(), 1);
	if (had_xdg)
		setenv("XDG_CONFIG_HOME", saved_xdg.c_str(), 1);
}

int	main(void)
{
	test_defaults();
	test_load_nonexistent_is_defaults_not_error();
	test_save_load_round_trip_and_atomic();
	test_load_malformed_json_fails_closed();
	test_load_missing_schema_version_fails_closed();
	test_load_unsupported_schema_version_fails_closed();
	test_overrides_round_trip_and_validation();
	test_resolve_path_env_override();
	printf("test_server_config: all tests passed\n");
	return (0);
}
