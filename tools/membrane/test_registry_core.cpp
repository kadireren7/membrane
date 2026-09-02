#include <cstdio>
#include <cstring>
#include <string>

#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include "registry_core.h"
#include "test_helpers.h"

/*
 * Mega Phase A, PR A2: unit tests for registry_core.h's pure add/remove/
 * find logic and its file-backed load/save (atomic write, malformed-JSON
 * handling, nonexistent-file-is-empty-not-error). Uses a real temp
 * directory this test creates and removes itself -- NEVER the user's own
 * real registry (registry_path_or_default()'s MEMBRANE_MODELS_PATH/XDG
 * resolution lives in model_cmd.cpp, not exercised here). GGUF validation
 * itself is model_cmd.cpp's job (needs a real GGUF file), not this
 * module's -- see registry_core.h's own top comment for the split.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-registry-test-XXXXXX";
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

static membrane_registry_entry_t	fake_entry(const std::string &name,
					const std::string &path, uint64_t size = 1000,
					int64_t mtime = 12345)
{
	membrane_registry_entry_t	e;

	e.name = name;
	e.path = path;
	e.basename = "fake.gguf";
	e.arch_name = "llama";
	e.model_max_context = 8192;
	e.file_size_bytes = size;
	e.file_mtime_ns = mtime;
	e.added_at_unix = 1700000000;
	return (e);
}

/* 1. Load of a nonexistent registry file is an empty registry, not an
 * error (Section 10: first `model add` on a fresh machine). */
static void	test_load_nonexistent_is_empty_not_error(void)
{
	std::string				dir = make_temp_dir();
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_load(dir + "/models.json", &reg, &err)
		== true, "loading a nonexistent registry file succeeds");
	TEST_ASSERT(!err.set, "no error populated");
	TEST_ASSERT(reg.entries.empty(), "the registry is empty");
	rmdir_recursive(dir);
}

/* 2. add/find/list round trip. */
static void	test_add_and_find(void)
{
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_add(&reg,
		fake_entry("qwen", "/models/qwen.gguf"), &err) == true,
		"adding a fresh entry succeeds");
	TEST_ASSERT(!err.set, "no error on successful add");
	TEST_ASSERT(reg.entries.size() == 1, "registry now has one entry");
	const membrane_registry_entry_t	*found = membrane_registry_find(reg,
			"qwen");

	TEST_ASSERT(found != NULL, "find() locates the entry by name");
	TEST_ASSERT(found->path == "/models/qwen.gguf", "path matches");
	TEST_ASSERT(membrane_registry_find(reg, "nonexistent") == NULL,
		"find() returns NULL for an unregistered name");
}

/* 3. Duplicate NAME is rejected, regardless of path. */
static void	test_duplicate_name_rejected(void)
{
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	membrane_registry_add(&reg, fake_entry("qwen", "/models/a.gguf"), &err);
	TEST_ASSERT(membrane_registry_add(&reg,
		fake_entry("qwen", "/models/b.gguf"), &err) == false,
		"a second entry with the SAME name is rejected, even with a "
		"different path");
	TEST_ASSERT(err.set, "error is populated");
	TEST_ASSERT(err.code == "DUPLICATE_NAME", "error code is DUPLICATE_NAME");
	TEST_ASSERT(reg.entries.size() == 1,
		"the registry still has only the original entry");
	TEST_ASSERT(membrane_registry_find(reg, "qwen")->path == "/models/a.gguf",
		"the original entry is unchanged");
}

/* A duplicate PATH under a genuinely different name is allowed (a
 * harmless alias). */
static void	test_duplicate_path_different_name_allowed(void)
{
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	membrane_registry_add(&reg, fake_entry("qwen", "/models/a.gguf"), &err);
	TEST_ASSERT(membrane_registry_add(&reg,
		fake_entry("qwen-alias", "/models/a.gguf"), &err) == true,
		"the same path under a different name is allowed");
	TEST_ASSERT(reg.entries.size() == 2, "both entries exist");
}

/* 4. remove(). */
static void	test_remove(void)
{
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	membrane_registry_add(&reg, fake_entry("qwen", "/models/a.gguf"), &err);
	TEST_ASSERT(membrane_registry_remove(&reg, "qwen", &err) == true,
		"removing a real entry succeeds");
	TEST_ASSERT(reg.entries.empty(), "the registry is empty again");
	TEST_ASSERT(membrane_registry_remove(&reg, "qwen", &err) == false,
		"removing an already-removed name fails");
	TEST_ASSERT(err.set && err.code == "NOT_FOUND",
		"error code is NOT_FOUND");
}

/* 5. Atomic save + load round trip: real file, real temp dir. */
static void	test_save_load_round_trip_and_atomic(void)
{
	std::string					dir = make_temp_dir();
	std::string					path = dir + "/models.json";
	membrane_registry_t			reg;
	membrane_registry_error_t	err;

	membrane_registry_add(&reg, fake_entry("qwen", "/models/qwen.gguf"),
		&err);
	membrane_registry_add(&reg, fake_entry("smol", "/models/smol.gguf"),
		&err);
	TEST_ASSERT(membrane_registry_save(path, reg, &err) == true,
		"save succeeds");
	TEST_ASSERT(!err.set, "no error on successful save");
	struct stat	st;

	TEST_ASSERT(stat(path.c_str(), &st) == 0,
		"the real registry file exists after save");
	/* No stray .tmp.* file left behind (rename() completed) -- scan the
	 * temp dir's own entries directly rather than shelling out. */
	DIR		*dh = opendir(dir.c_str());
	int		tmp_count = 0;
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

	membrane_registry_t	reloaded;

	TEST_ASSERT(membrane_registry_load(path, &reloaded, &err) == true,
		"reloading the saved file succeeds");
	TEST_ASSERT(reloaded.entries.size() == 2,
		"both entries survive the round trip");
	TEST_ASSERT(membrane_registry_find(reloaded, "qwen") != NULL
		&& membrane_registry_find(reloaded, "smol") != NULL,
		"both names are still findable after reload");
	TEST_ASSERT(membrane_registry_find(reloaded, "qwen")->model_max_context
		== 8192, "numeric metadata survives the round trip");

	/* Concurrent-read safety, "where practical" (Section 14): two
	 * independent loads of the same already-saved file see identical,
	 * consistent content -- no shared mutable state between calls. */
	membrane_registry_t	reader_a;
	membrane_registry_t	reader_b;

	TEST_ASSERT(membrane_registry_load(path, &reader_a, &err)
		&& membrane_registry_load(path, &reader_b, &err),
		"two independent loads of the same file both succeed");
	TEST_ASSERT(reader_a.entries.size() == reader_b.entries.size(),
		"both readers see the same entry count");
	rmdir_recursive(dir);
}

/* Malformed JSON fails closed, never silently discards. */
static void	test_load_malformed_json_fails_closed(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/models.json";
	FILE		*f = fopen(path.c_str(), "w");

	TEST_ASSERT(f != NULL, "could open the test file for writing");
	fprintf(f, "{ this is not valid json");
	fclose(f);

	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_load(path, &reg, &err) == false,
		"loading malformed JSON fails");
	TEST_ASSERT(err.set, "error is populated");
	TEST_ASSERT(err.code == "PARSE_ERROR", "error code is PARSE_ERROR");
	rmdir_recursive(dir);
}

/* A file with valid JSON but the wrong shape also fails closed. */
static void	test_load_wrong_shape_fails_closed(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/models.json";
	FILE		*f = fopen(path.c_str(), "w");

	fprintf(f, "{\"not_a_registry\": true}");
	fclose(f);

	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_load(path, &reg, &err) == false,
		"loading valid JSON with the wrong shape fails");
	TEST_ASSERT(err.set && err.code == "PARSE_ERROR",
		"error code is PARSE_ERROR");
	rmdir_recursive(dir);
}

/* 6. check_identity(): OK, MISSING, MODIFIED, UNREADABLE. */
static void	test_check_identity_all_states(void)
{
	membrane_registry_entry_t	e = fake_entry("qwen", "/models/qwen.gguf",
			1000, 12345);

	TEST_ASSERT(strcmp(membrane_registry_check_identity(e,
		MEMBRANE_REGISTRY_STAT_OK, 1000, 12345),
		MEMBRANE_REGISTRY_CHECK_OK) == 0,
		"matching size/mtime is OK");
	TEST_ASSERT(strcmp(membrane_registry_check_identity(e,
		MEMBRANE_REGISTRY_STAT_MISSING, 0, 0),
		MEMBRANE_REGISTRY_CHECK_MISSING) == 0,
		"a missing file is MISSING");
	TEST_ASSERT(strcmp(membrane_registry_check_identity(e,
		MEMBRANE_REGISTRY_STAT_OK, 2000, 12345),
		MEMBRANE_REGISTRY_CHECK_MODIFIED) == 0,
		"a different size is MODIFIED");
	TEST_ASSERT(strcmp(membrane_registry_check_identity(e,
		MEMBRANE_REGISTRY_STAT_OK, 1000, 99999),
		MEMBRANE_REGISTRY_CHECK_MODIFIED) == 0,
		"a different mtime is MODIFIED");
	TEST_ASSERT(strcmp(membrane_registry_check_identity(e,
		MEMBRANE_REGISTRY_STAT_ERROR, 0, 0),
		MEMBRANE_REGISTRY_CHECK_UNREADABLE) == 0,
		"a stat() error other than ENOENT is UNREADABLE");
}

/* 7. Default path resolution honors XDG_DATA_HOME / falls back to HOME. */
static void	test_default_path_xdg_and_fallback(void)
{
	std::string	saved_xdg;
	const char	*old_xdg = getenv("XDG_DATA_HOME");
	bool		had_xdg = old_xdg != NULL;

	if (had_xdg)
		saved_xdg = old_xdg;
	setenv("XDG_DATA_HOME", "/tmp/xdgtest", 1);
	TEST_ASSERT(membrane_registry_default_path()
		== "/tmp/xdgtest/membrane/models.json",
		"XDG_DATA_HOME is honored when set");
	unsetenv("XDG_DATA_HOME");
	std::string	fallback = membrane_registry_default_path();

	TEST_ASSERT(fallback.find("/.local/share/membrane/models.json")
		!= std::string::npos,
		"falls back to $HOME/.local/share/membrane/models.json");
	if (had_xdg)
		setenv("XDG_DATA_HOME", saved_xdg.c_str(), 1);
}

int	main(void)
{
	test_load_nonexistent_is_empty_not_error();
	test_add_and_find();
	test_duplicate_name_rejected();
	test_duplicate_path_different_name_allowed();
	test_remove();
	test_save_load_round_trip_and_atomic();
	test_load_malformed_json_fails_closed();
	test_load_wrong_shape_fails_closed();
	test_check_identity_all_states();
	test_default_path_xdg_and_fallback();
	printf("test_registry_core: all tests passed\n");
	return (0);
}
