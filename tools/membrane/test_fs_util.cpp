#include <cstdio>
#include <cstring>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fs_util.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B1: unit tests for fs_util.h's shared "mkdir -p +
 * atomic write" helper (registry_core.cpp, server_config.cpp,
 * service_cmd.cpp's unit-file write all depend on this being correct).
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-fs-util-test-XXXXXX";
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

static void	test_mkdir_parents_creates_nested_dirs(void)
{
	std::string			base = make_temp_dir();
	std::string			nested = base + "/a/b/c";
	membrane_fs_error_t	err;

	TEST_ASSERT(membrane_mkdir_parents(nested, &err) == true,
		"creating a 3-level nested directory succeeds");
	struct stat	st;

	TEST_ASSERT(stat(nested.c_str(), &st) == 0 && S_ISDIR(st.st_mode),
		"the deepest directory really exists");
	rmdir_recursive(base);
}

static void	test_mkdir_parents_eexist_is_not_an_error(void)
{
	std::string			base = make_temp_dir();
	std::string			nested = base + "/x/y";
	membrane_fs_error_t	err;

	TEST_ASSERT(membrane_mkdir_parents(nested, &err) == true,
		"first creation succeeds");
	TEST_ASSERT(membrane_mkdir_parents(nested, &err) == true,
		"a second call over the same already-existing path also succeeds "
		"-- EEXIST on any component is not an error");
	rmdir_recursive(base);
}

static void	test_atomic_write_creates_parent_and_writes_content(void)
{
	std::string			base = make_temp_dir();
	std::string			path = base + "/nested/dir/file.txt";
	membrane_fs_error_t	err;

	TEST_ASSERT(membrane_atomic_write_file(path, "hello world", &err)
		== true, "atomic write creates missing parent directories and "
		"writes the file");
	TEST_ASSERT(!err.set, "no error populated");

	FILE	*f = fopen(path.c_str(), "rb");

	TEST_ASSERT(f != NULL, "the file exists and is readable");
	char	buf[64] = {0};

	size_t	n = fread(buf, 1, sizeof(buf) - 1, f);

	(void)n;
	fclose(f);
	TEST_ASSERT(strcmp(buf, "hello world") == 0,
		"the file's content matches exactly what was written");
	rmdir_recursive(base);
}

static void	test_atomic_write_no_leftover_tmp_file(void)
{
	std::string			base = make_temp_dir();
	std::string			path = base + "/file.txt";
	membrane_fs_error_t	err;

	TEST_ASSERT(membrane_atomic_write_file(path, "content", &err) == true,
		"write succeeds");

	DIR				*dh = opendir(base.c_str());
	int				tmp_count = 0;
	int				real_count = 0;
	struct dirent	*de;

	TEST_ASSERT(dh != NULL, "could open the temp dir to scan it");
	if (dh != NULL)
	{
		while ((de = readdir(dh)) != NULL)
		{
			if (strstr(de->d_name, ".tmp.") != NULL)
				tmp_count++;
			if (strcmp(de->d_name, "file.txt") == 0)
				real_count++;
		}
		closedir(dh);
	}
	TEST_ASSERT(tmp_count == 0, "no leftover .tmp.* file after a "
		"successful atomic write -- rename() completed");
	TEST_ASSERT(real_count == 1, "exactly the real target file exists");
	rmdir_recursive(base);
}

static void	test_atomic_write_overwrites_existing_file(void)
{
	std::string			base = make_temp_dir();
	std::string			path = base + "/file.txt";
	membrane_fs_error_t	err;

	TEST_ASSERT(membrane_atomic_write_file(path, "version one", &err)
		== true, "first write succeeds");
	TEST_ASSERT(membrane_atomic_write_file(path, "version two", &err)
		== true, "a second write over the same path succeeds");

	FILE	*f = fopen(path.c_str(), "rb");
	char	buf[64] = {0};

	size_t	n = fread(buf, 1, sizeof(buf) - 1, f);

	(void)n;
	fclose(f);
	TEST_ASSERT(strcmp(buf, "version two") == 0,
		"the file now holds only the second write's content -- no "
		"leftover bytes from the first, longer-or-shorter write");
	rmdir_recursive(base);
}

int	main(void)
{
	test_mkdir_parents_creates_nested_dirs();
	test_mkdir_parents_eexist_is_not_an_error();
	test_atomic_write_creates_parent_and_writes_content();
	test_atomic_write_no_leftover_tmp_file();
	test_atomic_write_overwrites_existing_file();
	printf("test_fs_util: all tests passed\n");
	return (0);
}
