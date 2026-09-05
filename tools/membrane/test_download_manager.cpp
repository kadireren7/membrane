#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "download_manager.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D1: CI-safe (no real network) unit tests for
 * download_manager.h -- disk-space checking against a real temp
 * directory, sha256 computation against a real local file with a
 * known-correct digest, and the HTTPS-only URL guard. The real,
 * network-dependent end-to-end download path (a real HTTPS transfer,
 * real redirect-following, real atomic rename) was verified manually,
 * once, against a real small file from one of this catalog's own real
 * repos -- see results/model-catalog/validation.json -- matching this
 * project's own established "no real network/GGUF dependency in CI"
 * convention (docs/soak-and-concurrency-testing.md, model_cmd.cpp's
 * own test suite, etc.).
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-download-test-XXXXXX";
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

static void	test_disk_space_check_passes_for_small_requirement(void)
{
	std::string	dir = make_temp_dir();
	membrane_download_error_t	err;

	TEST_ASSERT(membrane_download_check_disk_space(dir, 1024, &err),
		"1 KiB requirement passes on a real, non-full filesystem");
	rmdir_recursive(dir);
}

static void	test_disk_space_check_fails_for_absurd_requirement(void)
{
	std::string	dir = make_temp_dir();
	membrane_download_error_t	err;
	/* Real, absurdly large -- no real dev/CI filesystem has this much
	 * free space, so this deterministically exercises the real refusal
	 * path (never a mocked filesystem). */
	uint64_t	absurd = 1000ull * 1024 * 1024 * 1024 * 1024;	/* 1000 TiB */

	TEST_ASSERT(!membrane_download_check_disk_space(dir, absurd, &err),
		"an absurd requirement is refused");
	TEST_ASSERT(err.set && err.code == "DISK_FULL", "error code is DISK_FULL");
	rmdir_recursive(dir);
}

static void	test_disk_space_check_nonexistent_dir_fails_closed(void)
{
	membrane_download_error_t	err;

	TEST_ASSERT(!membrane_download_check_disk_space(
		"/definitely/does/not/exist/at/all", 1024, &err),
		"a nonexistent directory fails closed, never assumed OK");
	TEST_ASSERT(err.set && err.code == "IO_ERROR", "error code is IO_ERROR");
}

static void	test_compute_sha256_matches_known_digest(void)
{
	std::string	dir = make_temp_dir();
	std::string	path = dir + "/hello.txt";
	std::ofstream	f(path);

	f << "membrane";
	f.close();

	/* Real, independently-verified SHA-256 of the exact 8-byte ASCII
	 * string "membrane" (`printf 'membrane' | sha256sum`, cross-checked
	 * against Python's own hashlib.sha256 -- both agree), not derived
	 * from or trusted blindly out of this module's own code. */
	const std::string	expected
			= "d686ae03d9a39fa9f9331bbc5d87b3533b0609b2f68856ddf49d99f7fd1"
			"6c2d1";
	std::string	actual;
	membrane_download_error_t	err;
	bool		ok = membrane_compute_sha256(path, &actual, &err);

	if (!ok && err.code == "TOOL_UNAVAILABLE")
	{
		printf("SKIP test_compute_sha256_matches_known_digest: "
			"sha256sum not available in this environment\n");
	}
	else
	{
		TEST_ASSERT(ok, "sha256sum ran successfully against a real file");
		TEST_ASSERT(actual == expected,
			"digest matches the real, independently-verified value");
	}
	rmdir_recursive(dir);
}

static void	test_download_rejects_non_https_url(void)
{
	membrane_download_error_t	err;
	bool	checksum_verified = false;

	TEST_ASSERT(!membrane_download_file("http://example.com/model.gguf",
		"/tmp/should-never-be-created.gguf", 0, "", NULL, NULL,
		&checksum_verified, &err),
		"a plain http:// URL is rejected before any real network call");
	TEST_ASSERT(err.set && err.code == "INVALID_URL",
		"error code is INVALID_URL");
	TEST_ASSERT(!checksum_verified, "checksum_verified stays false on "
		"an immediate rejection");
}

static void	test_download_rejects_non_url_string(void)
{
	membrane_download_error_t	err;
	bool	checksum_verified = false;

	TEST_ASSERT(!membrane_download_file("not-a-url-at-all",
		"/tmp/should-never-be-created2.gguf", 0, "", NULL, NULL,
		&checksum_verified, &err),
		"a non-URL string is rejected the same way");
	TEST_ASSERT(err.code == "INVALID_URL", "error code is INVALID_URL");
}

int	main(void)
{
	test_disk_space_check_passes_for_small_requirement();
	test_disk_space_check_fails_for_absurd_requirement();
	test_disk_space_check_nonexistent_dir_fails_closed();
	test_compute_sha256_matches_known_digest();
	test_download_rejects_non_https_url();
	test_download_rejects_non_url_string();
	printf("test_download_manager: all tests passed\n");
	return (0);
}
