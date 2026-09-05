#include "download_manager.h"
#include "fs_util.h"
#include "subprocess.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <sys/statvfs.h>
#include <sys/stat.h>
#include <unistd.h>

#include <curl/curl.h>

static void	set_err(membrane_download_error_t *err, const std::string &code,
				const std::string &message)
{
	err->set = true;
	err->code = code;
	err->message = message;
}

bool	membrane_download_check_disk_space(const std::string &dest_dir,
			uint64_t required_bytes, membrane_download_error_t *err)
{
	*err = membrane_download_error_t();
	struct statvfs	st;

	if (statvfs(dest_dir.c_str(), &st) != 0)
	{
		set_err(err, "IO_ERROR", std::string("could not stat filesystem "
			"for '") + dest_dir + "': " + strerror(errno));
		return (false);
	}
	uint64_t	available = (uint64_t)st.f_bavail * (uint64_t)st.f_frsize;
	/* A real, small safety margin (256 MiB) on top of the exact byte
	 * count -- a download that lands EXACTLY at the last free byte
	 * still leaves the filesystem with zero headroom for anything
	 * else (journal files, the registry's own atomic-write temp file,
	 * etc.), which is worse than a slightly early, clear refusal. */
	uint64_t	margin = 256ull * 1024 * 1024;
	uint64_t	needed = required_bytes + margin;

	if (needed < required_bytes)	/* overflow guard */
		needed = UINT64_MAX;
	if (available < needed)
	{
		set_err(err, "DISK_FULL", "need " + std::to_string(required_bytes)
			+ " bytes (+" + std::to_string(margin) + " safety margin), only "
			+ std::to_string(available) + " available on this filesystem");
		return (false);
	}
	return (true);
}

bool	membrane_compute_sha256(const std::string &path, std::string *out_hex,
			membrane_download_error_t *err)
{
	*err = membrane_download_error_t();
	membrane_subprocess_result_t	res;

	if (!membrane_run_subprocess({"sha256sum", path}, &res, 120))
	{
		set_err(err, "IO_ERROR", "could not spawn sha256sum");
		return (false);
	}
	if (res.exit_code == 127)
	{
		set_err(err, "TOOL_UNAVAILABLE", "sha256sum is not available in "
			"this environment -- checksum verification skipped");
		return (false);
	}
	if (res.exit_code != 0)
	{
		set_err(err, "IO_ERROR", "sha256sum exited " + std::to_string(
			res.exit_code) + ": " + res.stderr_output);
		return (false);
	}
	/* Output is "<hex>  <path>\n" -- the hex digest is always exactly
	 * 64 lowercase hex characters, real coreutils behavior, not
	 * guessed. */
	if (res.stdout_output.size() < 64)
	{
		set_err(err, "IO_ERROR", "unexpected sha256sum output: "
			+ res.stdout_output);
		return (false);
	}
	*out_hex = res.stdout_output.substr(0, 64);
	return (true);
}

namespace
{

struct s_write_ctx
{
	FILE	*f;
	uint64_t	written;
};

size_t	write_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
	s_write_ctx	*ctx = (s_write_ctx *)userdata;
	size_t		n = size * nmemb;
	size_t		wrote = fwrite(ptr, 1, n, ctx->f);

	ctx->written += wrote;
	return (wrote);
}

struct s_progress_ctx
{
	membrane_download_progress_cb	cb;
	void							*user_data;
	uint64_t						resume_offset;
};

int	xfer_cb(void *clientp, curl_off_t dltotal, curl_off_t dlnow,
			curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
	s_progress_ctx	*ctx = (s_progress_ctx *)clientp;

	if (ctx->cb != NULL)
	{
		uint64_t	total = dltotal > 0
				? (uint64_t)dltotal + ctx->resume_offset : 0;
		ctx->cb((uint64_t)dlnow + ctx->resume_offset, total, ctx->user_data);
	}
	return (0);
}

}	/* anonymous namespace */

bool	membrane_download_file(const std::string &url,
			const std::string &dest_path, uint64_t expected_size_bytes,
			const std::string &expected_sha256,
			membrane_download_progress_cb progress_cb, void *user_data,
			bool *out_checksum_verified, membrane_download_error_t *err)
{
	*err = membrane_download_error_t();
	if (out_checksum_verified != NULL)
		*out_checksum_verified = false;
	if (url.rfind("https://", 0) != 0)
	{
		set_err(err, "INVALID_URL", "only https:// URLs are ever "
			"downloaded (Section 6 of the task: 'HTTPS only')");
		return (false);
	}

	std::string	dir = dest_path.substr(0, dest_path.find_last_of('/'));
	membrane_fs_error_t	fs_err;

	if (!dir.empty() && !membrane_mkdir_parents(dir, &fs_err))
	{
		set_err(err, fs_err.code, fs_err.message);
		return (false);
	}
	if (expected_size_bytes > 0
		&& !membrane_download_check_disk_space(dir.empty() ? "." : dir,
			expected_size_bytes, err))
		return (false);

	std::string	partial_path = dest_path + ".partial";
	uint64_t	resume_offset = 0;
	struct stat	st;

	if (stat(partial_path.c_str(), &st) == 0)
		resume_offset = (uint64_t)st.st_size;

	FILE	*f = fopen(partial_path.c_str(), resume_offset > 0 ? "ab" : "wb");

	if (f == NULL)
	{
		set_err(err, "IO_ERROR", std::string("could not open '")
			+ partial_path + "' for writing: " + strerror(errno));
		return (false);
	}

	CURL	*curl = curl_easy_init();

	if (curl == NULL)
	{
		fclose(f);
		set_err(err, "IO_ERROR", "curl_easy_init() failed");
		return (false);
	}

	s_write_ctx	wctx = {f, 0};
	s_progress_ctx	pctx = {progress_cb, user_data, resume_offset};
	char	curl_err_buf[CURL_ERROR_SIZE] = {0};

	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "https");
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wctx);
	curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
	curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xfer_cb);
	curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pctx);
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "membrane-model-install/1.0");
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_buf);
	if (resume_offset > 0)
		curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE,
			(curl_off_t)resume_offset);

	CURLcode	rc = curl_easy_perform(curl);
	long		http_status = 0;

	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
	curl_easy_cleanup(curl);
	fclose(f);

	if (rc != CURLE_OK)
	{
		/* A genuine transfer error leaves the .partial file in place
		 * on purpose -- it may still be a valid, resumable prefix
		 * (e.g. a transient network drop), and only a later size/
		 * checksum MISMATCH (below) means the bytes on disk are
		 * actually wrong. */
		set_err(err, "HTTP_ERROR", std::string("download failed (curl: ")
			+ (curl_err_buf[0] != '\0' ? curl_err_buf
				: curl_easy_strerror(rc)) + ", http_status="
			+ std::to_string(http_status) + ") -- partial file kept at '"
			+ partial_path + "' for a resumable retry");
		return (false);
	}

	struct stat	final_st;

	if (stat(partial_path.c_str(), &final_st) != 0)
	{
		set_err(err, "IO_ERROR", "download reported success but "
			+ partial_path + " is missing");
		return (false);
	}
	if (expected_size_bytes > 0
		&& (uint64_t)final_st.st_size != expected_size_bytes)
	{
		/* A real size mismatch means the bytes on disk can never
		 * become correct by resuming further -- delete rather than
		 * leave a permanently-broken .partial a future retry would
		 * just resume from and still get wrong. */
		unlink(partial_path.c_str());
		set_err(err, "SIZE_MISMATCH", "expected " + std::to_string(
			expected_size_bytes) + " bytes, got " + std::to_string(
			final_st.st_size) + " -- partial file deleted");
		return (false);
	}
	bool	checksum_verified = false;

	if (!expected_sha256.empty())
	{
		std::string	actual_hex;
		membrane_download_error_t	sha_err;

		if (!membrane_compute_sha256(partial_path, &actual_hex, &sha_err))
		{
			if (sha_err.code != "TOOL_UNAVAILABLE")
			{
				*err = sha_err;
				return (false);
			}
			/* Disclosed degradation, not a hard failure -- see this
			 * module's own header comment. The file itself is still
			 * real and size-verified; we just could not additionally
			 * verify its content hash. checksum_verified stays false. */
		}
		else if (actual_hex != expected_sha256)
		{
			unlink(partial_path.c_str());
			set_err(err, "CHECKSUM_MISMATCH", "expected sha256 "
				+ expected_sha256 + ", got " + actual_hex
				+ " -- partial file deleted");
			return (false);
		}
		else
			checksum_verified = true;
	}
	if (rename(partial_path.c_str(), dest_path.c_str()) != 0)
	{
		set_err(err, "IO_ERROR", std::string("rename() to '") + dest_path
			+ "' failed: " + strerror(errno));
		return (false);
	}
	if (out_checksum_verified != NULL)
		*out_checksum_verified = checksum_verified;
	return (true);
}
