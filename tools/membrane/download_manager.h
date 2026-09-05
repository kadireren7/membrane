#ifndef MEMBRANE_DOWNLOAD_MANAGER_H
# define MEMBRANE_DOWNLOAD_MANAGER_H

# include <cstdint>
# include <string>

/*
 * Mega Phase D, PR D1: real HTTPS downloads for `membrane model
 * install`. Uses libcurl's easy API (already present as a real system
 * dependency: libcurl4-openssl-dev is installed on this project's own
 * dev host and its runtime library, libcurl4t64, is a common, already-
 * present Debian package -- chosen over adding OpenSSL directly to
 * cpp-httplib specifically because it required a NEW system dependency
 * this session had no root access to install, whereas libcurl's dev
 * headers were already available; libcurl is also the de facto
 * standard C library for exactly this use case). SHA-256 verification
 * shells out to `sha256sum` (GNU coreutils, a Debian "Essential:yes"
 * package -- present on every real Debian-family install by
 * definition, unlike `curl`, which this project already established
 * it never assumes at runtime) via subprocess.h's existing, safe
 * (execvp, no shell) runner -- if genuinely absent (exit 127), checksum
 * verification degrades to a disclosed WARN rather than a hard
 * failure, the same graceful-degradation precedent doctor_cmd.cpp's
 * own systemctl-availability check already established.
 *
 * Section 6 of the task ("download safety"): HTTPS only (rejected
 * before libcurl is ever invoked, and also enforced via
 * CURLOPT_PROTOCOLS_STR so a malicious redirect chain can never
 * downgrade to plaintext), a `.partial` temp file with atomic
 * rename() only after every check passes, real file-size verification,
 * real SHA-256 verification when the caller provides one, disk-space
 * precheck, and a partial file is only ever deleted on a genuine
 * validation MISMATCH (a resumable retry keeps a partial file that
 * merely hasn't finished yet).
 */

typedef struct s_membrane_download_error
{
	bool		set;
	std::string	code;		/* "INVALID_URL", "DISK_FULL", "HTTP_ERROR",
							 * "SIZE_MISMATCH", "CHECKSUM_MISMATCH",
							 * "IO_ERROR" */
	std::string	message;
}	membrane_download_error_t;

/* Called periodically during a real transfer. total_bytes is 0 if the
 * server never reported Content-Length (a real, disclosed possibility
 * with some CDN redirect chains) -- callers must handle that case
 * (e.g. an indeterminate progress display) rather than dividing by
 * zero. */
typedef void	(*membrane_download_progress_cb)(uint64_t downloaded_bytes,
					uint64_t total_bytes, void *user_data);

/* Real statvfs()-based free-space check against the filesystem holding
 * dest_dir (which must already exist). A real, small safety margin
 * (not just the exact byte count) is required to pass, since a
 * genuinely-full disk mid-download is worse than a slightly early,
 * clear refusal. */
bool	membrane_download_check_disk_space(const std::string &dest_dir,
			uint64_t required_bytes, membrane_download_error_t *err);

/* Downloads url to dest_path. expected_size_bytes == 0 skips size
 * verification (only used when the catalog genuinely has no known
 * size); expected_sha256 empty skips checksum verification outright.
 * progress_cb may be NULL. Returns false and fills *err on any failure
 * -- dest_path itself is NEVER left in a partially-written state: only
 * dest_path + ".partial" is ever touched before the final, atomic
 * rename().
 *
 * On SUCCESS (true), *out_checksum_verified (may be NULL if the caller
 * doesn't care) reports whether the checksum was actually confirmed:
 * false when `expected_sha256` was empty, OR when `sha256sum` was
 * genuinely unavailable on this host (a disclosed degradation, not a
 * hidden one -- doctor_cmd.cpp's own systemctl-availability check
 * established this same "still succeed, but say so" precedent). The
 * downloaded file itself is always real and, when expected_size_bytes
 * was given, always size-verified regardless of checksum
 * availability. */
bool	membrane_download_file(const std::string &url,
			const std::string &dest_path, uint64_t expected_size_bytes,
			const std::string &expected_sha256,
			membrane_download_progress_cb progress_cb, void *user_data,
			bool *out_checksum_verified, membrane_download_error_t *err);

/* Real sha256sum-based checksum computation, exposed separately so
 * callers (and tests) can verify an already-downloaded file without
 * re-downloading it. Returns false (never crashes/guesses) if
 * `sha256sum` is genuinely unavailable (exit 127) or the file can't be
 * read -- check *err for which. */
bool	membrane_compute_sha256(const std::string &path, std::string *out_hex,
			membrane_download_error_t *err);

#endif
