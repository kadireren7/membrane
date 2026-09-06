#ifndef MEMBRANE_POSIX_COMPAT_H
# define MEMBRANE_POSIX_COMPAT_H

/*
 * Mega Phase D, PR D5: a small, focused set of POSIX filesystem
 * functions this project's own real code calls directly
 * (src/backends/backend_file.c). access()/unlink()/fileno()/F_OK all
 * exist on Windows too, via <io.h> -- MSVC's own long-standing,
 * still-functional (if deprecation-warning-flagged) legacy POSIX-
 * compat aliases -- so only fsync() genuinely needs a real substitute
 * here: it has no equivalent name AT ALL on Windows (not even an
 * underscore-prefixed alias sharing its exact semantics); _commit()
 * is the real Win32 analog (flushes a CRT file descriptor's own OS
 * buffers), under a completely different name, which is what this
 * header actually provides.
 */

#ifdef _WIN32
# include <io.h>
# include <stdlib.h>
# include <time.h>
# include "membrane/windows_lean.h"

/* nanosleep() (used by tools/membrane/server.cpp's own graceful-
 * shutdown poll loop and setup_cmd.cpp's own retry-with-backoff
 * helper, both fixed, sub-second millisecond-scale intervals -- never
 * a case needing real nanosecond precision) has no Windows equivalent
 * NAME at all. Sleep() (Win32, millisecond granularity) is the real,
 * standard substitute -- real, disclosed precision loss below
 * millisecond scale (irrelevant to either real caller here, both of
 * which sleep for a fixed 100ms/200ms). `rem` (the real POSIX
 * signature's own "time remaining if interrupted by a signal" output)
 * is never populated -- Windows has no equivalent interruption signal
 * for Sleep() to report, and neither real caller here ever reads it
 * (both pass NULL). */
static inline int	nanosleep(const struct timespec *req, struct timespec *rem)
{
	(void)rem;
	Sleep((DWORD)(req->tv_sec * 1000 + req->tv_nsec / 1000000));
	return (0);
}

/* PATH_MAX is a POSIX macro -- MAX_PATH (windows.h) is the real
 * Windows analog (both name "the longest path this platform's own
 * filesystem APIs promise to handle"). realpath() itself has no
 * Windows equivalent NAME at all; _fullpath() is the real, standard
 * substitute (canonicalizes '.'/'..'  and produces an absolute path,
 * the same real property this project's own call site actually needs
 * -- it does not resolve symlinks the identical way realpath() does,
 * a real, minor, disclosed difference, but Windows symlinks are rare
 * enough in practice that this project's own single real call site
 * -- `membrane model add`'s own path canonicalization -- is not
 * meaningfully weakened by it). _fullpath()'s own argument order is
 * (absPath, relPath, maxLength) -- the reverse of realpath()'s (path,
 * resolved_path) -- this wrapper preserves realpath()'s own real
 * call-site signature exactly, so the one caller needs no change. */
# ifndef PATH_MAX
#  define PATH_MAX MAX_PATH
# endif

static inline char	*realpath(const char *path, char *resolved_path)
{
	return (_fullpath(resolved_path, path, PATH_MAX));
}

/* access()/unlink()/fileno() all resolve via <io.h>'s own legacy
 * POSIX-compat aliases on this MSVC/SDK combination (confirmed
 * directly: this project's own real Windows CI build compiled every
 * call site using them with no error) -- but F_OK itself is NOT
 * defined by <io.h> here (a real, first-attempt Windows CI compile
 * failure: "'F_OK': undeclared identifier"). Its value is standard
 * (0 -- "does this path exist at all") on every platform, POSIX and
 * Windows alike. */
# ifndef F_OK
#  define F_OK 0
# endif

/* dup()/dup2()/close() and STDOUT_FILENO/STDERR_FILENO (used by
 * tools/membrane-run/main.cpp's real --json parse-error stderr
 * capture, redirecting fd 2 to a temp file and back) -- dup/dup2/
 * close are NOT reliably available undecorated the way access/unlink/
 * fileno are (their _-prefixed forms are the only guaranteed names),
 * and the STD*_FILENO macros do not exist on Windows at all (there is
 * no <unistd.h> to define them). Standard fd numbers (0/1/2) are the
 * same constants on every platform, POSIX and Windows alike. */
# define dup _dup
# define dup2 _dup2
# define close _close
# ifndef STDOUT_FILENO
#  define STDOUT_FILENO 1
# endif
# ifndef STDERR_FILENO
#  define STDERR_FILENO 2
# endif

static inline int	fsync(int fd)
{
	return (_commit(fd) == 0 ? 0 : -1);
}

/* isatty()/open()/O_WRONLY (used by tools/membrane/setup_cmd.cpp's real
 * TTY check and its own "redirect stdout to the null device" helper) --
 * same reasoning as dup/dup2/close above: only the _-prefixed forms
 * are guaranteed. The null device itself has a different NAME on
 * Windows ("NUL", not "/dev/null") -- a real path string, not
 * something any function-name shim can paper over, so callers use
 * MEMBRANE_NULL_DEVICE instead of hardcoding either spelling. */
# define isatty _isatty
# define open _open
# ifndef O_WRONLY
#  define O_WRONLY _O_WRONLY
# endif
# define MEMBRANE_NULL_DEVICE "NUL"
#else
# include <fcntl.h>
# include <unistd.h>
# define MEMBRANE_NULL_DEVICE "/dev/null"
#endif

#endif
