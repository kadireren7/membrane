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
#else
# include <unistd.h>
#endif

#endif
