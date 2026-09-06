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

static inline int	fsync(int fd)
{
	return (_commit(fd) == 0 ? 0 : -1);
}
#else
# include <unistd.h>
#endif

#endif
