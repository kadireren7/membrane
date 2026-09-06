#ifndef MEMBRANE_CLOCK_COMPAT_H
# define MEMBRANE_CLOCK_COMPAT_H

/*
 * Mega Phase D, PR D5: clock_gettime()/CLOCK_MONOTONIC(_RAW) are
 * POSIX/Linux -- MSVC's <time.h> defines `struct timespec` (C11) but
 * neither clock_gettime() nor any CLOCK_* constant at all (a real,
 * first-attempt Windows CI compile failure: "'CLOCK_MONOTONIC':
 * undeclared identifier", in both src/stats.c and
 * tools/membrane-run/auto_fallback.c).
 *
 * This project's own real usage is narrow: read a monotonic
 * timestamp into a `struct timespec`, always via CLOCK_MONOTONIC or
 * CLOCK_MONOTONIC_RAW -- never CLOCK_REALTIME, never clock_settime(),
 * never a resolution query. QueryPerformanceCounter()/
 * QueryPerformanceFrequency() is the real, standard Win32 monotonic
 * high-resolution timer (guaranteed monotonic and glitch-free since
 * Windows Vista, per Microsoft's own documentation) -- this header
 * provides a real clock_gettime() built on it, and both CLOCK_*
 * constants (their exact values don't matter here: this
 * implementation ignores which one was requested and always returns
 * the same real monotonic reading, since this codebase never
 * distinguishes between the two on any platform it actually measures
 * timing precision on). Both real callers compile completely
 * unchanged on Windows. On every other platform, this header is a
 * pure passthrough to the real <time.h>.
 */

#ifdef _WIN32
# include <time.h>
# include "membrane/windows_lean.h"

# define CLOCK_MONOTONIC		0
# define CLOCK_MONOTONIC_RAW	1

static inline int	clock_gettime(int clk_id, struct timespec *ts)
{
	static LARGE_INTEGER	freq;
	static int				freq_queried;
	LARGE_INTEGER			counter;

	(void)clk_id;
	if (!freq_queried)
	{
		QueryPerformanceFrequency(&freq);
		freq_queried = 1;
	}
	QueryPerformanceCounter(&counter);
	ts->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
	ts->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart)
			* 1000000000LL) / freq.QuadPart);
	return (0);
}
#else
# include <time.h>
#endif

#endif
