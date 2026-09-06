#ifndef MEMBRANE_WINDOWS_LEAN_H
# define MEMBRANE_WINDOWS_LEAN_H

/*
 * Mega Phase D, PR D5: every one of this project's own real
 * <windows.h> includes goes through this header instead, never
 * <windows.h> directly -- windows.h defines min()/max() as raw
 * preprocessor macros unless NOMINMAX is defined first, which breaks
 * any C++ std::min/std::max call in the SAME translation unit (a real,
 * first-attempt Windows CI compile failure in decode_loop.cpp:
 * "error C2589: '(': illegal token on right side of '::'" -- the
 * exact, well-known symptom of `max(` silently macro-expanding where
 * `std::max(` was written). NOMINMAX must be defined before windows.h
 * is included ANYWHERE in a translation unit, so centralizing the
 * include here (rather than repeating a #define-then-#include pair at
 * every one of this project's real windows.h call sites) is the only
 * way to guarantee that ordering everywhere. WIN32_LEAN_AND_MEAN is
 * additionally defined as good, standard hygiene (excludes rarely-
 * needed subsystems -- e.g. sockets/GDI headers this project never
 * uses -- reducing both compile time and the surface area for a
 * similar unrelated macro collision).
 */

# ifndef NOMINMAX
#  define NOMINMAX
# endif
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# include <windows.h>

#endif
