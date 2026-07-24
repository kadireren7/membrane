#ifndef MEMBRANE_F16CONVERT_H
# define MEMBRANE_F16CONVERT_H

# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * IEEE-754 binary16 <-> binary32 conversion (Phase 3.1). Dependency-free,
 * bit-exact for every finite/inf/zero value (every finite half value is
 * exactly representable in float, and the reverse direction rounds to
 * nearest-even, matching the standard hardware behaviour). NaN inputs
 * convert to a quiet NaN with at least one mantissa bit preserved so the
 * "is NaN" property survives; exact NaN payload bits are not preserved.
 *
 * Correctness is exercised exhaustively in tests/unit/test_f16convert.c
 * (all 65536 uint16 patterns, half->float->half must reproduce the
 * original bits for every non-NaN value).
 */

float		membrane_f16_to_f32(uint16_t h);
uint16_t	membrane_f32_to_f16(float f);

# ifdef __cplusplus
}
# endif

#endif
