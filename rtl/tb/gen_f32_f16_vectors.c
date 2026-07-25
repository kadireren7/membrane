#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "membrane/f16convert.h"

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

int	main(void)
{
	unsigned int	h;
	uint32_t	fbits;
	uint16_t	got;
	float		f;
	uint32_t	state;
	long		i;

	/* 1) roundtrip: every F16 value widened via membrane_f16_to_f32, then
	 * narrowed back -- must reproduce the original bits for every
	 * non-NaN pattern (matches test_f16convert.c's own methodology). */
	h = 0;
	while (h < 65536)
	{
		if (!((h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0))
		{
			f = membrane_f16_to_f32((uint16_t)h);
			memcpy(&fbits, &f, sizeof(fbits));
			got = membrane_f32_to_f16(f);
			printf("%08x %04x\n", fbits, got);
		}
		h++;
	}
	/* 2) halfway-rounding-specific F32 values: construct F32 patterns
	 * whose low 13 mantissa bits are exactly the round-to-even boundary
	 * (0x1000), for both even and odd retained mantissa, across a
	 * spread of exponents. */
	{
		int	e;
		int	mtop;

		e = 100;
		while (e <= 150)
		{
			mtop = 0;
			while (mtop < 1024)
			{
				fbits = ((uint32_t)e << 23)
					| ((uint32_t)mtop << 13) | 0x1000u;
				memcpy(&f, &fbits, sizeof(f));
				got = membrane_f32_to_f16(f);
				printf("%08x %04x\n", fbits, got);
				mtop += 173;
			}
			e += 7;
		}
	}
	/* 3) random F32 patterns, finite and non-finite, both signs. */
	state = 0xABCDEF01u;
	i = 0;
	while (i < 200000)
	{
		fbits = rng_next(&state);
		memcpy(&f, &fbits, sizeof(f));
		got = membrane_f32_to_f16(f);
		printf("%08x %04x\n", fbits, got);
		i++;
	}
	return (0);
}
