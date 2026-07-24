#include <math.h>
#include <string.h>

#include "membrane/f16convert.h"
#include "test_helpers.h"

static int	is_nan_f16(uint16_t h)
{
	return (((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF) != 0);
}

/* Exhaustive: every one of the 65536 possible half bit patterns must
 * survive half->float->half exactly, except NaN payload bits (only
 * "is NaN" is required to survive). This is the primary correctness
 * proof for the hand-written conversion routines. */
static void	test_exhaustive_roundtrip(void)
{
	uint32_t	h;
	float		f;
	uint16_t	back;
	int			mismatches;

	mismatches = 0;
	h = 0;
	while (h < 65536)
	{
		f = membrane_f16_to_f32((uint16_t)h);
		back = membrane_f32_to_f16(f);
		if (is_nan_f16((uint16_t)h))
		{
			if (!is_nan_f16(back))
				mismatches++;
		}
		else if (back != (uint16_t)h)
			mismatches++;
		h++;
	}
	TEST_ASSERT(mismatches == 0, "all 65536 half values round-trip exactly");
}

static void	test_known_values(void)
{
	TEST_ASSERT(membrane_f16_to_f32(0x3C00) == 1.0f, "0x3C00 == 1.0");
	TEST_ASSERT(membrane_f16_to_f32(0xBC00) == -1.0f, "0xBC00 == -1.0");
	TEST_ASSERT(membrane_f16_to_f32(0x4000) == 2.0f, "0x4000 == 2.0");
	TEST_ASSERT(membrane_f16_to_f32(0x0000) == 0.0f, "0x0000 == +0.0");
	TEST_ASSERT(membrane_f16_to_f32(0x8000) == -0.0f, "0x8000 == -0.0");
	TEST_ASSERT(membrane_f16_to_f32(0x3800) == 0.5f, "0x3800 == 0.5");
	TEST_ASSERT(membrane_f32_to_f16(1.0f) == 0x3C00, "1.0 -> 0x3C00");
	TEST_ASSERT(membrane_f32_to_f16(-1.0f) == 0xBC00, "-1.0 -> 0xBC00");
	TEST_ASSERT(membrane_f32_to_f16(0.0f) == 0x0000, "0.0 -> 0x0000");
}

static void	test_inf_and_nan(void)
{
	float	inf;
	float	ninf;
	float	nan;

	inf = 1.0f / 0.0f;
	ninf = -1.0f / 0.0f;
	nan = 0.0f / 0.0f;
	TEST_ASSERT(membrane_f16_to_f32(0x7C00) == inf, "0x7C00 == +inf");
	TEST_ASSERT(membrane_f16_to_f32(0xFC00) == ninf, "0xFC00 == -inf");
	TEST_ASSERT(isnan(membrane_f16_to_f32(0x7E00)), "0x7E00 is NaN");
	TEST_ASSERT(membrane_f32_to_f16(inf) == 0x7C00, "+inf -> 0x7C00");
	TEST_ASSERT(membrane_f32_to_f16(ninf) == 0xFC00, "-inf -> 0xFC00");
	TEST_ASSERT(is_nan_f16(membrane_f32_to_f16(nan)), "NaN -> f16 NaN");
	/* Overflow: a finite float too large for f16 rounds to infinity. */
	TEST_ASSERT(membrane_f32_to_f16(1.0e30f) == 0x7C00, "huge finite -> +inf");
	TEST_ASSERT(membrane_f32_to_f16(-1.0e30f) == 0xFC00, "huge finite -> -inf");
	/* Underflow: a tiny finite float rounds to (signed) zero. */
	TEST_ASSERT(membrane_f32_to_f16(1.0e-30f) == 0x0000, "tiny finite -> +0.0");
}

static void	test_subnormals(void)
{
	float	smallest_sub;
	float	largest_sub;

	smallest_sub = membrane_f16_to_f32(0x0001);
	largest_sub = membrane_f16_to_f32(0x03FF);
	TEST_ASSERT(smallest_sub > 0.0f && smallest_sub < 1.0e-4f,
		"smallest subnormal is tiny and positive");
	TEST_ASSERT(largest_sub > smallest_sub, "subnormals are ordered");
	TEST_ASSERT(membrane_f32_to_f16(smallest_sub) == 0x0001,
		"smallest subnormal round-trips");
	TEST_ASSERT(membrane_f32_to_f16(largest_sub) == 0x03FF,
		"largest subnormal round-trips");
}

int	main(void)
{
	test_known_values();
	test_inf_and_nan();
	test_subnormals();
	test_exhaustive_roundtrip();
	return (0);
}
