#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

static void	emit(FILE *fa, FILE *fb, FILE *fr, uint32_t a_bits, uint32_t b_bits)
{
	float	a;
	float	b;
	float	r;
	uint32_t	r_bits;

	memcpy(&a, &a_bits, sizeof(a));
	memcpy(&b, &b_bits, sizeof(b));
	r = a / b;
	memcpy(&r_bits, &r, sizeof(r_bits));
	fprintf(fa, "%08x\n", a_bits);
	fprintf(fb, "%08x\n", b_bits);
	fprintf(fr, "%08x\n", r_bits);
}

int	main(int argc, char **argv)
{
	FILE		*fa;
	FILE		*fb;
	FILE		*fr;
	uint32_t	state;
	long		n;
	long		i;

	if (argc < 4)
	{
		fprintf(stderr, "usage: %s a_out.txt b_out.txt r_out.txt [N]\n",
			argv[0]);
		return (1);
	}
	n = argc > 4 ? atol(argv[4]) : 100000;
	fa = fopen(argv[1], "w");
	fb = fopen(argv[2], "w");
	fr = fopen(argv[3], "w");
	state = 0x1357ACE1u;
	/* Edge cases first. */
	emit(fa, fb, fr, 0x00000000u, 0x42FE0000u); /* 0/127 */
	emit(fa, fb, fr, 0x7F800000u, 0x42FE0000u); /* +Inf/127 */
	emit(fa, fb, fr, 0x42FE0000u, 0x7F800000u); /* 127/+Inf */
	emit(fa, fb, fr, 0x3F800000u, 0x7F800000u); /* 1/+Inf */
	emit(fa, fb, fr, 0x3F800000u, 0x00000000u); /* 1/0 */
	emit(fa, fb, fr, 0x00000000u, 0x00000000u); /* 0/0 -> NaN */
	emit(fa, fb, fr, 0x80000000u, 0x00000000u); /* -0/0 -> NaN */
	emit(fa, fb, fr, 0x7F800000u, 0x7F800000u); /* Inf/Inf -> NaN */
	emit(fa, fb, fr, 0xFF800000u, 0x7F800000u); /* -Inf/Inf -> NaN */
	emit(fa, fb, fr, 0x7FC00001u, 0x3F800000u); /* qNaN / 1.0 */
	emit(fa, fb, fr, 0x3F800000u, 0x7FC00001u); /* 1.0 / qNaN */
	emit(fa, fb, fr, 0x7F800001u, 0x3F800000u); /* sNaN / 1.0 -> quieted */
	i = 0;
	while (i < n)
	{
		float		amax;
		uint32_t	amax_bits;
		uint32_t	r = rng_next(&state);
		float		mant = (float)((r >> 8) & 0xFFFFFF) / (float)0x1000000;

		amax = mant * 131072.0f;
		memcpy(&amax_bits, &amax, sizeof(amax_bits));
		/* d = amax/127, id = 127/amax -- the real datapath's own ops. */
		emit(fa, fb, fr, amax_bits, 0x42FE0000u);
		emit(fa, fb, fr, 0x42FE0000u, amax_bits);
		/* mx/-8, 1/d -- Q4's ops (mx/-8 is exact power-of-two, still
		 * worth covering through the general divider). */
		emit(fa, fb, fr, amax_bits, 0xC1000000u);
		emit(fa, fb, fr, 0x3F800000u, amax_bits);
		/* Fully general random NORMAL pair (sign varies, exponent forced
		 * into [100,155] (both comfortably away from 0/255) so the
		 * result stays within F32's NORMAL range too (exp_a-exp_b
		 * bounded to [-55,55], giving a result exponent safely inside
		 * [72,182]) -- subnormal operands, and results that underflow
		 * into subnormal range, are a disclosed out-of-scope limitation
		 * of this divider (flush-to-zero instead of gradual underflow,
		 * see membrane_fp_divider.sv's header), so vectors here
		 * deliberately stay within its actual claimed domain rather than
		 * exercising a known, already-documented gap. */
		{
			uint32_t	a_bits = rng_next(&state);
			uint32_t	b_bits = rng_next(&state);
			uint32_t	a_exp = 100 + (a_bits % 56);
			uint32_t	b_exp = 100 + (b_bits % 56);

			a_bits = (a_bits & 0x807FFFFFu) | (a_exp << 23);
			b_bits = (b_bits & 0x807FFFFFu) | (b_exp << 23);
			emit(fa, fb, fr, a_bits, b_bits);
		}
		i++;
	}
	fclose(fa);
	fclose(fb);
	fclose(fr);
	return (0);
}
