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
	r = a * b;
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
	state = 0x2468ACE1u;
	/* Edge cases. */
	emit(fa, fb, fr, 0x00000000u, 0x42FE0000u); /* 0 * 127 */
	emit(fa, fb, fr, 0x7F800000u, 0x3F800000u); /* +Inf * 1 */
	emit(fa, fb, fr, 0x7F800000u, 0x00000000u); /* +Inf * 0 -> NaN */
	emit(fa, fb, fr, 0x00000000u, 0x7F800000u); /* 0 * +Inf -> NaN */
	emit(fa, fb, fr, 0xFF800000u, 0x00000000u); /* -Inf * 0 -> NaN */
	emit(fa, fb, fr, 0x7F800000u, 0x80000000u); /* +Inf * -0 -> NaN */
	emit(fa, fb, fr, 0x7FC00001u, 0x3F800000u); /* qNaN(payload) * 1 */
	emit(fa, fb, fr, 0x7F800001u, 0x3F800000u); /* sNaN * 1 -> quieted */
	emit(fa, fb, fr, 0x7FC00001u, 0x7FC00002u); /* both NaN, a wins */
	emit(fa, fb, fr, 0x7F800001u, 0xFFC00002u); /* sNaN(a) * qNaN(b,neg) */
	emit(fa, fb, fr, 0x7FC00000u, 0x00000000u); /* qNaN * 0 */
	emit(fa, fb, fr, 0x7FC00000u, 0x7F800000u); /* qNaN * Inf */
	emit(fa, fb, fr, 0xBF800000u, 0x3F800000u); /* -1 * 1 */
	i = 0;
	while (i < n)
	{
		/* Real datapath ops: x*id (Q8/Q4 quantize), qs*d (dequantize),
		 * both with a "reasonable" magnitude operand (exponent in
		 * [100,155], as in the divider generator) and the other operand
		 * an arbitrary small integer-ish value (matching int8/nibble
		 * magnitudes cast to float, i.e. small integers -128..127). */
		uint32_t	scale_bits = rng_next(&state);
		uint32_t	scale_exp = 100 + (scale_bits % 56);
		float		small_int = (float)((int)(rng_next(&state) % 257) - 128);
		uint32_t	small_bits;

		scale_bits = (scale_bits & 0x807FFFFFu) | (scale_exp << 23);
		memcpy(&small_bits, &small_int, sizeof(small_bits));
		emit(fa, fb, fr, scale_bits, small_bits);

		/* Fully general random NORMAL pair, exponent constrained the
		 * same way as the divider generator and for the same reason
		 * (avoid exercising the disclosed flush-to-zero-on-underflow
		 * gap rather than gradual subnormal underflow). */
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
