#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

static void	emit(FILE *fa, FILE *fb, FILE *fs, FILE *fr, uint32_t a_bits,
	uint32_t b_bits, int sub)
{
	float	a;
	float	b;
	float	r;
	uint32_t	r_bits;

	memcpy(&a, &a_bits, sizeof(a));
	memcpy(&b, &b_bits, sizeof(b));
	r = sub ? (a - b) : (a + b);
	memcpy(&r_bits, &r, sizeof(r_bits));
	fprintf(fa, "%08x\n", a_bits);
	fprintf(fb, "%08x\n", b_bits);
	fprintf(fs, "%d\n", sub);
	fprintf(fr, "%08x\n", r_bits);
}

int	main(int argc, char **argv)
{
	FILE		*fa;
	FILE		*fb;
	FILE		*fs;
	FILE		*fr;
	uint32_t	state;
	long		n;
	long		i;

	if (argc < 5)
	{
		fprintf(stderr, "usage: %s a_out b_out s_out r_out [N]\n", argv[0]);
		return (1);
	}
	n = argc > 5 ? atol(argv[5]) : 100000;
	fa = fopen(argv[1], "w");
	fb = fopen(argv[2], "w");
	fs = fopen(argv[3], "w");
	fr = fopen(argv[4], "w");
	state = 0x13572468u;
	/* Edge cases. */
	emit(fa, fb, fs, fr, 0x00000000u, 0x41080000u, 0); /* 0 + 8.5 */
	emit(fa, fb, fs, fr, 0x41080000u, 0x41080000u, 1); /* 8.5 - 8.5 = 0 */
	emit(fa, fb, fs, fr, 0xC1080000u, 0x41080000u, 0); /* -8.5 + 8.5 = 0 */
	emit(fa, fb, fs, fr, 0x7F800000u, 0x41080000u, 0); /* +Inf + 8.5 */
	emit(fa, fb, fs, fr, 0x7F800000u, 0x7F800000u, 1); /* Inf - Inf -> NaN */
	emit(fa, fb, fs, fr, 0x7FC00001u, 0x41080000u, 0); /* qNaN + 8.5 */
	emit(fa, fb, fs, fr, 0x00000000u, 0x00000000u, 0); /* 0 + 0 */
	emit(fa, fb, fs, fr, 0x80000000u, 0x00000000u, 0); /* -0 + 0 */
	i = 0;
	while (i < n)
	{
		/* Real datapath op: x*id (already computed by the multiplier,
		 * modeled here as a random "reasonable magnitude" value, since
		 * this adder is tested standalone) + 8.5f exactly. */
		uint32_t	x_bits = rng_next(&state);
		uint32_t	x_exp = 118 + (x_bits % 20); /* covers roughly +-64 */

		x_bits = (x_bits & 0x807FFFFFu) | (x_exp << 23);
		emit(fa, fb, fs, fr, x_bits, 0x41080000u, 0);

		/* General random NORMAL pair, both add and subtract, exponent
		 * range constrained to stay within this adder's disclosed
		 * cancellation-depth scope (see membrane_fp_adder.sv's header:
		 * lzc_9 only inspects 9 bits, so exponent differences much
		 * larger than that combined with near-total cancellation are
		 * out of scope, matching what this datapath's own +8.5 step
		 * actually needs). */
		{
			uint32_t	a_bits = rng_next(&state);
			uint32_t	b_bits = rng_next(&state);
			uint32_t	a_exp = 118 + (a_bits % 20);
			uint32_t	b_exp = 118 + (b_bits % 20);
			int			sub = (rng_next(&state) & 1);

			a_bits = (a_bits & 0x807FFFFFu) | (a_exp << 23);
			b_bits = (b_bits & 0x807FFFFFu) | (b_exp << 23);
			emit(fa, fb, fs, fr, a_bits, b_bits, sub);
		}
		i++;
	}
	fclose(fa);
	fclose(fb);
	fclose(fs);
	fclose(fr);
	return (0);
}
