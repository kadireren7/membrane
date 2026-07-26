#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

/* Matches sat_i8_from_rounded_f32(rintf(f)) in src/quant/quant_simd.c */
static int8_t	sat_i8_from_rounded_f32(float r)
{
	int32_t	iv;

	if (!isfinite(r) || r >= 2147483648.0f || r < -2147483648.0f)
		return (INT8_MIN);
	iv = (int32_t)r;
	if (iv > 127)
		return (127);
	if (iv < -128)
		return (-128);
	return ((int8_t)iv);
}

int	main(int argc, char **argv)
{
	FILE		*f_i9;
	FILE		*f_i9r;
	FILE		*f_r8;
	FILE		*f_r8r;
	FILE		*f_t;
	FILE		*f_tr;
	uint32_t	state;
	long		n;
	long		i;

	if (argc < 7)
	{
		fprintf(stderr, "usage: %s i9_in i9_r r8_in r8_r t_in t_r [N]\n",
			argv[0]);
		return (1);
	}
	n = argc > 7 ? atol(argv[7]) : 20000;
	f_i9 = fopen(argv[1], "w");
	f_i9r = fopen(argv[2], "w");
	f_r8 = fopen(argv[3], "w");
	f_r8r = fopen(argv[4], "w");
	f_t = fopen(argv[5], "w");
	f_tr = fopen(argv[6], "w");
	state = 0xABCD1234u;
	/* int9_to_f32: exhaustive over the full -256..255 range. */
	{
		int	v;

		v = -256;
		while (v <= 255)
		{
			float		f = (float)v;
			uint32_t	fb;

			memcpy(&fb, &f, sizeof(fb));
			fprintf(f_i9, "%03x\n", v & 0x1FF);
			fprintf(f_i9r, "%08x\n", fb);
			v++;
		}
	}
	/* f32_round_sat_to_i8: edge cases + random. */
	{
		uint32_t	edge[] = {0x00000000u, 0x7F800000u, 0xFF800000u,
			0x7FC00001u, 0x42FE0000u, 0xC2FE0000u, 0x3F000000u,
			0x3EFFFFFFu, 0xBF000000u, 0x43000000u, 0xC3000000u,
			0x437F0000u, 0x42FF0000u};
		int			ne = 13;
		int			e;

		e = 0;
		while (e < ne)
		{
			float	f;
			int8_t	r;
			uint32_t	fb = edge[e];

			memcpy(&f, &fb, sizeof(f));
			r = sat_i8_from_rounded_f32(rintf(f));
			fprintf(f_r8, "%08x\n", fb);
			fprintf(f_r8r, "%02x\n", (uint8_t)r);
			e++;
		}
	}
	i = 0;
	while (i < n)
	{
		uint32_t	bits = rng_next(&state);
		uint32_t	exp = 100 + (bits % 60);
		float		f;
		int8_t		r;

		bits = (bits & 0x807FFFFFu) | (exp << 23);
		memcpy(&f, &bits, sizeof(f));
		r = sat_i8_from_rounded_f32(rintf(f));
		fprintf(f_r8, "%08x\n", bits);
		fprintf(f_r8r, "%02x\n", (uint8_t)r);
		i++;
	}
	/* f32_trunc_to_i32: edge cases + random (reasonable range, matching
	 * what x*id+8.5 actually produces in the real datapath). */
	{
		uint32_t	edge[] = {0x00000000u, 0x7F800000u, 0xFF800000u,
			0x7FC00001u, 0x41880000u, 0xC1880000u, 0x3F000000u,
			0xBF000000u};
		int			ne = 8;
		int			e;

		e = 0;
		while (e < ne)
		{
			float		f;
			int32_t		r;
			uint32_t	fb = edge[e];

			memcpy(&f, &fb, sizeof(f));
			if (!isfinite(f) || f >= 2147483648.0f || f < -2147483648.0f)
				r = INT32_MIN;
			else
				r = (int32_t)f;
			fprintf(f_t, "%08x\n", fb);
			fprintf(f_tr, "%08x\n", (uint32_t)r);
			e++;
		}
	}
	i = 0;
	while (i < n)
	{
		uint32_t	bits = rng_next(&state);
		uint32_t	exp = 118 + (bits % 20);
		float		f;
		int32_t		r;

		bits = (bits & 0x807FFFFFu) | (exp << 23);
		memcpy(&f, &bits, sizeof(f));
		r = (int32_t)f;
		fprintf(f_t, "%08x\n", bits);
		fprintf(f_tr, "%08x\n", (uint32_t)r);
		i++;
	}
	return (0);
}
