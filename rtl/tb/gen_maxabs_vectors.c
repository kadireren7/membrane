#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "membrane/f16convert.h"

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

int	main(int argc, char **argv)
{
	uint32_t	state;
	long		blk;
	int		j;
	uint16_t	x[32];
	float		amax;
	float		v;
	uint16_t	amax_f16;
	FILE		*fx;
	FILE		*fa;

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s x_out.txt amax_out.txt\n", argv[0]);
		return (1);
	}
	fx = fopen(argv[1], "w");
	fa = fopen(argv[2], "w");
	state = 0x9E3779B9u;
	blk = 0;
	while (blk < 20000)
	{
		int	special = (int)(blk % 20);

		j = 0;
		while (j < 32)
		{
			uint32_t	r = rng_next(&state);

			if (special == 0 && (r % 7) == 0)
				x[j] = 0x7C01u;
			else if (special == 1 && (r % 7) == 0)
				x[j] = 0x7C00u;
			else if (special == 2 && (r % 7) == 0)
				x[j] = 0xFC00u;
			else
			{
				float	fv = ((float)((r >> 8) & 0xFFFF) / 65535.0f
						- 0.5f) * 8.0f;

				x[j] = membrane_f32_to_f16(fv);
			}
			j++;
		}
		amax = 0.0f;
		j = 0;
		while (j < 32)
		{
			v = fabsf(membrane_f16_to_f32(x[j]));
			if (v > amax)
				amax = v;
			j++;
		}
		amax_f16 = membrane_f32_to_f16(amax);
		j = 0;
		while (j < 32)
		{
			fprintf(fx, "%04x\n", x[j]);
			j++;
		}
		fprintf(fa, "%04x\n", amax_f16);
		blk++;
	}
	fclose(fx);
	fclose(fa);
	return (0);
}
