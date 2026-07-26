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
	long		n_blocks;
	long		blk;
	int		j;
	uint16_t	x[32];
	FILE		*fx;

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s n_blocks x_out.txt\n", argv[0]);
		return (1);
	}
	n_blocks = atol(argv[1]);
	fx = fopen(argv[2], "w");
	state = 0x2545F491u;
	blk = 0;
	while (blk < n_blocks)
	{
		int	special = (int)(blk % 20);

		j = 0;
		while (j < 32)
		{
			uint32_t	r = rng_next(&state);

			if (special == 0 && (r % 7) == 0)
				x[j] = 0x7C01u; // NaN
			else if (special == 1 && (r % 7) == 0)
				x[j] = 0x7C00u; // +Inf
			else if (special == 2 && (r % 7) == 0)
				x[j] = 0xFC00u; // -Inf
			else if (special == 3 && (r % 5) == 0)
				x[j] = (uint16_t)(r & 0x3FFu); // subnormal
			else if (special == 4 && (r % 5) == 0)
				x[j] = 0x0000u; // zero
			else if (special == 5 && (r % 5) == 0)
				x[j] = 0x8000u; // -zero
			else
			{
				float	fv = ((float)((r >> 8) & 0xFFFF) / 65535.0f
						- 0.5f) * 16.0f;

				x[j] = membrane_f32_to_f16(fv);
			}
			j++;
		}
		j = 0;
		while (j < 32)
		{
			fprintf(fx, "%04x\n", x[j]);
			j++;
		}
		blk++;
		if (blk % 20000 == 0)
			fprintf(stderr, "  generated %ld / %ld blocks\n", blk, n_blocks);
	}
	fclose(fx);
	fprintf(stderr, "generated %ld blocks\n", blk);
	return (0);
}
