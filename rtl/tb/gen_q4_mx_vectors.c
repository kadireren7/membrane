#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "membrane/f16convert.h"

int	main(int argc, char **argv)
{
	FILE		*fin;
	FILE		*fout;
	long		blk;

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s x_in.txt mx_f32_out.txt\n", argv[0]);
		return (1);
	}
	fin = fopen(argv[1], "r");
	fout = fopen(argv[2], "w");
	blk = 0;
	while (1)
	{
		uint16_t	x[32];
		int			j;
		int			ok;
		float		amax;
		float		mx;
		uint32_t	mx_bits;

		ok = 1;
		j = 0;
		while (j < 32)
		{
			unsigned int	v;

			if (fscanf(fin, "%x", &v) != 1)
			{
				ok = 0;
				break;
			}
			x[j] = (uint16_t)v;
			j++;
		}
		if (!ok)
			break;
		amax = 0.0f;
		mx = 0.0f;
		j = 0;
		while (j < 32)
		{
			float	v = membrane_f16_to_f32(x[j]);

			if (amax < fabsf(v))
			{
				amax = fabsf(v);
				mx = v;
			}
			j++;
		}
		memcpy(&mx_bits, &mx, sizeof(mx_bits));
		fprintf(fout, "%08x\n", mx_bits);
		blk++;
	}
	fclose(fin);
	fclose(fout);
	fprintf(stderr, "generated %ld mx values\n", blk);
	return (0);
}
