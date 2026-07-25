#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "membrane/quant_simd.h"

int	main(int argc, char **argv)
{
	FILE		*fin;
	FILE		*fout;
	long		blk;

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s x_in.txt packed_out.txt\n", argv[0]);
		return (1);
	}
	fin = fopen(argv[1], "r");
	fout = fopen(argv[2], "w");
	blk = 0;
	while (1)
	{
		uint16_t	x[32];
		uint8_t		packed[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
		int			j;
		int			ok;

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
		membrane_simd_q8_0_quantize(MEMBRANE_SIMD_SCALAR, x, 32, packed);
		j = 0;
		while (j < (int)MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES)
		{
			fprintf(fout, "%02x\n", packed[j]);
			j++;
		}
		blk++;
	}
	fclose(fin);
	fclose(fout);
	fprintf(stderr, "generated %ld blocks\n", blk);
	return (0);
}
