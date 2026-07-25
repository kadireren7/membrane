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
		fprintf(stderr, "usage: %s packed_in.txt f16_out.txt\n", argv[0]);
		return (1);
	}
	fin = fopen(argv[1], "r");
	fout = fopen(argv[2], "w");
	blk = 0;
	while (1)
	{
		uint8_t		packed[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
		uint16_t	out[32];
		int			j;
		int			ok;

		ok = 1;
		j = 0;
		while (j < (int)MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES)
		{
			unsigned int	v;

			if (fscanf(fin, "%x", &v) != 1)
			{
				ok = 0;
				break;
			}
			packed[j] = (uint8_t)v;
			j++;
		}
		if (!ok)
			break;
		membrane_simd_q8_0_dequantize(MEMBRANE_SIMD_SCALAR, packed, 32, out);
		j = 0;
		while (j < 32)
		{
			fprintf(fout, "%04x\n", out[j]);
			j++;
		}
		blk++;
	}
	fclose(fin);
	fclose(fout);
	fprintf(stderr, "generated %ld blocks\n", blk);
	return (0);
}
