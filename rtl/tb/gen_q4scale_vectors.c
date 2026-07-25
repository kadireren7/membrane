#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "membrane/f16convert.h"

int	main(int argc, char **argv)
{
	FILE		*fin;
	FILE		*fd;
	FILE		*fid;
	long		blk;

	if (argc != 4)
	{
		fprintf(stderr, "usage: %s x_in.txt d_out.txt id_out.txt\n",
			argv[0]);
		return (1);
	}
	fin = fopen(argv[1], "r");
	fd = fopen(argv[2], "w");
	fid = fopen(argv[3], "w");
	blk = 0;
	while (1)
	{
		uint16_t	x[32];
		int			j;
		int			ok;
		float		amax;
		float		mx;
		float		d;
		float		id;
		uint16_t	d_h;
		uint32_t	id_bits;

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
		d = mx / -8.0f;
		id = (d != 0.0f) ? 1.0f / d : 0.0f;
		d_h = membrane_f32_to_f16(d);
		memcpy(&id_bits, &id, sizeof(id_bits));
		fprintf(fd, "%04x\n", d_h);
		fprintf(fid, "%08x\n", id_bits);
		blk++;
	}
	fclose(fin);
	fclose(fd);
	fclose(fid);
	fprintf(stderr, "generated %ld blocks\n", blk);
	return (0);
}
