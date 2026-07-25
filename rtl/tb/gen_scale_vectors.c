#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "membrane/f16convert.h"

int	main(int argc, char **argv)
{
	FILE		*fin;
	FILE		*fd;
	FILE		*fid;
	unsigned int	amax_h;

	if (argc != 4)
	{
		fprintf(stderr, "usage: %s amax_in.txt d_out.txt id_out.txt\n",
			argv[0]);
		return (1);
	}
	fin = fopen(argv[1], "r");
	fd = fopen(argv[2], "w");
	fid = fopen(argv[3], "w");
	while (fscanf(fin, "%x", &amax_h) == 1)
	{
		float		amax;
		float		d;
		float		id;
		uint16_t	d_h;
		uint32_t	id_bits;

		amax = membrane_f16_to_f32((uint16_t)amax_h);
		d = amax / 127.0f;
		id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
		d_h = membrane_f32_to_f16(d);
		memcpy(&id_bits, &id, sizeof(id_bits));
		fprintf(fd, "%04x\n", d_h);
		fprintf(fid, "%08x\n", id_bits);
	}
	fclose(fin);
	fclose(fd);
	fclose(fid);
	return (0);
}
