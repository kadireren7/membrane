#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "membrane/f16convert.h"

int	main(void)
{
	unsigned int	h;
	uint32_t	bits;
	float		f;

	h = 0;
	while (h < 65536)
	{
		f = membrane_f16_to_f32((uint16_t)h);
		memcpy(&bits, &f, sizeof(bits));
		printf("%04x %08x\n", h, bits);
		h++;
	}
	return (0);
}
