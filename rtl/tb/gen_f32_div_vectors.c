#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint32_t	rng_next(uint32_t *s)
{
	*s = *s * 1103515245u + 12345u;
	return (*s);
}

int	main(void)
{
	uint32_t	state;
	long		i;
	float		amax;
	float		d;
	float		id;
	uint32_t	amax_bits;
	uint32_t	d_bits;
	uint32_t	id_bits;

	state = 0x1234ABCDu;
	i = 0;
	while (i < 100000)
	{
		uint32_t	r = rng_next(&state);
		float		mant = (float)((r >> 8) & 0xFFFFFF) / (float)0x1000000;

		amax = mant * 131072.0f;
		memcpy(&amax_bits, &amax, sizeof(amax_bits));
		d = amax / 127.0f;
		id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
		memcpy(&d_bits, &d, sizeof(d_bits));
		memcpy(&id_bits, &id, sizeof(id_bits));
		printf("%08x %08x %08x\n", amax_bits, d_bits, id_bits);
		i++;
	}
	return (0);
}
