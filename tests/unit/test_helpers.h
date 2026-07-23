#ifndef MEMBRANE_TEST_HELPERS_H
# define MEMBRANE_TEST_HELPERS_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>
# include <stdlib.h>

# define TEST_ASSERT(cond, msg) \
	do { \
		if (!(cond)) \
		{ \
			fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
			abort(); \
		} \
	} while (0)

/* Deterministic xorshift PRNG so failures reproduce across runs. */
static inline uint32_t	test_rand_next(uint32_t *state)
{
	uint32_t	x;

	x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return (x);
}

static inline void	test_fill_random(uint8_t *buf, size_t len, uint32_t seed)
{
	uint32_t	state;
	size_t		i;

	state = seed;
	if (state == 0)
		state = 1;
	i = 0;
	while (i < len)
	{
		buf[i] = (uint8_t)test_rand_next(&state);
		i++;
	}
}

#endif
