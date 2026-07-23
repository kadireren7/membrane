/* rand_r is POSIX; expose it under strict -std=c11. */
#define _DEFAULT_SOURCE

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "store_internal.h"
#include "test_helpers.h"

# define BLK 512
# define NBLOCKS 32
# define NTHREADS 4
# define NITERS 4000

/* Block `id` holds BLK copies of the byte (uint8_t)id, so any successful
 * get can be verified without coordinating with concurrent writers. */
typedef struct s_worker
{
	membrane_store_t	*store;
	unsigned int		seed;
	int					mixed;
	int					failures;
}	worker_t;

static void	seed_store(membrane_store_t *s)
{
	uint8_t		buf[BLK];
	uint64_t	id;

	id = 0;
	while (id < NBLOCKS)
	{
		memset(buf, (int)(uint8_t)id, BLK);
		TEST_ASSERT(membrane_store_put(s, id, buf, BLK) == MEMBRANE_OK,
			"seed put");
		id++;
	}
}

static int	verify_block(uint64_t id, const uint8_t *buf, size_t n)
{
	size_t	i;

	if (n != BLK)
		return (0);
	i = 0;
	while (i < n)
	{
		if (buf[i] != (uint8_t)id)
			return (0);
		i++;
	}
	return (1);
}

static void	worker_get(worker_t *w, uint64_t id)
{
	uint8_t				out[BLK];
	size_t				got;
	membrane_status_t	st;

	st = membrane_store_get(w->store, id, out, sizeof(out), &got);
	if (st == MEMBRANE_OK)
	{
		if (!verify_block(id, out, got))
			w->failures += 1;
	}
	else if (!(w->mixed && st == MEMBRANE_ERR_NOT_FOUND))
		w->failures += 1;
}

static void	worker_mutate(worker_t *w, uint64_t id)
{
	uint8_t	buf[BLK];

	memset(buf, (int)(uint8_t)id, BLK);
	if (rand_r(&w->seed) & 1)
		membrane_store_put(w->store, id, buf, BLK);
	else
		membrane_store_remove(w->store, id);
}

static void	*worker_main(void *arg)
{
	worker_t	*w;
	int			i;
	uint64_t	id;

	w = arg;
	i = 0;
	while (i < NITERS)
	{
		id = (uint64_t)(rand_r(&w->seed) % NBLOCKS);
		if (w->mixed && (rand_r(&w->seed) % 4) == 0)
			worker_mutate(w, id);
		else
			worker_get(w, id);
		i++;
	}
	return (NULL);
}

static int	run_phase(membrane_store_t *s, int mixed)
{
	pthread_t	threads[NTHREADS];
	worker_t	workers[NTHREADS];
	int			i;
	int			failures;

	i = 0;
	while (i < NTHREADS)
	{
		workers[i].store = s;
		workers[i].seed = (unsigned int)(i + 1) * 2654435761u;
		workers[i].mixed = mixed;
		workers[i].failures = 0;
		pthread_create(&threads[i], NULL, worker_main, &workers[i]);
		i++;
	}
	failures = 0;
	i = 0;
	while (i < NTHREADS)
	{
		pthread_join(threads[i], NULL);
		failures += workers[i].failures;
		i++;
	}
	return (failures);
}

int	main(void)
{
	membrane_store_config_t	cfg;
	membrane_store_t		*s;

	cfg.budget_bytes = NBLOCKS * BLK;
	cfg.default_codec = MEMBRANE_CODEC_RLE;
	cfg.index_capacity = 64;
	s = membrane_store_create(&cfg);
	TEST_ASSERT(s != NULL, "store creates");
	seed_store(s);
	TEST_ASSERT(run_phase(s, 0) == 0, "concurrent gets all verify");
	TEST_ASSERT(run_phase(s, 1) == 0,
		"concurrent get/put/remove mix stays consistent");
	membrane_store_destroy(s);
	return (0);
}
