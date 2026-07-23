/* rand_r/mkdtemp are POSIX; expose them under strict -std=c11. */
#define _DEFAULT_SOURCE

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/backend.h"
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

static membrane_store_t	*make_store(size_t budget, membrane_backend_t *be)
{
	membrane_store_config_t	cfg;

	cfg.budget_bytes = budget;
	cfg.default_codec = MEMBRANE_CODEC_RLE;
	cfg.index_capacity = 64;
	cfg.backend = be;
	return (membrane_store_create(&cfg));
}

/* Tiny budget + backend: constant evict/promote churn across threads,
 * exercising concurrent promotion of the same evicted block and get/remove
 * races. Content is all-one-byte so any successful get is verifiable. */
static void	run_backend_phase(void)
{
	char				dir[] = "/tmp/membrane-cc-XXXXXX";
	membrane_backend_t	*be;
	membrane_store_t	*s;

	TEST_ASSERT(mkdtemp(dir) != NULL, "temp dir");
	be = membrane_backend_file_create(dir);
	TEST_ASSERT(be != NULL, "file backend");
	s = make_store(64, be);
	TEST_ASSERT(s != NULL, "backed store creates");
	seed_store(s);
	TEST_ASSERT(run_phase(s, 0) == 0, "concurrent promotes all verify");
	TEST_ASSERT(run_phase(s, 1) == 0, "concurrent promote/remove stays sane");
	membrane_store_destroy(s);
	membrane_backend_destroy(be);
	rmdir(dir);
}

int	main(void)
{
	membrane_store_t	*s;

	s = make_store(NBLOCKS * BLK, NULL);
	TEST_ASSERT(s != NULL, "store creates");
	seed_store(s);
	TEST_ASSERT(run_phase(s, 0) == 0, "concurrent gets all verify");
	TEST_ASSERT(run_phase(s, 1) == 0,
		"concurrent get/put/remove mix stays consistent");
	membrane_store_destroy(s);
	run_backend_phase();
	return (0);
}
