#define _DEFAULT_SOURCE

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "membrane/backend.h"
#include "store_internal.h"
#include "test_helpers.h"

# define BLK 1024

static char	g_dir[] = "/tmp/membrane-be-XXXXXX";

static membrane_store_t	*make_store(size_t budget, membrane_backend_t *be)
{
	membrane_store_config_t	cfg;

	cfg.budget_bytes = budget;
	cfg.default_codec = MEMBRANE_CODEC_RLE;
	cfg.index_capacity = 64;
	cfg.backend = be;
	return (membrane_store_create(&cfg));
}

static void	put_random(membrane_store_t *s, uint64_t id, uint32_t seed)
{
	uint8_t	buf[BLK];

	test_fill_random(buf, BLK, seed);
	TEST_ASSERT(membrane_store_put(s, id, buf, BLK) == MEMBRANE_OK,
		"put succeeds");
}

static void	check_get(membrane_store_t *s, uint64_t id, uint32_t seed)
{
	uint8_t	want[BLK];
	uint8_t	got[BLK];
	size_t	n;

	test_fill_random(want, BLK, seed);
	TEST_ASSERT(membrane_store_get(s, id, got, BLK, &n) == MEMBRANE_OK
		&& n == BLK && memcmp(want, got, BLK) == 0,
		"get round-trips through the backend");
}

static size_t	resident(membrane_store_t *s)
{
	membrane_store_stats_t	st;

	membrane_store_get_stats(s, &st);
	return (st.resident_bytes);
}

static void	test_evict_then_get(membrane_backend_t *be)
{
	membrane_store_t		*s;
	membrane_store_stats_t	st;

	s = make_store(2 * BLK, be);
	put_random(s, 1, 11);
	put_random(s, 2, 22);
	put_random(s, 3, 33);
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(st.evictions >= 1 && st.backend_writes >= 1,
		"putting past budget evicts to the backend");
	check_get(s, 1, 11);
	check_get(s, 2, 22);
	check_get(s, 3, 33);
	TEST_ASSERT(resident(s) <= 2 * BLK, "budget still respected after promotes");
	membrane_store_destroy(s);
}

static void	test_evict_promote_cycles(membrane_backend_t *be)
{
	membrane_store_t	*s;
	uint64_t			id;
	int					round;

	s = make_store(4 * BLK, be);
	id = 0;
	while (id < 16)
	{
		put_random(s, id, (uint32_t)(id + 100));
		id++;
	}
	round = 0;
	while (round < 3)
	{
		id = 0;
		while (id < 16)
		{
			check_get(s, id, (uint32_t)(id + 100));
			TEST_ASSERT(resident(s) <= 4 * BLK, "budget held every promote");
			id++;
		}
		round++;
	}
	membrane_store_destroy(s);
}

static store_entry_t	*find(membrane_store_t *s, uint64_t id)
{
	store_entry_t	*e;

	pthread_mutex_lock(&s->lock);
	e = store_index_find(s, id);
	pthread_mutex_unlock(&s->lock);
	return (e);
}

static void	corrupt_backend_file(uint64_t id, int truncate_it)
{
	char	path[512];
	FILE	*f;

	snprintf(path, sizeof(path), "%s/%016llx.mbk", g_dir,
		(unsigned long long)id);
	if (truncate_it)
	{
		TEST_ASSERT(truncate(path, 40) == 0, "truncate backend file");
		return ;
	}
	f = fopen(path, "r+b");
	TEST_ASSERT(f != NULL, "reopen backend file");
	fseek(f, 60, SEEK_SET);
	fputc(0xFF ^ fgetc(f), f);
	fclose(f);
}

static void	test_corrupt_and_truncated(membrane_backend_t *be)
{
	membrane_store_t	*s;
	uint8_t				out[BLK];
	size_t				n;

	s = make_store(BLK, be);
	put_random(s, 1, 7);
	put_random(s, 2, 8);		/* forces id 1 to the backend */
	TEST_ASSERT(find(s, 1)->state == ENTRY_EVICTED, "id 1 is evicted");
	corrupt_backend_file(1, 0);
	TEST_ASSERT(membrane_store_get(s, 1, out, BLK, &n) != MEMBRANE_OK,
		"corrupted backend block is rejected, not returned");
	put_random(s, 3, 9);		/* re-evict id 2 */
	corrupt_backend_file(2, 1);
	TEST_ASSERT(membrane_store_get(s, 2, out, BLK, &n) != MEMBRANE_OK,
		"truncated backend block is rejected");
	membrane_store_destroy(s);
}

static void	test_failed_write_keeps_resident(void)
{
	membrane_backend_t	*ro;
	membrane_store_t	*s;
	char				rodir[] = "/tmp/membrane-ro-XXXXXX";

	TEST_ASSERT(mkdtemp(rodir) != NULL, "temp dir");
	ro = membrane_backend_file_create(rodir);
	TEST_ASSERT(ro != NULL && chmod(rodir, 0500) == 0, "make backend dir read-only");
	s = make_store(BLK, ro);
	put_random(s, 1, 1);
	TEST_ASSERT(membrane_store_put(s, 2, (const uint8_t *)"x", 1) != MEMBRANE_OK
		|| find(s, 1) != NULL,
		"a failed eviction write keeps the original resident");
	check_get(s, 1, 1);
	membrane_store_destroy(s);
	membrane_backend_destroy(ro);
	chmod(rodir, 0700);
	rmdir(rodir);
}

static void	test_overwrite_evicted(membrane_backend_t *be)
{
	membrane_store_t	*s;
	uint8_t				out[BLK];
	size_t				n;

	s = make_store(BLK, be);
	put_random(s, 1, 5);
	put_random(s, 2, 6);					/* id 1 to backend */
	TEST_ASSERT(find(s, 1)->state == ENTRY_EVICTED, "id 1 evicted");
	memset(out, 0xCD, BLK);
	TEST_ASSERT(membrane_store_put(s, 1, out, BLK) == MEMBRANE_OK,
		"overwrite an evicted id");
	TEST_ASSERT(membrane_store_get(s, 1, out, BLK, &n) == MEMBRANE_OK
		&& out[0] == 0xCD, "overwritten evicted id returns new data");
	membrane_store_destroy(s);
}

static int	dir_file_count(void)
{
	DIR				*d;
	struct dirent	*e;
	int				count;

	d = opendir(g_dir);
	TEST_ASSERT(d != NULL, "open backend dir");
	count = 0;
	while ((e = readdir(d)) != NULL)
		if (strstr(e->d_name, ".mbk") != NULL)
			count++;
	closedir(d);
	return (count);
}

static void	test_remove_and_cleanup(void)
{
	membrane_backend_t	*be;
	membrane_store_t	*s;
	membrane_block_meta_t	meta;

	be = membrane_backend_file_create(g_dir);
	s = make_store(BLK, be);
	put_random(s, 1, 1);
	put_random(s, 2, 2);				/* id 1 to backend */
	TEST_ASSERT(dir_file_count() >= 1, "backend has a file");
	TEST_ASSERT(membrane_store_remove(s, 1) == MEMBRANE_OK
		&& membrane_store_query(s, 1, &meta) == MEMBRANE_ERR_NOT_FOUND,
		"remove clears an evicted block");
	membrane_store_destroy(s);
	membrane_backend_destroy(be);
	TEST_ASSERT(dir_file_count() == 0, "destroy leaves no backend files");
}

int	main(void)
{
	membrane_backend_t	*be;

	TEST_ASSERT(mkdtemp(g_dir) != NULL, "temp dir created");
	be = membrane_backend_file_create(g_dir);
	TEST_ASSERT(be != NULL, "file backend created");
	test_evict_then_get(be);
	test_evict_promote_cycles(be);
	test_corrupt_and_truncated(be);
	test_overwrite_evicted(be);
	membrane_backend_destroy(be);
	test_failed_write_keeps_resident();
	test_remove_and_cleanup();
	rmdir(g_dir);
	return (0);
}
