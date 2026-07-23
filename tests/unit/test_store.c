#include <string.h>

#include "store_internal.h"
#include "test_helpers.h"

# define BLK 1024

static membrane_store_t	*make_store(size_t budget)
{
	membrane_store_config_t	cfg;

	cfg.budget_bytes = budget;
	cfg.default_codec = MEMBRANE_CODEC_RLE;
	cfg.index_capacity = 64;
	cfg.backend = NULL;
	return (membrane_store_create(&cfg));
}

static void	put_random(membrane_store_t *s, uint64_t id, uint8_t *buf,
				size_t n, uint32_t seed)
{
	test_fill_random(buf, n, seed);
	TEST_ASSERT(membrane_store_put(s, id, buf, n) == MEMBRANE_OK,
		"put of random data succeeds");
}

static int	present(membrane_store_t *s, uint64_t id)
{
	membrane_block_meta_t	meta;

	return (membrane_store_query(s, id, &meta) == MEMBRANE_OK);
}

static size_t	resident(membrane_store_t *s)
{
	membrane_store_stats_t	st;

	membrane_store_get_stats(s, &st);
	return (st.resident_bytes);
}

static void	test_put_get_roundtrip(void)
{
	membrane_store_t		*s;
	membrane_block_meta_t	meta;
	uint8_t					in[BLK];
	uint8_t					out[BLK];
	size_t					got;

	s = make_store(1u << 20);
	TEST_ASSERT(s != NULL, "store creates");
	put_random(s, 1, in, BLK, 5);
	TEST_ASSERT(membrane_store_get(s, 1, out, BLK, &got) == MEMBRANE_OK
		&& got == BLK && memcmp(in, out, BLK) == 0,
		"put/get round-trips bit-identically");
	TEST_ASSERT(membrane_store_query(s, 1, &meta) == MEMBRANE_OK
		&& meta.original_size == BLK && meta.stored_codec == MEMBRANE_CODEC_RAW,
		"query reports RAW storage for incompressible data");
	membrane_store_destroy(s);
}

static void	test_buffer_too_small(void)
{
	membrane_store_t	*s;
	uint8_t				in[BLK];
	uint8_t				out[16];
	size_t				got;

	s = make_store(1u << 20);
	put_random(s, 1, in, BLK, 6);
	TEST_ASSERT(membrane_store_get(s, 1, out, sizeof(out), &got)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL && got == BLK,
		"get reports required size when buffer too small");
	membrane_store_destroy(s);
}

static void	test_budget_never_exceeded(void)
{
	membrane_store_t		*s;
	membrane_store_stats_t	st;
	uint8_t					buf[BLK];
	uint64_t				id;

	s = make_store(4 * BLK);
	id = 0;
	while (id < 20)
	{
		put_random(s, id, buf, BLK, (uint32_t)(id + 1));
		TEST_ASSERT(resident(s) <= 4 * BLK, "resident never exceeds budget");
		id++;
	}
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(st.block_count <= 4 && st.evictions >= 16,
		"store keeps at most budget-many blocks and counts evictions");
	membrane_store_destroy(s);
}

static void	test_lru_evicts_correct(void)
{
	membrane_store_t	*s;
	uint8_t				buf[BLK];

	s = make_store(3 * BLK);
	put_random(s, 1, buf, BLK, 1);
	put_random(s, 2, buf, BLK, 2);
	put_random(s, 3, buf, BLK, 3);
	put_random(s, 4, buf, BLK, 4);
	TEST_ASSERT(!present(s, 1), "LRU block 1 was evicted");
	TEST_ASSERT(present(s, 2) && present(s, 3) && present(s, 4),
		"more recent blocks survive");
	membrane_store_destroy(s);
}

static void	test_get_updates_mru(void)
{
	membrane_store_t	*s;
	uint8_t				buf[BLK];
	size_t				got;

	s = make_store(3 * BLK);
	put_random(s, 1, buf, BLK, 1);
	put_random(s, 2, buf, BLK, 2);
	put_random(s, 3, buf, BLK, 3);
	TEST_ASSERT(membrane_store_get(s, 1, buf, BLK, &got) == MEMBRANE_OK,
		"touch block 1 to make it MRU");
	put_random(s, 4, buf, BLK, 4);
	TEST_ASSERT(!present(s, 2), "block 2 is now LRU and gets evicted");
	TEST_ASSERT(present(s, 1) && present(s, 3) && present(s, 4),
		"recently-read block 1 survives");
	membrane_store_destroy(s);
}

static void	test_pinned_not_evicted(void)
{
	membrane_store_t	*s;
	store_entry_t		*e;
	uint8_t				buf[BLK];

	s = make_store(2 * BLK);
	put_random(s, 1, buf, BLK, 1);
	put_random(s, 2, buf, BLK, 2);
	pthread_mutex_lock(&s->lock);
	e = store_index_find(s, 1);
	e->pin_count += 1;
	pthread_mutex_unlock(&s->lock);
	put_random(s, 3, buf, BLK, 3);
	TEST_ASSERT(present(s, 1) && !present(s, 2) && present(s, 3),
		"pinned block 1 survives; unpinned LRU block 2 is evicted");
	pthread_mutex_lock(&s->lock);
	e->pin_count -= 1;
	pthread_mutex_unlock(&s->lock);
	membrane_store_destroy(s);
}

/* Replicates the store's last-unpin free path (entry_free is static). */
static void	manual_unpin(membrane_store_t *s, store_entry_t *e)
{
	pthread_mutex_lock(&s->lock);
	e->pin_count -= 1;
	if (e->pin_count == 0 && e->pending_remove)
	{
		membrane_block_destroy(e->block);
		free(e);
	}
	pthread_mutex_unlock(&s->lock);
}

static void	test_remove_pinned_defers(void)
{
	membrane_store_t	*s;
	store_entry_t		*e;
	uint8_t				buf[BLK];

	s = make_store(1u << 20);
	put_random(s, 1, buf, BLK, 1);
	pthread_mutex_lock(&s->lock);
	e = store_index_find(s, 1);
	e->pin_count += 1;
	pthread_mutex_unlock(&s->lock);
	TEST_ASSERT(membrane_store_remove(s, 1) == MEMBRANE_OK,
		"remove of a pinned block succeeds");
	TEST_ASSERT(!present(s, 1) && e->pending_remove,
		"removed block is unfindable but its free is deferred");
	manual_unpin(s, e);
	membrane_store_destroy(s);
}

static void	test_overwrite_accounting(void)
{
	membrane_store_t		*s;
	membrane_store_stats_t	st;
	uint8_t					buf[2 * BLK];

	s = make_store(1u << 20);
	put_random(s, 5, buf, BLK, 5);
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(st.block_count == 1 && st.raw_blocks == 1
		&& st.resident_bytes == BLK, "first put accounted as one RAW block");
	memset(buf, 0, sizeof(buf));
	TEST_ASSERT(membrane_store_put(s, 5, buf, sizeof(buf)) == MEMBRANE_OK,
		"overwrite with compressible data");
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(st.block_count == 1 && st.raw_blocks == 0
		&& st.compressed_blocks == 1 && st.logical_bytes == 2 * BLK
		&& st.resident_bytes < BLK, "overwrite replaces accounting cleanly");
	membrane_store_destroy(s);
}

static void	test_remove_and_duplicate(void)
{
	membrane_store_t		*s;
	membrane_store_stats_t	st;
	uint8_t					buf[BLK];
	size_t					got;

	s = make_store(1u << 20);
	put_random(s, 7, buf, BLK, 70);
	TEST_ASSERT(membrane_store_remove(s, 7) == MEMBRANE_OK
		&& membrane_store_remove(s, 7) == MEMBRANE_ERR_NOT_FOUND,
		"remove is idempotent-safe: second remove reports NOT_FOUND");
	put_random(s, 7, buf, BLK, 71);
	memset(buf, 0xAB, BLK);
	TEST_ASSERT(membrane_store_put(s, 7, buf, BLK) == MEMBRANE_OK,
		"duplicate id overwrites");
	membrane_store_get(s, 7, buf, BLK, &got);
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(buf[0] == 0xAB && st.block_count == 1,
		"duplicate id keeps the latest value and a single entry");
	membrane_store_destroy(s);
}

static void	test_edge_budgets(void)
{
	membrane_store_t	*s;
	uint8_t				buf[BLK];
	size_t				got;

	s = make_store(0);
	test_fill_random(buf, BLK, 9);
	TEST_ASSERT(membrane_store_put(s, 1, buf, BLK) == MEMBRANE_ERR_BUDGET_FULL,
		"non-empty put fails under zero budget");
	TEST_ASSERT(membrane_store_put(s, 2, NULL, 0) == MEMBRANE_OK
		&& membrane_store_get(s, 2, NULL, 0, &got) == MEMBRANE_OK && got == 0,
		"empty block fits in a zero budget");
	membrane_store_destroy(s);
	s = make_store(BLK / 2);
	TEST_ASSERT(membrane_store_put(s, 1, buf, BLK) == MEMBRANE_ERR_BUDGET_FULL
		&& !present(s, 1), "a block larger than the whole budget is rejected");
	membrane_store_destroy(s);
}

static void	test_codec_selection(void)
{
	membrane_store_t		*s;
	membrane_store_stats_t	st;
	uint8_t					buf[BLK];

	s = make_store(1u << 20);
	test_fill_random(buf, BLK, 3);
	membrane_store_put(s, 1, buf, BLK);
	memset(buf, 0, BLK);
	membrane_store_put(s, 2, buf, BLK);
	membrane_store_get_stats(s, &st);
	TEST_ASSERT(st.raw_blocks == 1 && st.compressed_blocks == 1,
		"random data stays RAW, zero data compresses");
	TEST_ASSERT(st.effective_capacity_ratio > 1.0,
		"effective capacity exceeds 1x once a block compresses");
	membrane_store_destroy(s);
}

int	main(void)
{
	test_put_get_roundtrip();
	test_buffer_too_small();
	test_budget_never_exceeded();
	test_lru_evicts_correct();
	test_get_updates_mru();
	test_pinned_not_evicted();
	test_remove_pinned_defers();
	test_overwrite_accounting();
	test_remove_and_duplicate();
	test_edge_budgets();
	test_codec_selection();
	return (0);
}
