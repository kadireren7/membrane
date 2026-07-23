#include <stdlib.h>

#include "store_internal.h"

# define STORE_DEFAULT_BUCKETS 1024

static void	lru_unlink(struct s_membrane_store *s, store_entry_t *e)
{
	if (e->lru_prev != NULL)
		e->lru_prev->lru_next = e->lru_next;
	else
		s->lru_head = e->lru_next;
	if (e->lru_next != NULL)
		e->lru_next->lru_prev = e->lru_prev;
	else
		s->lru_tail = e->lru_prev;
	e->lru_prev = NULL;
	e->lru_next = NULL;
}

static void	lru_push_front(struct s_membrane_store *s, store_entry_t *e)
{
	e->lru_prev = NULL;
	e->lru_next = s->lru_head;
	if (s->lru_head != NULL)
		s->lru_head->lru_prev = e;
	else
		s->lru_tail = e;
	s->lru_head = e;
}

static void	lru_move_front(struct s_membrane_store *s, store_entry_t *e)
{
	if (s->lru_head == e)
		return ;
	lru_unlink(s, e);
	lru_push_front(s, e);
}

static void	account_add(struct s_membrane_store *s, const membrane_block_t *b)
{
	s->resident_bytes += b->stored_size;
	s->stored_bytes += b->stored_size;
	s->logical_bytes += b->original_size;
	s->block_count += 1;
	if (b->stored_codec == MEMBRANE_CODEC_RAW)
		s->raw_blocks += 1;
	else
		s->compressed_blocks += 1;
	if (s->resident_bytes > s->peak_resident_bytes)
		s->peak_resident_bytes = s->resident_bytes;
}

static void	account_sub(struct s_membrane_store *s, const membrane_block_t *b)
{
	s->resident_bytes -= b->stored_size;
	s->stored_bytes -= b->stored_size;
	s->logical_bytes -= b->original_size;
	s->block_count -= 1;
	if (b->stored_codec == MEMBRANE_CODEC_RAW)
		s->raw_blocks -= 1;
	else
		s->compressed_blocks -= 1;
}

static void	entry_free(store_entry_t *e)
{
	membrane_block_destroy(e->block);
	free(e);
}

/* Logically removes `e`; frees now, or defers to the last unpin. */
static void	entry_detach(struct s_membrane_store *s, store_entry_t *e)
{
	store_index_remove(s, e);
	lru_unlink(s, e);
	account_sub(s, e->block);
	if (e->pin_count > 0)
		e->pending_remove = 1;
	else
		entry_free(e);
}

static store_entry_t	*evictable_lru(struct s_membrane_store *s,
							const store_entry_t *protect)
{
	store_entry_t	*e;

	e = s->lru_tail;
	while (e != NULL)
	{
		if (e->pin_count == 0 && e != protect)
			return (e);
		e = e->lru_prev;
	}
	return (NULL);
}

/* Overflow-safe: all terms are <= budget, no addition is performed. */
static int	store_fits(const struct s_membrane_store *s, size_t need,
				size_t old_size)
{
	return (need <= s->budget_bytes - (s->resident_bytes - old_size));
}

static int	store_make_room(struct s_membrane_store *s, size_t need,
				size_t old_size, const store_entry_t *protect)
{
	store_entry_t	*victim;

	while (!store_fits(s, need, old_size))
	{
		victim = evictable_lru(s, protect);
		if (victim == NULL)
			return (0);
		entry_detach(s, victim);
		s->evictions += 1;
	}
	return (1);
}

static membrane_block_t	*store_compress_block(struct s_membrane_store *s,
							uint64_t id, const uint8_t *bytes, size_t len,
							membrane_status_t *status)
{
	membrane_block_t	*block;

	block = membrane_block_create(id, s->default_codec);
	if (block == NULL)
	{
		*status = MEMBRANE_ERR_ALLOC_FAILED;
		return (NULL);
	}
	*status = membrane_block_write(block, bytes, len);
	if (*status != MEMBRANE_OK)
	{
		membrane_block_destroy(block);
		return (NULL);
	}
	return (block);
}

static membrane_status_t	store_insert(struct s_membrane_store *s,
								store_entry_t *entry)
{
	store_entry_t	*old;
	size_t			old_size;

	pthread_mutex_lock(&s->lock);
	s->puts += 1;
	old = store_index_find(s, entry->id);
	old_size = 0;
	if (old != NULL)
		old_size = old->block->stored_size;
	if (!store_make_room(s, entry->block->stored_size, old_size, old))
	{
		pthread_mutex_unlock(&s->lock);
		entry_free(entry);
		return (MEMBRANE_ERR_BUDGET_FULL);
	}
	if (old != NULL)
		entry_detach(s, old);
	store_index_insert(s, entry);
	lru_push_front(s, entry);
	account_add(s, entry->block);
	pthread_mutex_unlock(&s->lock);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_store_put(membrane_store_t *store, uint64_t id,
						const uint8_t *bytes, size_t len)
{
	membrane_block_t	*block;
	store_entry_t		*entry;
	membrane_status_t	status;

	if (store == NULL || (bytes == NULL && len > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	block = store_compress_block(store, id, bytes, len, &status);
	if (block == NULL)
		return (status);
	entry = calloc(1, sizeof(*entry));
	if (entry == NULL)
	{
		membrane_block_destroy(block);
		return (MEMBRANE_ERR_ALLOC_FAILED);
	}
	entry->id = id;
	entry->block = block;
	return (store_insert(store, entry));
}

static store_entry_t	*store_get_pin(struct s_membrane_store *s, uint64_t id,
							size_t out_cap, size_t *out_len,
							membrane_status_t *status)
{
	store_entry_t	*entry;

	pthread_mutex_lock(&s->lock);
	s->gets += 1;
	entry = store_index_find(s, id);
	if (entry == NULL)
	{
		s->misses += 1;
		*status = MEMBRANE_ERR_NOT_FOUND;
		return (pthread_mutex_unlock(&s->lock), (store_entry_t *)NULL);
	}
	s->hits += 1;
	if (out_cap < entry->block->original_size)
	{
		*out_len = entry->block->original_size;
		*status = MEMBRANE_ERR_BUFFER_TOO_SMALL;
		return (pthread_mutex_unlock(&s->lock), (store_entry_t *)NULL);
	}
	entry->pin_count += 1;
	membrane_block_touch(entry->block);
	lru_move_front(s, entry);
	pthread_mutex_unlock(&s->lock);
	return (entry);
}

static void	store_get_unpin(struct s_membrane_store *s, store_entry_t *entry)
{
	pthread_mutex_lock(&s->lock);
	entry->pin_count -= 1;
	if (entry->pin_count == 0 && entry->pending_remove)
		entry_free(entry);
	pthread_mutex_unlock(&s->lock);
}

membrane_status_t	membrane_store_get(membrane_store_t *store, uint64_t id,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	store_entry_t		*entry;
	membrane_status_t	status;

	if (store == NULL || out_len == NULL || (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	entry = store_get_pin(store, id, out_cap, out_len, &status);
	if (entry == NULL)
		return (status);
	status = membrane_block_decode(entry->block, out, out_cap, out_len);
	store_get_unpin(store, entry);
	return (status);
}

membrane_status_t	membrane_store_remove(membrane_store_t *store, uint64_t id)
{
	store_entry_t	*entry;

	if (store == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	pthread_mutex_lock(&store->lock);
	entry = store_index_find(store, id);
	if (entry == NULL)
	{
		pthread_mutex_unlock(&store->lock);
		return (MEMBRANE_ERR_NOT_FOUND);
	}
	entry_detach(store, entry);
	pthread_mutex_unlock(&store->lock);
	return (MEMBRANE_OK);
}

static void	store_fill_meta(membrane_block_meta_t *m, const store_entry_t *e)
{
	m->id = e->id;
	m->original_size = e->block->original_size;
	m->stored_size = e->block->stored_size;
	m->access_count = e->block->access_count;
	m->last_access_ns = e->block->last_access_ns;
	m->stored_codec = e->block->stored_codec;
	m->resident = 1;
}

membrane_status_t	membrane_store_query(membrane_store_t *store, uint64_t id,
						membrane_block_meta_t *out_meta)
{
	store_entry_t	*entry;

	if (store == NULL || out_meta == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	pthread_mutex_lock(&store->lock);
	entry = store_index_find(store, id);
	if (entry == NULL)
	{
		pthread_mutex_unlock(&store->lock);
		return (MEMBRANE_ERR_NOT_FOUND);
	}
	store_fill_meta(out_meta, entry);
	pthread_mutex_unlock(&store->lock);
	return (MEMBRANE_OK);
}

static double	store_ratio(const struct s_membrane_store *s)
{
	if (s->resident_bytes == 0)
		return (0.0);
	return ((double)s->logical_bytes / (double)s->resident_bytes);
}

static void	store_copy_stats(const struct s_membrane_store *s,
				membrane_store_stats_t *o)
{
	o->budget_bytes = s->budget_bytes;
	o->resident_bytes = s->resident_bytes;
	o->peak_resident_bytes = s->peak_resident_bytes;
	o->logical_bytes = s->logical_bytes;
	o->stored_bytes = s->stored_bytes;
	o->block_count = s->block_count;
	o->puts = s->puts;
	o->gets = s->gets;
	o->hits = s->hits;
	o->misses = s->misses;
	o->evictions = s->evictions;
	o->raw_blocks = s->raw_blocks;
	o->compressed_blocks = s->compressed_blocks;
	o->effective_capacity_ratio = store_ratio(s);
}

void	membrane_store_get_stats(membrane_store_t *store,
			membrane_store_stats_t *out)
{
	if (store == NULL || out == NULL)
		return ;
	pthread_mutex_lock(&store->lock);
	store_copy_stats(store, out);
	pthread_mutex_unlock(&store->lock);
}

membrane_store_t	*membrane_store_create(const membrane_store_config_t *cfg)
{
	struct s_membrane_store	*s;
	size_t					buckets;

	if (cfg == NULL || membrane_codec_get(cfg->default_codec) == NULL)
		return (NULL);
	s = calloc(1, sizeof(*s));
	if (s == NULL)
		return (NULL);
	buckets = cfg->index_capacity;
	if (buckets == 0)
		buckets = STORE_DEFAULT_BUCKETS;
	s->buckets = calloc(buckets, sizeof(*s->buckets));
	if (s->buckets == NULL || pthread_mutex_init(&s->lock, NULL) != 0)
		return (free(s->buckets), free(s), (membrane_store_t *)NULL);
	s->bucket_count = buckets;
	s->budget_bytes = cfg->budget_bytes;
	s->default_codec = cfg->default_codec;
	return (s);
}

void	membrane_store_destroy(membrane_store_t *store)
{
	store_entry_t	*entry;
	store_entry_t	*next;

	if (store == NULL)
		return ;
	entry = store->lru_head;
	while (entry != NULL)
	{
		next = entry->lru_next;
		entry_free(entry);
		entry = next;
	}
	pthread_mutex_destroy(&store->lock);
	free(store->buckets);
	free(store);
}
