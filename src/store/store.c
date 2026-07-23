#include <stdlib.h>

#include "membrane/backend.h"
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

/* Accounts a newly resident block (fresh put). */
static void	acct_insert(struct s_membrane_store *s, const membrane_block_t *b)
{
	s->resident_bytes += b->stored_size;
	s->logical_bytes += b->original_size;
	s->block_count += 1;
	if (b->stored_codec == MEMBRANE_CODEC_RAW)
		s->raw_blocks += 1;
	else
		s->compressed_blocks += 1;
	if (s->resident_bytes > s->peak_resident_bytes)
		s->peak_resident_bytes = s->resident_bytes;
}

/* Drops a block's contribution to logical/count/codec tallies. */
static void	acct_forget(struct s_membrane_store *s, const membrane_block_t *b)
{
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

/*
 * Fully removes a stable (RESIDENT or EVICTED) entry from the store,
 * deleting its backend copy when evicted. Frees now, or defers to the
 * last unpin if a decoder still holds it.
 */
static void	entry_detach(struct s_membrane_store *s, store_entry_t *e)
{
	store_index_remove(s, e);
	if (e->state == ENTRY_RESIDENT)
	{
		lru_unlink(s, e);
		s->resident_bytes -= e->block->stored_size;
	}
	else
	{
		s->backend_bytes -= e->block->stored_size;
		if (s->backend != NULL)
			membrane_backend_remove(s->backend, e->id);
	}
	acct_forget(s, e->block);
	if (e->pin_count > 0)
		e->pending_remove = 1;
	else
		entry_free(e);
}

static int	store_fits(const struct s_membrane_store *s, size_t need)
{
	return (need <= s->budget_bytes - s->resident_bytes);
}

static int	evict_finish(struct s_membrane_store *s, store_entry_t *v,
				size_t size, membrane_status_t st)
{
	if (st != MEMBRANE_OK)
	{
		s->resident_bytes += size;
		v->state = ENTRY_RESIDENT;
		lru_push_front(s, v);
		s->backend_write_failures += 1;
		pthread_cond_broadcast(&s->cond);
		return (-1);
	}
	free(v->block->data);
	v->block->data = NULL;
	v->state = ENTRY_EVICTED;
	s->backend_bytes += size;
	s->backend_writes += 1;
	s->evictions += 1;
	s->eviction_bytes += size;
	pthread_cond_broadcast(&s->cond);
	return (1);
}

/*
 * Evicts the LRU resident block. With a backend the payload is written out
 * (lock released for the I/O) and the entry stays EVICTED; without one it
 * is dropped. Returns 1 on success, 0 if nothing is evictable, -1 if the
 * backend write failed (the block is kept resident).
 */
static int	evict_one(struct s_membrane_store *s)
{
	store_entry_t		*v;
	size_t				size;
	membrane_status_t	st;

	v = s->lru_tail;
	while (v != NULL && v->pin_count != 0)
		v = v->lru_prev;
	if (v == NULL)
		return (0);
	if (s->backend == NULL)
		return (entry_detach(s, v), s->evictions += 1, 1);
	size = v->block->stored_size;
	v->state = ENTRY_EVICTING;
	lru_unlink(s, v);
	s->resident_bytes -= size;
	pthread_mutex_unlock(&s->lock);
	st = membrane_backend_store(s->backend, v->block);
	pthread_mutex_lock(&s->lock);
	return (evict_finish(s, v, size, st));
}

static int	store_make_room(struct s_membrane_store *s, size_t need)
{
	int	r;

	while (!store_fits(s, need))
	{
		r = evict_one(s);
		if (r <= 0)
			return (0);
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
		return (*status = MEMBRANE_ERR_ALLOC_FAILED, (membrane_block_t *)NULL);
	*status = membrane_block_write(block, bytes, len);
	if (*status != MEMBRANE_OK)
		return (membrane_block_destroy(block), (membrane_block_t *)NULL);
	return (block);
}

/* Waits until any transient (EVICTING/LOADING) entry for `id` settles. */
static store_entry_t	*find_settled(struct s_membrane_store *s, uint64_t id)
{
	store_entry_t	*e;

	e = store_index_find(s, id);
	while (e != NULL
		&& (e->state == ENTRY_EVICTING || e->state == ENTRY_LOADING))
	{
		pthread_cond_wait(&s->cond, &s->lock);
		e = store_index_find(s, id);
	}
	return (e);
}

static membrane_status_t	store_insert(struct s_membrane_store *s,
								store_entry_t *entry)
{
	store_entry_t	*old;

	pthread_mutex_lock(&s->lock);
	s->puts += 1;
	old = find_settled(s, entry->id);
	if (old != NULL)
		entry_detach(s, old);
	if (!store_make_room(s, entry->block->stored_size))
	{
		pthread_mutex_unlock(&s->lock);
		return (entry_free(entry), MEMBRANE_ERR_BUDGET_FULL);
	}
	store_index_insert(s, entry);
	lru_push_front(s, entry);
	acct_insert(s, entry->block);
	pthread_cond_broadcast(&s->cond);
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
		return (membrane_block_destroy(block), MEMBRANE_ERR_ALLOC_FAILED);
	entry->id = id;
	entry->block = block;
	entry->state = ENTRY_RESIDENT;
	return (store_insert(store, entry));
}

/* Serves a resident entry: pin, drop the lock, decode, unpin. */
static membrane_status_t	get_resident(struct s_membrane_store *s,
								store_entry_t *e, uint8_t *out, size_t cap,
								size_t *out_len)
{
	membrane_status_t	status;

	if (cap < e->block->original_size)
	{
		*out_len = e->block->original_size;
		return (pthread_mutex_unlock(&s->lock), MEMBRANE_ERR_BUFFER_TOO_SMALL);
	}
	e->pin_count += 1;
	membrane_block_touch(e->block);
	lru_move_front(s, e);
	pthread_mutex_unlock(&s->lock);
	status = membrane_block_decode(e->block, out, cap, out_len);
	pthread_mutex_lock(&s->lock);
	e->pin_count -= 1;
	if (e->pin_count == 0 && e->pending_remove)
		entry_free(e);
	pthread_cond_broadcast(&s->cond);
	pthread_mutex_unlock(&s->lock);
	return (status);
}

/* Installs a loaded block as resident (room already reserved). */
static void	promote_install(struct s_membrane_store *s, store_entry_t *e,
				membrane_block_t *nb)
{
	size_t	size;

	size = nb->stored_size;
	membrane_backend_remove(s->backend, e->id);
	membrane_block_destroy(e->block);
	e->block = nb;
	e->state = ENTRY_RESIDENT;
	lru_push_front(s, e);
	s->backend_bytes -= size;
	s->resident_bytes += size;
	if (s->resident_bytes > s->peak_resident_bytes)
		s->peak_resident_bytes = s->resident_bytes;
	s->promotions += 1;
	s->promotion_bytes += size;
}

/* Finishes a promotion after the decode: promote if room, else stay cold. */
static membrane_status_t	promote_finish(struct s_membrane_store *s,
								store_entry_t *e, membrane_block_t *nb,
								membrane_status_t decode_st)
{
	s->backend_reads += 1;
	if (decode_st != MEMBRANE_OK)
	{
		e->state = ENTRY_EVICTED;
		s->backend_read_failures += 1;
		membrane_block_destroy(nb);
	}
	else if (store_make_room(s, nb->stored_size))
		promote_install(s, e, nb);
	else
	{
		e->state = ENTRY_EVICTED;
		membrane_block_destroy(nb);
	}
	pthread_cond_broadcast(&s->cond);
	pthread_mutex_unlock(&s->lock);
	return (decode_st);
}

/* Serves an evicted entry: load and decode outside the lock, then promote. */
static membrane_status_t	get_promote(struct s_membrane_store *s,
								store_entry_t *e, uint8_t *out, size_t cap,
								size_t *out_len)
{
	membrane_block_t	*nb;
	membrane_status_t	status;

	if (cap < e->block->original_size)
	{
		*out_len = e->block->original_size;
		return (pthread_mutex_unlock(&s->lock), MEMBRANE_ERR_BUFFER_TOO_SMALL);
	}
	e->state = ENTRY_LOADING;
	pthread_mutex_unlock(&s->lock);
	status = membrane_backend_load(s->backend, e->id, &nb);
	if (status != MEMBRANE_OK)
	{
		pthread_mutex_lock(&s->lock);
		e->state = ENTRY_EVICTED;
		s->backend_read_failures += 1;
		pthread_cond_broadcast(&s->cond);
		return (pthread_mutex_unlock(&s->lock), status);
	}
	status = membrane_block_decode(nb, out, cap, out_len);
	pthread_mutex_lock(&s->lock);
	return (promote_finish(s, e, nb, status));
}

membrane_status_t	membrane_store_get(membrane_store_t *store, uint64_t id,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	store_entry_t	*e;

	if (store == NULL || out_len == NULL || (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	pthread_mutex_lock(&store->lock);
	store->gets += 1;
	e = find_settled(store, id);
	if (e == NULL)
	{
		store->misses += 1;
		pthread_mutex_unlock(&store->lock);
		return (MEMBRANE_ERR_NOT_FOUND);
	}
	store->hits += 1;
	if (e->state == ENTRY_RESIDENT)
		return (get_resident(store, e, out, out_cap, out_len));
	return (get_promote(store, e, out, out_cap, out_len));
}

membrane_status_t	membrane_store_remove(membrane_store_t *store, uint64_t id)
{
	store_entry_t	*entry;

	if (store == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	pthread_mutex_lock(&store->lock);
	entry = find_settled(store, id);
	if (entry == NULL)
	{
		pthread_mutex_unlock(&store->lock);
		return (MEMBRANE_ERR_NOT_FOUND);
	}
	entry_detach(store, entry);
	pthread_cond_broadcast(&store->cond);
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
	m->resident = (e->state == ENTRY_RESIDENT);
}

membrane_status_t	membrane_store_query(membrane_store_t *store, uint64_t id,
						membrane_block_meta_t *out_meta)
{
	store_entry_t	*entry;

	if (store == NULL || out_meta == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	pthread_mutex_lock(&store->lock);
	entry = find_settled(store, id);
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
	uint64_t	footprint;

	footprint = (uint64_t)s->resident_bytes + s->backend_bytes;
	if (footprint == 0)
		return (0.0);
	return ((double)s->logical_bytes / (double)footprint);
}

static void	store_copy_counters(const struct s_membrane_store *s,
				membrane_store_stats_t *o)
{
	o->puts = s->puts;
	o->gets = s->gets;
	o->hits = s->hits;
	o->misses = s->misses;
	o->evictions = s->evictions;
	o->promotions = s->promotions;
	o->eviction_bytes = s->eviction_bytes;
	o->promotion_bytes = s->promotion_bytes;
	o->backend_writes = s->backend_writes;
	o->backend_reads = s->backend_reads;
	o->backend_write_failures = s->backend_write_failures;
	o->backend_read_failures = s->backend_read_failures;
	o->raw_blocks = s->raw_blocks;
	o->compressed_blocks = s->compressed_blocks;
}

static void	store_copy_stats(const struct s_membrane_store *s,
				membrane_store_stats_t *o)
{
	o->budget_bytes = s->budget_bytes;
	o->resident_bytes = s->resident_bytes;
	o->peak_resident_bytes = s->peak_resident_bytes;
	o->logical_bytes = s->logical_bytes;
	o->backend_bytes = s->backend_bytes;
	o->stored_bytes = (uint64_t)s->resident_bytes + s->backend_bytes;
	o->block_count = s->block_count;
	o->effective_capacity_ratio = store_ratio(s);
	store_copy_counters(s, o);
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

static int	store_init_sync(struct s_membrane_store *s)
{
	if (pthread_mutex_init(&s->lock, NULL) != 0)
		return (-1);
	if (pthread_cond_init(&s->cond, NULL) != 0)
	{
		pthread_mutex_destroy(&s->lock);
		return (-1);
	}
	return (0);
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
	if (s->buckets == NULL || store_init_sync(s) != 0)
		return (free(s->buckets), free(s), (membrane_store_t *)NULL);
	s->bucket_count = buckets;
	s->budget_bytes = cfg->budget_bytes;
	s->default_codec = cfg->default_codec;
	s->backend = cfg->backend;
	return (s);
}

/* Frees every entry, resident or evicted, across all hash buckets. */
static void	store_free_entries(struct s_membrane_store *s)
{
	store_entry_t	*e;
	store_entry_t	*next;
	size_t			i;

	i = 0;
	while (i < s->bucket_count)
	{
		e = s->buckets[i];
		while (e != NULL)
		{
			next = e->hash_next;
			entry_free(e);
			e = next;
		}
		i++;
	}
}

void	membrane_store_destroy(membrane_store_t *store)
{
	if (store == NULL)
		return ;
	store_free_entries(store);
	pthread_cond_destroy(&store->cond);
	pthread_mutex_destroy(&store->lock);
	free(store->buckets);
	free(store);
}
