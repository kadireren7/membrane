#ifndef MEMBRANE_STORE_INTERNAL_H
# define MEMBRANE_STORE_INTERNAL_H

# include <pthread.h>

# include "membrane/store.h"

/*
 * A store record. It owns exactly one membrane_block_t (the compressed
 * payload plus metadata). pin_count > 0 marks a block whose data is
 * being decoded outside the store lock: it must not be freed. When a
 * pinned block is removed or overwritten, pending_remove defers the
 * free until the last unpin.
 */
typedef struct s_store_entry
{
	uint64_t				id;
	membrane_block_t		*block;
	uint32_t				pin_count;
	int						pending_remove;
	struct s_store_entry	*lru_prev;
	struct s_store_entry	*lru_next;
	struct s_store_entry	*hash_next;
}	store_entry_t;

struct s_membrane_store
{
	pthread_mutex_t		lock;
	store_entry_t		**buckets;
	size_t				bucket_count;
	store_entry_t		*lru_head;	/* most-recently-used */
	store_entry_t		*lru_tail;	/* least-recently-used */
	size_t				budget_bytes;
	membrane_codec_t	default_codec;
	size_t				resident_bytes;
	size_t				peak_resident_bytes;
	uint64_t			logical_bytes;
	uint64_t			stored_bytes;
	uint64_t			block_count;
	uint64_t			puts;
	uint64_t			gets;
	uint64_t			hits;
	uint64_t			misses;
	uint64_t			evictions;
	uint64_t			raw_blocks;
	uint64_t			compressed_blocks;
};

/* store_index.c — all callers hold store->lock. */
size_t			store_bucket_of(const struct s_membrane_store *store,
					uint64_t id);
store_entry_t	*store_index_find(struct s_membrane_store *store, uint64_t id);
void			store_index_insert(struct s_membrane_store *store,
					store_entry_t *entry);
void			store_index_remove(struct s_membrane_store *store,
					store_entry_t *entry);

#endif
