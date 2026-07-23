#ifndef MEMBRANE_STORE_INTERNAL_H
# define MEMBRANE_STORE_INTERNAL_H

# include <pthread.h>

# include "membrane/store.h"

/*
 * Per-entry lifecycle. RESIDENT and EVICTED are stable; EVICTING and
 * LOADING are transient states held by the one thread performing backend
 * I/O for that entry with the store lock released. Any other thread that
 * finds an entry in a transient state waits on the store condvar and
 * retries, so per-entry I/O is effectively serialized without holding the
 * lock across it.
 */
typedef enum e_entry_state
{
	ENTRY_RESIDENT = 0,
	ENTRY_EVICTING,
	ENTRY_EVICTED,
	ENTRY_LOADING
}	entry_state_t;

/*
 * A store record owning one membrane_block_t. When RESIDENT, block->data
 * holds the payload and the entry is in the LRU list. When EVICTED, the
 * payload lives in the backend, block->data is NULL, and the entry is out
 * of the LRU list (retaining its metadata for query/accounting).
 * pin_count > 0 (only possible while RESIDENT) marks a block being decoded
 * outside the lock; pending_remove defers its free to the last unpin.
 */
typedef struct s_store_entry
{
	uint64_t				id;
	membrane_block_t		*block;
	uint32_t				pin_count;
	int						pending_remove;
	entry_state_t			state;
	struct s_store_entry	*lru_prev;
	struct s_store_entry	*lru_next;
	struct s_store_entry	*hash_next;
}	store_entry_t;

struct s_membrane_store
{
	pthread_mutex_t		lock;
	pthread_cond_t		cond;
	membrane_backend_t	*backend;
	store_entry_t		**buckets;
	size_t				bucket_count;
	store_entry_t		*lru_head;	/* most-recently-used */
	store_entry_t		*lru_tail;	/* least-recently-used */
	size_t				budget_bytes;
	membrane_codec_t	default_codec;
	size_t				resident_bytes;
	size_t				peak_resident_bytes;
	uint64_t			backend_bytes;
	uint64_t			logical_bytes;
	uint64_t			block_count;
	uint64_t			puts;
	uint64_t			gets;
	uint64_t			hits;
	uint64_t			misses;
	uint64_t			evictions;
	uint64_t			promotions;
	uint64_t			eviction_bytes;
	uint64_t			promotion_bytes;
	uint64_t			backend_writes;
	uint64_t			backend_reads;
	uint64_t			backend_write_failures;
	uint64_t			backend_read_failures;
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
