#ifndef MEMBRANE_STORE_H
# define MEMBRANE_STORE_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/block.h"
# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

typedef struct s_membrane_store	membrane_store_t;

/*
 * budget_bytes:    hard cap on resident (compressed) bytes; never exceeded.
 * default_codec:   codec attempted for every put (RAW fallback still applies).
 * index_capacity:  hash bucket count hint; 0 selects a built-in default.
 */
typedef struct s_membrane_store_config
{
	size_t				budget_bytes;
	membrane_codec_t	default_codec;
	size_t				index_capacity;
}	membrane_store_config_t;

typedef struct s_membrane_store_stats
{
	size_t		budget_bytes;
	size_t		resident_bytes;
	size_t		peak_resident_bytes;
	uint64_t	logical_bytes;
	uint64_t	stored_bytes;
	uint64_t	block_count;
	uint64_t	puts;
	uint64_t	gets;
	uint64_t	hits;
	uint64_t	misses;
	uint64_t	evictions;
	uint64_t	raw_blocks;
	uint64_t	compressed_blocks;
	double		effective_capacity_ratio;
}	membrane_store_stats_t;

/* Metadata snapshot returned by membrane_store_query without decoding. */
typedef struct s_membrane_block_meta
{
	uint64_t			id;
	size_t				original_size;
	size_t				stored_size;
	uint64_t			access_count;
	uint64_t			last_access_ns;
	membrane_codec_t	stored_codec;
	int					resident;
}	membrane_block_meta_t;

/* Returns NULL on bad config or allocation failure. */
membrane_store_t	*membrane_store_create(const membrane_store_config_t *cfg);

/*
 * Frees the store and every block it holds. The caller must ensure no
 * other thread is using the store (no in-flight get/put/remove).
 */
void				membrane_store_destroy(membrane_store_t *store);

/*
 * Stores `len` bytes under `id`, compressing with the default codec.
 * Overwrites any existing block for `id`. Evicts least-recently-used
 * blocks to stay within budget; returns MEMBRANE_ERR_BUDGET_FULL if
 * room cannot be made (existing data for `id` is then left intact).
 */
membrane_status_t	membrane_store_put(membrane_store_t *store, uint64_t id,
						const uint8_t *bytes, size_t len);

/*
 * Decodes the block for `id` into `out` (out_cap bytes) and marks it
 * most-recently-used. Returns MEMBRANE_ERR_NOT_FOUND if absent, or
 * MEMBRANE_ERR_BUFFER_TOO_SMALL with *out_len set to the required size.
 */
membrane_status_t	membrane_store_get(membrane_store_t *store, uint64_t id,
						uint8_t *out, size_t out_cap, size_t *out_len);

/* Removes the block for `id`. Returns MEMBRANE_ERR_NOT_FOUND if absent. */
membrane_status_t	membrane_store_remove(membrane_store_t *store, uint64_t id);

/* Fills *out_meta without decoding. Does not affect LRU order. */
membrane_status_t	membrane_store_query(membrane_store_t *store, uint64_t id,
						membrane_block_meta_t *out_meta);

/* Snapshots current counters into *out. */
void				membrane_store_get_stats(membrane_store_t *store,
						membrane_store_stats_t *out);

# ifdef __cplusplus
}
# endif

#endif
