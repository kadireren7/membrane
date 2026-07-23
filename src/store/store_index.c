#include "store_internal.h"

size_t	store_bucket_of(const struct s_membrane_store *store, uint64_t id)
{
	uint64_t	h;

	h = id * 1099511628211ull;	/* FNV prime, spreads sequential ids */
	h ^= h >> 29;
	return ((size_t)(h % store->bucket_count));
}

store_entry_t	*store_index_find(struct s_membrane_store *store, uint64_t id)
{
	store_entry_t	*entry;

	entry = store->buckets[store_bucket_of(store, id)];
	while (entry != NULL)
	{
		if (entry->id == id)
			return (entry);
		entry = entry->hash_next;
	}
	return (NULL);
}

void	store_index_insert(struct s_membrane_store *store, store_entry_t *entry)
{
	size_t	bucket;

	bucket = store_bucket_of(store, entry->id);
	entry->hash_next = store->buckets[bucket];
	store->buckets[bucket] = entry;
}

void	store_index_remove(struct s_membrane_store *store, store_entry_t *entry)
{
	store_entry_t	**pp;

	pp = &store->buckets[store_bucket_of(store, entry->id)];
	while (*pp != NULL)
	{
		if (*pp == entry)
		{
			*pp = entry->hash_next;
			entry->hash_next = NULL;
			return ;
		}
		pp = &(*pp)->hash_next;
	}
}
