#ifndef MEMBRANE_BACKEND_H
# define MEMBRANE_BACKEND_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/block.h"
# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * A cold-tier backend for evicted blocks. Opaque handle over an internal
 * vtable so future backends (RAM, CXL, FPGA) plug in without touching the
 * store. All state lives in the handle; there is no global state.
 */
typedef struct s_membrane_backend	membrane_backend_t;

/*
 * File backend: one file per block under `dir`, which must already exist
 * and be writable. Writes are atomic (temp file + rename). Session-scoped:
 * it does not index pre-existing files on open. Returns NULL on error.
 */
membrane_backend_t	*membrane_backend_file_create(const char *dir);

/* Persists `block` (metadata + payload) keyed by block->id. */
membrane_status_t	membrane_backend_store(membrane_backend_t *be,
						const membrane_block_t *block);

/*
 * Loads the block stored under `id` into a freshly allocated block that the
 * caller owns. Returns MEMBRANE_ERR_NOT_FOUND if absent, or
 * MEMBRANE_ERR_CORRUPT_DATA for a corrupted/truncated record.
 */
membrane_status_t	membrane_backend_load(membrane_backend_t *be,
						uint64_t id, membrane_block_t **out);

/* Deletes the record for `id`. A missing record is not an error. */
membrane_status_t	membrane_backend_remove(membrane_backend_t *be,
						uint64_t id);

int					membrane_backend_contains(membrane_backend_t *be,
						uint64_t id);
uint64_t			membrane_backend_used_bytes(membrane_backend_t *be);

/* Frees the handle and removes every record and temp file it created. */
void				membrane_backend_destroy(membrane_backend_t *be);

# ifdef __cplusplus
}
# endif

#endif
