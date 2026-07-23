#include <stdlib.h>

#include "backend_internal.h"

membrane_status_t	membrane_backend_store(membrane_backend_t *be,
						const membrane_block_t *block)
{
	if (be == NULL || block == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	return (be->vt->store(be->ctx, block));
}

membrane_status_t	membrane_backend_load(membrane_backend_t *be,
						uint64_t id, membrane_block_t **out)
{
	if (be == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	return (be->vt->load(be->ctx, id, out));
}

membrane_status_t	membrane_backend_remove(membrane_backend_t *be, uint64_t id)
{
	if (be == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	return (be->vt->remove(be->ctx, id));
}

int	membrane_backend_contains(membrane_backend_t *be, uint64_t id)
{
	if (be == NULL)
		return (0);
	return (be->vt->contains(be->ctx, id));
}

uint64_t	membrane_backend_used_bytes(membrane_backend_t *be)
{
	if (be == NULL)
		return (0);
	return (be->vt->used_bytes(be->ctx));
}

void	membrane_backend_destroy(membrane_backend_t *be)
{
	if (be == NULL)
		return ;
	be->vt->destroy(be->ctx);
	free(be);
}
