#ifndef MEMBRANE_BACKEND_INTERNAL_H
# define MEMBRANE_BACKEND_INTERNAL_H

# include "membrane/backend.h"

typedef struct s_membrane_backend_vtable
{
	membrane_status_t	(*store)(void *ctx, const membrane_block_t *block);
	membrane_status_t	(*load)(void *ctx, uint64_t id,
							membrane_block_t **out);
	membrane_status_t	(*remove)(void *ctx, uint64_t id);
	int					(*contains)(void *ctx, uint64_t id);
	uint64_t			(*used_bytes)(void *ctx);
	void				(*destroy)(void *ctx);
}	membrane_backend_vtable_t;

struct s_membrane_backend
{
	const membrane_backend_vtable_t	*vt;
	void							*ctx;
};

#endif
