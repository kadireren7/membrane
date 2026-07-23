#include <string.h>

#include "membrane/codec.h"

extern const membrane_codec_vtable_t	g_membrane_codec_raw;
extern const membrane_codec_vtable_t	g_membrane_codec_rle;

static membrane_status_t	unimpl_compress(const uint8_t *in, size_t in_len,
				uint8_t *out, size_t out_cap, size_t *out_len)
{
	(void)in;
	(void)in_len;
	(void)out;
	(void)out_cap;
	(void)out_len;
	return (MEMBRANE_ERR_UNIMPLEMENTED);
}

static membrane_status_t	unimpl_decompress(const uint8_t *in, size_t in_len,
				uint8_t *out, size_t out_cap, size_t *out_len)
{
	(void)in;
	(void)in_len;
	(void)out;
	(void)out_cap;
	(void)out_len;
	return (MEMBRANE_ERR_UNIMPLEMENTED);
}

/*
 * Returns in_len so callers pass the bound sanity check and reach
 * compress(), which reports MEMBRANE_ERR_UNIMPLEMENTED — the honest
 * error for a registered-but-unimplemented codec.
 */
static size_t	unimpl_bound(size_t in_len)
{
	return (in_len);
}

static const membrane_codec_vtable_t	g_membrane_codec_lz4 = {
	.name = "lz4",
	.id = MEMBRANE_CODEC_LZ4,
	.compress = unimpl_compress,
	.decompress = unimpl_decompress,
	.bound = unimpl_bound,
};

static const membrane_codec_vtable_t	g_membrane_codec_bitpack = {
	.name = "bitpack",
	.id = MEMBRANE_CODEC_BITPACK,
	.compress = unimpl_compress,
	.decompress = unimpl_decompress,
	.bound = unimpl_bound,
};

const membrane_codec_vtable_t	*membrane_codec_get(membrane_codec_t id)
{
	static const membrane_codec_vtable_t	*const table[] = {
		&g_membrane_codec_raw,
		&g_membrane_codec_rle,
		&g_membrane_codec_lz4,
		&g_membrane_codec_bitpack,
	};

	if ((int)id < 0 || id >= MEMBRANE_CODEC_COUNT)
		return (NULL);
	return (table[id]);
}

int	membrane_codec_from_name(const char *name, membrane_codec_t *out_id)
{
	const membrane_codec_vtable_t	*codec;
	int								id;

	if (name == NULL || out_id == NULL)
		return (0);
	id = 0;
	while (id < MEMBRANE_CODEC_COUNT)
	{
		codec = membrane_codec_get((membrane_codec_t)id);
		if (codec != NULL && strcmp(codec->name, name) == 0)
		{
			*out_id = (membrane_codec_t)id;
			return (1);
		}
		id++;
	}
	return (0);
}
