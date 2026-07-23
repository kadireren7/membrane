#ifndef MEMBRANE_CODEC_H
# define MEMBRANE_CODEC_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

typedef enum e_membrane_codec
{
	MEMBRANE_CODEC_RAW = 0,
	MEMBRANE_CODEC_RLE,
	MEMBRANE_CODEC_LZ4,
	MEMBRANE_CODEC_BITPACK,
	MEMBRANE_CODEC_COUNT
}	membrane_codec_t;

typedef enum e_membrane_status
{
	MEMBRANE_OK = 0,
	MEMBRANE_ERR_INVALID_ARG,
	MEMBRANE_ERR_BUFFER_TOO_SMALL,
	MEMBRANE_ERR_CORRUPT_DATA,
	MEMBRANE_ERR_UNIMPLEMENTED,
	MEMBRANE_ERR_ALLOC_FAILED
}	membrane_status_t;

/*
 * Compress `in` (in_len bytes) into `out` (capacity out_cap).
 * On success, returns MEMBRANE_OK and sets *out_len to the number of
 * bytes written. Must not write past out_cap; if out_cap is
 * insufficient, returns MEMBRANE_ERR_BUFFER_TOO_SMALL without partial
 * writes.
 */
typedef membrane_status_t	(*membrane_compress_fn)(
	const uint8_t *in, size_t in_len,
	uint8_t *out, size_t out_cap, size_t *out_len);

/*
 * Decompress `in` (in_len bytes) into `out` (capacity out_cap).
 * On success, returns MEMBRANE_OK and sets *out_len to the number of
 * bytes written. Must detect malformed/truncated input and return
 * MEMBRANE_ERR_CORRUPT_DATA rather than reading or writing out of
 * bounds.
 */
typedef membrane_status_t	(*membrane_decompress_fn)(
	const uint8_t *in, size_t in_len,
	uint8_t *out, size_t out_cap, size_t *out_len);

/*
 * Worst-case output size for compressing an input of `in_len` bytes.
 * Callers use this to size their output buffer before calling compress.
 */
typedef size_t				(*membrane_bound_fn)(size_t in_len);

typedef struct s_membrane_codec_vtable
{
	const char				*name;
	membrane_codec_t		id;
	membrane_compress_fn	compress;
	membrane_decompress_fn	decompress;
	membrane_bound_fn		bound;
}	membrane_codec_vtable_t;

/* Returns NULL if `id` is out of range. */
const membrane_codec_vtable_t	*membrane_codec_get(membrane_codec_t id);

/*
 * Looks up a codec by its short name (e.g. "raw", "rle").
 * Returns 1 and sets *out_id on match, 0 if no codec matches `name`.
 */
int								membrane_codec_from_name(const char *name,
									membrane_codec_t *out_id);

# ifdef __cplusplus
}
# endif

#endif
