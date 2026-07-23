#ifndef MEMBRANE_KVDUMP_H
# define MEMBRANE_KVDUMP_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Versioned binary container for KV-cache tensor dumps. A dump file is a
 * sequence of records, each a fixed 144-byte little-endian header followed
 * by the raw tensor payload. The header carries its own CRC32 and the
 * payload's CRC32, so corruption and truncation are both detectable.
 */
# define MEMBRANE_KV_MAGIC 0x31564B4DU	/* "MKV1" */
# define MEMBRANE_KV_VERSION 1U
# define MEMBRANE_KV_HEADER_SIZE 144U
# define MEMBRANE_KV_MODEL_CAP 64
# define MEMBRANE_KV_MAX_DIMS 4
# define MEMBRANE_KV_MAX_PAYLOAD (1ull << 40)

typedef enum e_membrane_kv_tensor
{
	MEMBRANE_KV_TENSOR_K = 0,
	MEMBRANE_KV_TENSOR_V = 1
}	membrane_kv_tensor_t;

typedef struct s_membrane_kv_header
{
	char		model[MEMBRANE_KV_MODEL_CAP];
	uint32_t	layer;
	uint32_t	tensor_type;	/* membrane_kv_tensor_t */
	uint32_t	token_start;
	uint32_t	token_end;
	uint32_t	dtype;			/* producer's element type id (e.g. ggml) */
	uint32_t	n_dims;
	uint64_t	dims[MEMBRANE_KV_MAX_DIMS];
	uint64_t	payload_size;
	uint32_t	checksum;		/* CRC32 of the payload */
}	membrane_kv_header_t;

/* Appends one record. payload may be NULL only when payload_size is 0. */
membrane_status_t	membrane_kvdump_write(FILE *f,
						const membrane_kv_header_t *h, const void *payload);

/*
 * Reads the next record header. Returns MEMBRANE_ERR_NOT_FOUND at clean
 * EOF, MEMBRANE_ERR_CORRUPT_DATA on a bad magic/version/CRC or an
 * implausible size.
 */
membrane_status_t	membrane_kvdump_read_header(FILE *f,
						membrane_kv_header_t *h);

/*
 * Reads the payload for a just-read header into a caller-owned buffer
 * (free() it). Verifies the payload CRC; truncation or a mismatch is
 * MEMBRANE_ERR_CORRUPT_DATA. For empty payloads *out is set to NULL.
 */
membrane_status_t	membrane_kvdump_read_payload(FILE *f,
						const membrane_kv_header_t *h, uint8_t **out);

# ifdef __cplusplus
}
# endif

#endif
