#ifndef MEMBRANE_HASH_H
# define MEMBRANE_HASH_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

# define MEMBRANE_SHA256_DIGEST_BYTES	32
# define MEMBRANE_SHA256_HEX_LEN		64

/* Standard FIPS 180-4 SHA-256, one-shot over an in-memory buffer. */
void				membrane_sha256(const uint8_t *data, size_t len,
						uint8_t out_digest[MEMBRANE_SHA256_DIGEST_BYTES]);

/*
 * Incremental SHA-256, for hashing a byte range too large to hold in
 * memory at once (e.g. an out-of-core trace file's payload region --
 * see attntrace3.h) -- membrane_sha256() and membrane_sha256_file()
 * are themselves thin wrappers around this same state machine.
 */
typedef struct s_membrane_sha256_ctx
{
	uint32_t	state[8];
	uint64_t	total_len;
	uint8_t		buf[64];
	size_t		buf_len;
}	membrane_sha256_ctx_t;

void				membrane_sha256_init(membrane_sha256_ctx_t *ctx);
void				membrane_sha256_update(membrane_sha256_ctx_t *ctx,
						const uint8_t *data, size_t len);
void				membrane_sha256_final(membrane_sha256_ctx_t *ctx,
						uint8_t out_digest[MEMBRANE_SHA256_DIGEST_BYTES]);

/*
 * Lowercase hex encoding of `digest`, e.g. for embedding in a policy
 * file or comparing against a documented model hash. `out_hex` must hold
 * at least MEMBRANE_SHA256_HEX_LEN + 1 bytes (the +1 is the NUL).
 */
void				membrane_sha256_to_hex(
						const uint8_t digest[MEMBRANE_SHA256_DIGEST_BYTES],
						char out_hex[MEMBRANE_SHA256_HEX_LEN + 1]);

/*
 * Hashes the whole file at `path` and hex-encodes the digest into
 * `out_hex` (see membrane_sha256_to_hex). Returns MEMBRANE_ERR_IO if the
 * file cannot be opened or read.
 */
membrane_status_t	membrane_sha256_file(const char *path,
						char out_hex[MEMBRANE_SHA256_HEX_LEN + 1]);

# ifdef __cplusplus
}
# endif

#endif
