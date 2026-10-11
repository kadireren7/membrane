#ifndef MEMBRANE_SHA256_H
# define MEMBRANE_SHA256_H

# include <cstddef>
# include <cstdint>
# include <string>

/*
 * Milestone M: product-owned SHA-256 (FIPS 180-4). Downloads are verified
 * with this on every platform, so MEMBRANE needs no `sha256sum`/`shasum`/
 * `certutil` on the user's machine. Streaming, constant memory.
 */

typedef struct s_membrane_sha256
{
	uint32_t	h[8];
	uint8_t		buf[64];
	uint64_t	total;
	size_t		buflen;
}	membrane_sha256_t;

void		membrane_sha256_init(membrane_sha256_t *c);
void		membrane_sha256_update(membrane_sha256_t *c, const void *data,
				size_t len);
/* 64 lowercase hex characters. */
std::string	membrane_sha256_final_hex(membrane_sha256_t *c);

std::string	membrane_sha256_hex(const void *data, size_t len);
/* Streams the file in 1 MiB chunks. False if it cannot be opened/read. */
bool		membrane_sha256_file(const std::string &path, std::string *out_hex);

#endif
