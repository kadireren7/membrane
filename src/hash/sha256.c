#include <stdio.h>
#include <string.h>

#include "membrane/hash.h"

typedef membrane_sha256_ctx_t	sha256_ctx_t;

static const uint32_t	g_k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint32_t	rotr(uint32_t x, uint32_t n)
{
	return ((x >> n) | (x << (32 - n)));
}

static void	sha256_init(sha256_ctx_t *ctx)
{
	static const uint32_t	iv[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	memcpy(ctx->state, iv, sizeof(iv));
	ctx->total_len = 0;
	ctx->buf_len = 0;
}

static void	sha256_compress(sha256_ctx_t *ctx, const uint8_t block[64])
{
	uint32_t	w[64];
	uint32_t	a;
	uint32_t	b;
	uint32_t	c;
	uint32_t	d;
	uint32_t	e;
	uint32_t	f;
	uint32_t	g;
	uint32_t	h;
	uint32_t	s0;
	uint32_t	s1;
	uint32_t	ch;
	uint32_t	maj;
	uint32_t	t1;
	uint32_t	t2;
	int			i;

	i = 0;
	while (i < 16)
	{
		w[i] = ((uint32_t)block[i * 4] << 24)
			| ((uint32_t)block[i * 4 + 1] << 16)
			| ((uint32_t)block[i * 4 + 2] << 8)
			| ((uint32_t)block[i * 4 + 3]);
		i++;
	}
	while (i < 64)
	{
		s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		i++;
	}
	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];
	i = 0;
	while (i < 64)
	{
		s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
		ch = (e & f) ^ (~e & g);
		t1 = h + s1 + ch + g_k[i] + w[i];
		s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
		i++;
	}
	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

static void	sha256_update(sha256_ctx_t *ctx, const uint8_t *data, size_t len)
{
	size_t	take;

	ctx->total_len += len;
	while (len > 0)
	{
		take = sizeof(ctx->buf) - ctx->buf_len;
		if (take > len)
			take = len;
		memcpy(ctx->buf + ctx->buf_len, data, take);
		ctx->buf_len += take;
		data += take;
		len -= take;
		if (ctx->buf_len == sizeof(ctx->buf))
		{
			sha256_compress(ctx, ctx->buf);
			ctx->buf_len = 0;
		}
	}
}

static void	sha256_final(sha256_ctx_t *ctx, uint8_t out[MEMBRANE_SHA256_DIGEST_BYTES])
{
	uint64_t	bit_len;
	uint8_t		pad;
	int			i;

	bit_len = ctx->total_len * 8;
	pad = 0x80;
	sha256_update(ctx, &pad, 1);
	pad = 0x00;
	while (ctx->buf_len != 56)
		sha256_update(ctx, &pad, 1);
	i = 7;
	while (i >= 0)
	{
		pad = (uint8_t)(bit_len >> (i * 8));
		sha256_update(ctx, &pad, 1);
		i--;
	}
	i = 0;
	while (i < 8)
	{
		out[i * 4] = (uint8_t)(ctx->state[i] >> 24);
		out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
		out[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
		out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
		i++;
	}
}

void	membrane_sha256_init(membrane_sha256_ctx_t *ctx)
{
	sha256_init(ctx);
}

void	membrane_sha256_update(membrane_sha256_ctx_t *ctx,
			const uint8_t *data, size_t len)
{
	if (len > 0)
		sha256_update(ctx, data, len);
}

void	membrane_sha256_final(membrane_sha256_ctx_t *ctx,
			uint8_t out_digest[MEMBRANE_SHA256_DIGEST_BYTES])
{
	sha256_final(ctx, out_digest);
}

void	membrane_sha256(const uint8_t *data, size_t len,
			uint8_t out_digest[MEMBRANE_SHA256_DIGEST_BYTES])
{
	sha256_ctx_t	ctx;

	sha256_init(&ctx);
	if (len > 0)
		sha256_update(&ctx, data, len);
	sha256_final(&ctx, out_digest);
}

void	membrane_sha256_to_hex(
			const uint8_t digest[MEMBRANE_SHA256_DIGEST_BYTES],
			char out_hex[MEMBRANE_SHA256_HEX_LEN + 1])
{
	static const char	hex[] = "0123456789abcdef";
	int					i;

	i = 0;
	while (i < MEMBRANE_SHA256_DIGEST_BYTES)
	{
		out_hex[i * 2] = hex[digest[i] >> 4];
		out_hex[i * 2 + 1] = hex[digest[i] & 0x0f];
		i++;
	}
	out_hex[MEMBRANE_SHA256_HEX_LEN] = '\0';
}

membrane_status_t	membrane_sha256_file(const char *path,
						char out_hex[MEMBRANE_SHA256_HEX_LEN + 1])
{
	FILE			*f;
	sha256_ctx_t	ctx;
	uint8_t			buf[65536];
	uint8_t			digest[MEMBRANE_SHA256_DIGEST_BYTES];
	size_t			n;

	f = fopen(path, "rb");
	if (f == NULL)
		return (MEMBRANE_ERR_IO);
	sha256_init(&ctx);
	n = fread(buf, 1, sizeof(buf), f);
	while (n > 0)
	{
		sha256_update(&ctx, buf, n);
		n = fread(buf, 1, sizeof(buf), f);
	}
	if (ferror(f))
		return (fclose(f), MEMBRANE_ERR_IO);
	fclose(f);
	sha256_final(&ctx, digest);
	membrane_sha256_to_hex(digest, out_hex);
	return (MEMBRANE_OK);
}
