#define _DEFAULT_SOURCE

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "membrane/block.h"
#include "backend_internal.h"

/*
 * On-disk record: a 48-byte header followed by the compressed payload.
 * Filenames are derived only from the numeric id (16 hex digits), so no
 * caller string ever reaches the path — there is no traversal surface.
 */
# define FILE_MAGIC 0x314b424dU		/* "MBK1" */
# define FILE_VERSION 1U
# define HEADER_SIZE 48U
# define PATH_CAP 4096

typedef struct s_file_backend
{
	char	*dir;
	size_t	dir_len;
}	file_backend_t;

static void	put_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static void	put_le64(uint8_t *p, uint64_t v)
{
	put_le32(p, (uint32_t)v);
	put_le32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t	get_le32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static uint64_t	get_le64(const uint8_t *p)
{
	return ((uint64_t)get_le32(p) | ((uint64_t)get_le32(p + 4) << 32));
}

static int	build_path(const file_backend_t *fb, uint64_t id,
				const char *suffix, char *out)
{
	int	n;

	n = snprintf(out, PATH_CAP, "%s/%016llx.mbk%s",
			fb->dir, (unsigned long long)id, suffix);
	if (n < 0 || n >= PATH_CAP)
		return (-1);
	return (0);
}

static void	fill_header(uint8_t *h, const membrane_block_t *b)
{
	put_le32(h + 0, FILE_MAGIC);
	put_le32(h + 4, FILE_VERSION);
	put_le64(h + 8, b->id);
	put_le64(h + 16, b->original_size);
	put_le64(h + 24, b->stored_size);
	put_le32(h + 32, (uint32_t)b->requested_codec);
	put_le32(h + 36, (uint32_t)b->stored_codec);
	put_le32(h + 40, b->checksum);
	put_le32(h + 44, membrane_block_checksum(h, 44));
}

static membrane_status_t	write_record(FILE *f, const membrane_block_t *b)
{
	uint8_t	header[HEADER_SIZE];

	fill_header(header, b);
	if (fwrite(header, 1, HEADER_SIZE, f) != HEADER_SIZE)
		return (MEMBRANE_ERR_IO);
	if (b->stored_size > 0
		&& fwrite(b->data, 1, b->stored_size, f) != b->stored_size)
		return (MEMBRANE_ERR_IO);
	if (fflush(f) != 0 || fsync(fileno(f)) != 0)
		return (MEMBRANE_ERR_IO);
	return (MEMBRANE_OK);
}

static membrane_status_t	file_store(void *ctx, const membrane_block_t *b)
{
	file_backend_t		*fb;
	char				tmp[PATH_CAP];
	char				final[PATH_CAP];
	FILE				*f;
	membrane_status_t	status;

	fb = ctx;
	if (build_path(fb, b->id, ".tmp", tmp) != 0
		|| build_path(fb, b->id, "", final) != 0)
		return (MEMBRANE_ERR_IO);
	f = fopen(tmp, "wb");
	if (f == NULL)
		return (MEMBRANE_ERR_IO);
	status = write_record(f, b);
	if (fclose(f) != 0)
		status = MEMBRANE_ERR_IO;
	if (status == MEMBRANE_OK && rename(tmp, final) == 0)
		return (MEMBRANE_OK);
	unlink(tmp);
	return (MEMBRANE_ERR_IO);
}

/* Validates the header and returns the payload length, or -1 on corruption. */
static long	parse_header(const uint8_t *h, uint64_t id, long file_size)
{
	uint64_t	stored;

	if (get_le32(h + 0) != FILE_MAGIC || get_le32(h + 4) != FILE_VERSION)
		return (-1);
	if (get_le32(h + 44) != membrane_block_checksum(h, 44))
		return (-1);
	if (get_le64(h + 8) != id)
		return (-1);
	stored = get_le64(h + 24);
	if (stored > (uint64_t)(file_size - (long)HEADER_SIZE))
		return (-1);
	return ((long)stored);
}

static membrane_block_t	*build_block(const uint8_t *h, uint8_t *payload,
								size_t stored)
{
	membrane_block_t	*b;

	b = calloc(1, sizeof(*b));
	if (b == NULL)
		return (NULL);
	b->id = get_le64(h + 8);
	b->original_size = (size_t)get_le64(h + 16);
	b->stored_size = stored;
	b->requested_codec = (membrane_codec_t)get_le32(h + 32);
	b->stored_codec = (membrane_codec_t)get_le32(h + 36);
	b->checksum = get_le32(h + 40);
	b->data = payload;
	return (b);
}

static long	file_size_of(FILE *f)
{
	long	sz;

	if (fseek(f, 0, SEEK_END) != 0)
		return (-1);
	sz = ftell(f);
	if (sz < (long)HEADER_SIZE || fseek(f, 0, SEEK_SET) != 0)
		return (-1);
	return (sz);
}

static membrane_status_t	read_payload(FILE *f, size_t stored, uint8_t **out)
{
	uint8_t	*payload;

	payload = NULL;
	if (stored > 0)
	{
		payload = malloc(stored);
		if (payload == NULL)
			return (MEMBRANE_ERR_ALLOC_FAILED);
		if (fread(payload, 1, stored, f) != stored)
			return (free(payload), MEMBRANE_ERR_CORRUPT_DATA);
	}
	*out = payload;
	return (MEMBRANE_OK);
}

static membrane_status_t	load_open(FILE *f, uint64_t id,
								membrane_block_t **out)
{
	uint8_t				header[HEADER_SIZE];
	uint8_t				*payload;
	long				fsz;
	long				stored;
	membrane_status_t	status;

	fsz = file_size_of(f);
	if (fsz < 0 || fread(header, 1, HEADER_SIZE, f) != HEADER_SIZE)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	stored = parse_header(header, id, fsz);
	if (stored < 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	status = read_payload(f, (size_t)stored, &payload);
	if (status != MEMBRANE_OK)
		return (status);
	*out = build_block(header, payload, (size_t)stored);
	if (*out == NULL)
		return (free(payload), MEMBRANE_ERR_ALLOC_FAILED);
	return (MEMBRANE_OK);
}

static membrane_status_t	file_load(void *ctx, uint64_t id,
								membrane_block_t **out)
{
	file_backend_t		*fb;
	char				path[PATH_CAP];
	FILE				*f;
	membrane_status_t	status;

	fb = ctx;
	if (build_path(fb, id, "", path) != 0)
		return (MEMBRANE_ERR_IO);
	f = fopen(path, "rb");
	if (f == NULL)
		return (MEMBRANE_ERR_NOT_FOUND);
	status = load_open(f, id, out);
	fclose(f);
	return (status);
}

static membrane_status_t	file_remove(void *ctx, uint64_t id)
{
	file_backend_t	*fb;
	char			path[PATH_CAP];

	fb = ctx;
	if (build_path(fb, id, "", path) != 0)
		return (MEMBRANE_ERR_IO);
	if (unlink(path) != 0 && errno != ENOENT)
		return (MEMBRANE_ERR_IO);
	return (MEMBRANE_OK);
}

static int	file_contains(void *ctx, uint64_t id)
{
	file_backend_t	*fb;
	char			path[PATH_CAP];

	fb = ctx;
	if (build_path(fb, id, "", path) != 0)
		return (0);
	return (access(path, F_OK) == 0);
}

static int	name_is_record(const char *name)
{
	size_t	len;

	len = strlen(name);
	return (len > 4 && strcmp(name + len - 4, ".mbk") == 0);
}

static uint64_t	file_used_bytes(void *ctx)
{
	file_backend_t	*fb;
	DIR				*d;
	struct dirent	*e;
	char			path[PATH_CAP];
	uint64_t		total;
	struct stat		st;

	fb = ctx;
	d = opendir(fb->dir);
	if (d == NULL)
		return (0);
	total = 0;
	e = readdir(d);
	while (e != NULL)
	{
		if (name_is_record(e->d_name)
			&& snprintf(path, PATH_CAP, "%s/%s", fb->dir, e->d_name) < PATH_CAP
			&& stat(path, &st) == 0)
			total += (uint64_t)st.st_size;
		e = readdir(d);
	}
	closedir(d);
	return (total);
}

static int	name_is_ours(const char *name)
{
	return (strstr(name, ".mbk") != NULL);
}

static void	file_destroy(void *ctx)
{
	file_backend_t	*fb;
	DIR				*d;
	struct dirent	*e;
	char			path[PATH_CAP];

	fb = ctx;
	d = opendir(fb->dir);
	while (d != NULL && (e = readdir(d)) != NULL)
	{
		if (name_is_ours(e->d_name)
			&& snprintf(path, PATH_CAP, "%s/%s", fb->dir, e->d_name) < PATH_CAP)
			unlink(path);
	}
	if (d != NULL)
		closedir(d);
	free(fb->dir);
	free(fb);
}

static const membrane_backend_vtable_t	g_file_vtable = {
	.store = file_store,
	.load = file_load,
	.remove = file_remove,
	.contains = file_contains,
	.used_bytes = file_used_bytes,
	.destroy = file_destroy,
};

static file_backend_t	*file_ctx_create(const char *dir)
{
	file_backend_t	*fb;

	fb = calloc(1, sizeof(*fb));
	if (fb == NULL)
		return (NULL);
	fb->dir_len = strlen(dir);
	fb->dir = malloc(fb->dir_len + 1);
	if (fb->dir == NULL)
		return (free(fb), (file_backend_t *)NULL);
	memcpy(fb->dir, dir, fb->dir_len + 1);
	return (fb);
}

membrane_backend_t	*membrane_backend_file_create(const char *dir)
{
	membrane_backend_t	*be;
	file_backend_t		*fb;
	struct stat			st;

	if (dir == NULL || strlen(dir) + 32 >= PATH_CAP
		|| stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
		return (NULL);
	be = calloc(1, sizeof(*be));
	fb = file_ctx_create(dir);
	if (be == NULL || fb == NULL)
	{
		free(be);
		if (fb != NULL)
			free(fb->dir);
		free(fb);
		return (NULL);
	}
	be->vt = &g_file_vtable;
	be->ctx = fb;
	return (be);
}
