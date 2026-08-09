#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "export_core.h"
#include "trace_format.h"

int	membrane_export_record_selected(const membrane_kv_header_t *h,
			const membrane_export_range_t *range)
{
	if (h->dtype != MEMBRANE_EXPORT_DTYPE_F16_GGML)
		return (0);
	if (range->layer_start >= 0 && h->layer < (uint32_t)range->layer_start)
		return (0);
	if (range->layer_end >= 0 && h->layer > (uint32_t)range->layer_end)
		return (0);
	if (range->tensor_filter != MEMBRANE_EXPORT_TENSOR_BOTH
		&& h->tensor_type != (uint32_t)range->tensor_filter)
		return (0);
	return (1);
}

void	membrane_export_batch_filename(char *out, size_t out_cap,
			uint32_t layer, uint32_t tensor_type)
{
	snprintf(out, out_cap, "layer-%03u-%c.memkv", layer,
		tensor_type == MEMBRANE_KV_TENSOR_K ? 'k' : 'v');
}

typedef struct s_seen_entry
{
	uint32_t	layer;
	uint32_t	tensor_type;
}	seen_entry_t;

static int	already_seen(const seen_entry_t *seen, uint32_t n,
				uint32_t layer, uint32_t tensor_type)
{
	uint32_t	i;

	i = 0;
	while (i < n)
	{
		if (seen[i].layer == layer && seen[i].tensor_type == tensor_type)
			return (1);
		i++;
	}
	return (0);
}

/* Deliberately omits h->model: single-record --output's build_label()
 * (main.c) includes it because that path exports exactly one
 * caller-chosen record, but a batch run may write dozens of files from
 * one capture, and this label is the only place per-file metadata
 * lands -- keeping it to layer/tensor/token-count fields the caller
 * already controls via --layer-start/--layer-end/--tensor avoids
 * carrying a capture-supplied (not this tool's) string into every
 * output file by default. */
static void	build_batch_label(char *out, size_t out_cap,
				const membrane_kv_header_t *h)
{
	snprintf(out, out_cap, "layer=%u tensor=%s tokens=%u", h->layer,
		h->tensor_type == MEMBRANE_KV_TENSOR_K ? "K" : "V",
		h->token_end - h->token_start);
}

/* Same block-count derivation as main.c's single-record path: whole
 * F16 elements, then whole o->elements_per_block-sized blocks. Returns
 * 0 (with *out_block_count set to 0) if the record is too small for
 * even one block -- caller decides whether that is fatal or skippable. */
static membrane_status_t	compute_block_count(uint64_t payload_size,
								uint32_t elements_per_block,
								uint64_t *out_block_count)
{
	uint64_t	total_elems;

	if (payload_size % sizeof(uint16_t) != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	total_elems = payload_size / sizeof(uint16_t);
	*out_block_count = total_elems / elements_per_block;
	return (MEMBRANE_OK);
}

static membrane_status_t	export_one(const char *output_dir,
								const membrane_kv_header_t *h,
								const uint8_t *payload,
								uint32_t elements_per_block,
								uint64_t block_count, char *err_buf,
								size_t err_cap)
{
	char				filename[64];
	char				path[4096];
	char				label[192];
	FILE				*out_f;
	membrane_status_t	st;
	int					fd;

	membrane_export_batch_filename(filename, sizeof(filename), h->layer,
		h->tensor_type);
	if (snprintf(path, sizeof(path), "%s/%s", output_dir, filename)
		>= (int)sizeof(path))
	{
		snprintf(err_buf, err_cap, "output path too long for %s", filename);
		return (MEMBRANE_ERR_INVALID_ARG);
	}
	/* O_EXCL: refuse to follow a pre-existing symlink at this
	 * predictable, deterministic path (a shared/attacker-writable
	 * output directory could plant one to redirect the write) and
	 * refuse to silently overwrite a stale file left by an earlier,
	 * narrower --layer-start/--layer-end run -- either case fails
	 * clearly instead of truncating an unexpected target. */
	fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
	if (fd < 0)
	{
		snprintf(err_buf, err_cap, "cannot create %s (already exists, or "
			"a symlink -- refusing to overwrite/follow it)", path);
		return (MEMBRANE_ERR_IO);
	}
	out_f = fdopen(fd, "wb");
	if (out_f == NULL)
	{
		close(fd);
		snprintf(err_buf, err_cap, "cannot create %s", path);
		return (MEMBRANE_ERR_IO);
	}
	build_batch_label(label, sizeof(label), h);
	st = membrane_trace_write(out_f, MEMBRANE_TRACE_DTYPE_F16,
			elements_per_block, block_count, (const uint16_t *)payload,
			label, (uint64_t)time(NULL));
	if (st == MEMBRANE_OK && fclose(out_f) != 0)
		st = MEMBRANE_ERR_IO;
	else if (st != MEMBRANE_OK)
		fclose(out_f);
	if (st != MEMBRANE_OK)
		snprintf(err_buf, err_cap, "write failed for %s", path);
	return (st);
}

static int	dir_exists(const char *path)
{
	struct stat	st;

	if (stat(path, &st) != 0)
		return (0);
	return (S_ISDIR(st.st_mode) ? 1 : 0);
}

membrane_status_t	membrane_export_batch_run(
						const membrane_export_batch_opts_t *o,
						membrane_export_batch_result_t *out,
						char *err_buf, size_t err_cap)
{
	FILE				*in_f;
	membrane_kv_header_t	h;
	uint8_t				*payload;
	uint64_t				block_count;
	seen_entry_t			*seen;
	uint32_t				seen_count;
	membrane_status_t		st;
	char					discard_buf[1];

	memset(out, 0, sizeof(*out));
	/* Every error path below writes unconditionally -- redirect a
	 * NULL/zero-capacity err_buf to a throwaway buffer here once,
	 * rather than guarding every individual snprintf call site. */
	if (err_buf == NULL || err_cap == 0)
	{
		err_buf = discard_buf;
		err_cap = sizeof(discard_buf);
	}
	err_buf[0] = '\0';
	if (o == NULL || o->input_path == NULL || o->output_dir == NULL
		|| o->elements_per_block == 0 || o->elements_per_block % 32 != 0)
		return (snprintf(err_buf, err_cap, "invalid batch export arguments"),
			MEMBRANE_ERR_INVALID_ARG);
	if (!dir_exists(o->output_dir))
		return (snprintf(err_buf, err_cap,
				"output directory does not exist or is not writable: %s",
				o->output_dir), MEMBRANE_ERR_IO);
	in_f = fopen(o->input_path, "rb");
	if (in_f == NULL)
		return (snprintf(err_buf, err_cap, "cannot open %s", o->input_path),
			MEMBRANE_ERR_IO);
	seen = malloc(MEMBRANE_EXPORT_BATCH_MAX_RECORDS * sizeof(*seen));
	if (seen == NULL)
		return (fclose(in_f), snprintf(err_buf, err_cap, "out of memory"),
			MEMBRANE_ERR_ALLOC_FAILED);
	seen_count = 0;
	st = membrane_kvdump_read_header(in_f, &h);
	while (st == MEMBRANE_OK)
	{
		if (!membrane_export_record_selected(&h, &o->range))
		{
			st = membrane_kvdump_read_payload(in_f, &h, &payload);
			free(payload);
		}
		else if (seen_count >= MEMBRANE_EXPORT_BATCH_MAX_RECORDS)
		{
			st = MEMBRANE_ERR_INVALID_ARG;
			snprintf(err_buf, err_cap, "more than %u candidate records "
				"(cap)", MEMBRANE_EXPORT_BATCH_MAX_RECORDS);
			break ;
		}
		else if (already_seen(seen, seen_count, h.layer, h.tensor_type))
		{
			st = MEMBRANE_ERR_INVALID_ARG;
			snprintf(err_buf, err_cap, "duplicate layer=%u tensor=%s F16 "
				"record in %s -- corrupt capture?", h.layer,
				h.tensor_type == MEMBRANE_KV_TENSOR_K ? "K" : "V",
				o->input_path);
			break ;
		}
		else
		{
			/* Recorded as seen -- for both the MAX_RECORDS cap and
			 * duplicate detection -- as soon as a record is SELECTED,
			 * not only once it's actually exported: otherwise an
			 * unlimited run of too-small selected records would bypass
			 * the cap entirely, and a too-small record followed later
			 * by a real duplicate of the same (layer, tensor) would
			 * never be caught (the first occurrence never having made
			 * it into `seen`). */
			seen[seen_count].layer = h.layer;
			seen[seen_count].tensor_type = h.tensor_type;
			seen_count++;
			st = membrane_kvdump_read_payload(in_f, &h, &payload);
			if (st == MEMBRANE_OK)
			{
				st = compute_block_count(h.payload_size,
						o->elements_per_block, &block_count);
				if (st == MEMBRANE_OK && block_count == 0)
				{
					out->skipped_too_small_count++;
					st = MEMBRANE_OK;
				}
				else if (st == MEMBRANE_OK)
				{
					st = export_one(o->output_dir, &h, payload,
							o->elements_per_block, block_count, err_buf,
							err_cap);
					if (st == MEMBRANE_OK)
						out->exported_count++;
				}
				free(payload);
			}
		}
		if (st != MEMBRANE_OK)
			break ;
		st = membrane_kvdump_read_header(in_f, &h);
	}
	free(seen);
	fclose(in_f);
	if (st == MEMBRANE_ERR_NOT_FOUND)
		return (MEMBRANE_OK);
	if (st != MEMBRANE_OK && err_buf != NULL && err_cap > 0
		&& err_buf[0] == '\0')
		snprintf(err_buf, err_cap, "error reading %s: status=%d (truncated "
			"or corrupt file?)", o->input_path, (int)st);
	return (st);
}
