#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "export_core.h"
#include "membrane/block.h"
#include "test_helpers.h"
#include "trace_format.h"

static char	g_kvdump_path[] = "/tmp/membrane-export-core-kvdump-XXXXXX";
static char	g_outdir_path[] = "/tmp/membrane-export-core-dir-XXXXXX";

static void	write_record(FILE *f, uint32_t layer, uint32_t tensor_type,
				uint32_t dtype, uint32_t n_elems)
{
	membrane_kv_header_t	h;
	uint16_t				*payload;
	uint32_t				i;

	payload = malloc((size_t)n_elems * sizeof(uint16_t));
	TEST_ASSERT(payload != NULL, "allocate fixture payload");
	i = 0;
	while (i < n_elems)
	{
		payload[i] = (uint16_t)(i * 7u + layer * 131u + tensor_type);
		i++;
	}
	memset(&h, 0, sizeof(h));
	snprintf(h.model, sizeof(h.model), "test-model");
	h.layer = layer;
	h.tensor_type = tensor_type;
	h.token_start = 0;
	h.token_end = 64;
	h.dtype = dtype;
	h.n_dims = 2;
	h.dims[0] = n_elems;
	h.dims[1] = 1;
	h.payload_size = (uint64_t)n_elems * sizeof(uint16_t);
	h.checksum = membrane_block_checksum((const uint8_t *)payload,
			h.payload_size);
	TEST_ASSERT(membrane_kvdump_write(f, &h, payload) == MEMBRANE_OK,
		"write fixture record");
	free(payload);
}

/* layers 0..3, K then V per layer, all F16, 256 elements (2 blocks of
 * 128) -- mirrors membrane-kv-capture's own record order. */
static void	write_standard_fixture(void)
{
	FILE	*f;
	uint32_t	layer;

	f = fopen(g_kvdump_path, "wb");
	TEST_ASSERT(f != NULL, "open kvdump fixture for write");
	layer = 0;
	while (layer < 4)
	{
		write_record(f, layer, MEMBRANE_KV_TENSOR_K,
			MEMBRANE_EXPORT_DTYPE_F16_GGML, 256);
		write_record(f, layer, MEMBRANE_KV_TENSOR_V,
			MEMBRANE_EXPORT_DTYPE_F16_GGML, 256);
		layer++;
	}
	fclose(f);
}

static membrane_export_batch_opts_t	default_opts(void)
{
	membrane_export_batch_opts_t	o;

	o.input_path = g_kvdump_path;
	o.output_dir = g_outdir_path;
	o.range.layer_start = -1;
	o.range.layer_end = -1;
	o.range.tensor_filter = MEMBRANE_EXPORT_TENSOR_BOTH;
	o.elements_per_block = 128;
	return (o);
}

/* Every export test reuses the one fixture output directory, so tests
 * that assert a file is ABSENT must start from an empty directory --
 * otherwise a prior test's leftover file would make the assertion pass
 * for the wrong reason. */
static void	clean_outdir(void)
{
	DIR				*d;
	struct dirent	*ent;
	char			path[512];

	d = opendir(g_outdir_path);
	TEST_ASSERT(d != NULL, "reopen output dir fixture for cleanup");
	while ((ent = readdir(d)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
			continue ;
		snprintf(path, sizeof(path), "%s/%s", g_outdir_path, ent->d_name);
		unlink(path);
	}
	closedir(d);
}

static int	file_exists(const char *dir, const char *name)
{
	char	path[512];
	FILE	*f;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	f = fopen(path, "rb");
	if (f == NULL)
		return (0);
	fclose(f);
	return (1);
}

/* ------------------------------------------------------------------ */

static void	test_record_selected_filters(void)
{
	membrane_kv_header_t		h;
	membrane_export_range_t	range;

	memset(&h, 0, sizeof(h));
	h.dtype = MEMBRANE_EXPORT_DTYPE_F16_GGML;
	h.layer = 5;
	h.tensor_type = MEMBRANE_KV_TENSOR_K;
	range.layer_start = -1;
	range.layer_end = -1;
	range.tensor_filter = MEMBRANE_EXPORT_TENSOR_BOTH;
	TEST_ASSERT(membrane_export_record_selected(&h, &range),
		"unbounded range matches any F16 record");
	h.dtype = 0;	/* F32 */
	TEST_ASSERT(!membrane_export_record_selected(&h, &range),
		"non-F16 dtype never selected");
	h.dtype = MEMBRANE_EXPORT_DTYPE_F16_GGML;
	range.layer_start = 6;
	TEST_ASSERT(!membrane_export_record_selected(&h, &range),
		"layer below range rejected");
	range.layer_start = -1;
	range.layer_end = 4;
	TEST_ASSERT(!membrane_export_record_selected(&h, &range),
		"layer above range rejected");
	range.layer_end = -1;
	range.tensor_filter = MEMBRANE_KV_TENSOR_V;
	TEST_ASSERT(!membrane_export_record_selected(&h, &range),
		"tensor filter mismatch rejected");
	range.tensor_filter = MEMBRANE_KV_TENSOR_K;
	TEST_ASSERT(membrane_export_record_selected(&h, &range),
		"tensor filter match selected");
}

static void	test_batch_filename_deterministic(void)
{
	char	buf[64];

	membrane_export_batch_filename(buf, sizeof(buf), 0, MEMBRANE_KV_TENSOR_K);
	TEST_ASSERT(strcmp(buf, "layer-000-k.memkv") == 0,
		"zero-padded layer, lowercase k");
	membrane_export_batch_filename(buf, sizeof(buf), 42, MEMBRANE_KV_TENSOR_V);
	TEST_ASSERT(strcmp(buf, "layer-042-v.memkv") == 0,
		"zero-padded layer, lowercase v");
	membrane_export_batch_filename(buf, sizeof(buf), 1234,
		MEMBRANE_KV_TENSOR_K);
	TEST_ASSERT(strcmp(buf, "layer-1234-k.memkv") == 0,
		"layer wider than 3 digits is not truncated");
}

static void	test_batch_export_all_records(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	write_standard_fixture();
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	TEST_ASSERT(r.exported_count == 8, "4 layers x {K,V} = 8 files");
	TEST_ASSERT(r.skipped_too_small_count == 0, "nothing skipped");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-000-k.memkv"), "layer 0 K");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-000-v.memkv"), "layer 0 V");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-003-k.memkv"), "layer 3 K");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-003-v.memkv"), "layer 3 V");
}

static void	test_batch_export_round_trips_payload(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	membrane_trace_reader_t			*reader;
	membrane_trace_info_t			info;
	uint16_t						block[128];
	char							path[512];
	char							err[256];
	uint32_t						i;

	clean_outdir();

	write_standard_fixture();
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	snprintf(path, sizeof(path), "%s/layer-002-k.memkv", g_outdir_path);
	TEST_ASSERT(membrane_trace_open(path, &reader) == MEMBRANE_OK,
		"exported file opens as a valid trace");
	membrane_trace_info(reader, &info);
	TEST_ASSERT(info.block_count == 2 && info.elements_per_block == 128,
		"256 F16 elements -> 2 blocks of 128");
	TEST_ASSERT(membrane_trace_read_block(reader, block) == MEMBRANE_OK,
		"first block reads");
	TEST_ASSERT(block[0] == (uint16_t)(0 * 7u + 2u * 131u + 0u),
		"payload content matches the source record (layer=2, tensor=K)");
	i = 0;
	while (i < 128)
	{
		TEST_ASSERT(block[i] == (uint16_t)(i * 7u + 2u * 131u + 0u),
			"every element in the first block matches");
		i++;
	}
	membrane_trace_close(reader);
}

static void	test_batch_export_layer_range_filter(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	write_standard_fixture();
	o = default_opts();
	o.range.layer_start = 1;
	o.range.layer_end = 2;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	TEST_ASSERT(r.exported_count == 4, "layers 1-2 x {K,V} = 4 files");
	TEST_ASSERT(!file_exists(g_outdir_path, "layer-000-k.memkv"),
		"layer 0 excluded");
	TEST_ASSERT(!file_exists(g_outdir_path, "layer-003-k.memkv"),
		"layer 3 excluded");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-001-k.memkv"),
		"layer 1 included");
	TEST_ASSERT(file_exists(g_outdir_path, "layer-002-v.memkv"),
		"layer 2 included");
}

static void	test_batch_export_tensor_filter(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	write_standard_fixture();
	o = default_opts();
	o.range.tensor_filter = MEMBRANE_KV_TENSOR_K;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	TEST_ASSERT(r.exported_count == 4, "K only across 4 layers");
	TEST_ASSERT(!file_exists(g_outdir_path, "layer-000-v.memkv"),
		"V excluded");
}

static void	test_batch_export_skips_non_f16(void)
{
	FILE							*f;
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	f = fopen(g_kvdump_path, "wb");
	TEST_ASSERT(f != NULL, "open");
	write_record(f, 0, MEMBRANE_KV_TENSOR_K, MEMBRANE_EXPORT_DTYPE_F16_GGML,
		256);
	write_record(f, 1, MEMBRANE_KV_TENSOR_K, 0 /* F32, not F16 */, 256);
	fclose(f);
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	TEST_ASSERT(r.exported_count == 1, "only the F16 record is exported");
	TEST_ASSERT(!file_exists(g_outdir_path, "layer-001-k.memkv"),
		"non-F16 record produces no output file");
}

static void	test_batch_export_too_small_skipped_not_fatal(void)
{
	FILE							*f;
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	f = fopen(g_kvdump_path, "wb");
	TEST_ASSERT(f != NULL, "open");
	write_record(f, 0, MEMBRANE_KV_TENSOR_K, MEMBRANE_EXPORT_DTYPE_F16_GGML,
		64);	/* fewer than 128 elements: no whole block */
	write_record(f, 1, MEMBRANE_KV_TENSOR_K, MEMBRANE_EXPORT_DTYPE_F16_GGML,
		256);
	fclose(f);
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_OK, err);
	TEST_ASSERT(r.exported_count == 1, "the valid record still exports");
	TEST_ASSERT(r.skipped_too_small_count == 1,
		"the too-small record is skipped, not fatal");
}

static void	test_batch_export_duplicate_rejected(void)
{
	FILE							*f;
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	f = fopen(g_kvdump_path, "wb");
	TEST_ASSERT(f != NULL, "open");
	write_record(f, 0, MEMBRANE_KV_TENSOR_K, MEMBRANE_EXPORT_DTYPE_F16_GGML,
		256);
	write_record(f, 0, MEMBRANE_KV_TENSOR_K, MEMBRANE_EXPORT_DTYPE_F16_GGML,
		256);
	fclose(f);
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_INVALID_ARG,
		"a duplicate (layer, tensor) F16 record fails clearly");
	TEST_ASSERT(err[0] != '\0', "an error message is provided");
}

static void	test_batch_export_missing_output_dir_fails(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	write_standard_fixture();
	o = default_opts();
	o.output_dir = "/no/such/directory/at/all";
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_IO, "missing output directory rejected");
}

/* Defense in depth: membrane_export_batch_run is a public entry point
 * a caller could reach directly (bypassing main.c's own CLI-level
 * UINT32_MAX/reversed-range/tensor-filter checks) with an unvalidated
 * range -- it must reject these itself, not silently truncate/misbehave. */
static void	test_batch_export_rejects_invalid_range(void)
{
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();
	write_standard_fixture();
	o = default_opts();
	o.range.layer_start = (long)UINT32_MAX + 1;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_INVALID_ARG,
		"layer_start beyond UINT32_MAX rejected, not silently truncated");
	clean_outdir();
	o = default_opts();
	o.range.layer_end = (long)UINT32_MAX + 1;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_INVALID_ARG,
		"layer_end beyond UINT32_MAX rejected, not silently truncated");
	clean_outdir();
	o = default_opts();
	o.range.layer_start = 3;
	o.range.layer_end = 1;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_INVALID_ARG, "reversed range rejected");
	clean_outdir();
	o = default_opts();
	o.range.tensor_filter = 99;
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		== MEMBRANE_ERR_INVALID_ARG, "invalid tensor filter value rejected");
}

static void	test_batch_export_corrupt_input_fails(void)
{
	FILE							*f;
	membrane_export_batch_opts_t	o;
	membrane_export_batch_result_t	r;
	char							err[256];

	clean_outdir();

	f = fopen(g_kvdump_path, "wb");
	TEST_ASSERT(f != NULL, "open");
	fprintf(f, "not a kvdump file");
	fclose(f);
	o = default_opts();
	TEST_ASSERT(membrane_export_batch_run(&o, &r, err, sizeof(err))
		!= MEMBRANE_OK, "corrupt input rejected, not silently empty");
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_kvdump_path);
	TEST_ASSERT(fd >= 0, "kvdump fixture temp file");
	close(fd);
	TEST_ASSERT(mkdtemp(g_outdir_path) != NULL, "output dir fixture");
	test_record_selected_filters();
	test_batch_filename_deterministic();
	test_batch_export_all_records();
	test_batch_export_round_trips_payload();
	test_batch_export_layer_range_filter();
	test_batch_export_tensor_filter();
	test_batch_export_skips_non_f16();
	test_batch_export_too_small_skipped_not_fatal();
	test_batch_export_duplicate_rejected();
	test_batch_export_rejects_invalid_range();
	test_batch_export_missing_output_dir_fails();
	test_batch_export_corrupt_input_fails();
	unlink(g_kvdump_path);
	return (0);
}
