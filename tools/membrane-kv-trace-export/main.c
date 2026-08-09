#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "membrane/kvdump.h"
#include "trace_format.h"

/*
 * membrane-kv-trace-export: converts an existing .kvdump record (written
 * by the llama-gated tools/membrane-kv-capture, which is unchanged by
 * this tool) into the portable .memkv trace format
 * tools/membrane-quant-policy-bench consumes via --trace.
 *
 * Deliberately built UNCONDITIONALLY (no MEMBRANE_ENABLE_LLAMA gate):
 * it never includes or links against llama.cpp, only
 * membrane/kvdump.h -- which is itself llama-free and already read
 * unconditionally by tools/membrane-kv-analyze. Only the ORIGINAL live
 * capture step (membrane-kv-capture) touches llama.cpp; this tool
 * converts its already-captured output offline.
 *
 * Privacy: membrane_kv_header_t (include/membrane/kvdump.h) carries no
 * prompt/token TEXT field at all -- only a model name string, a layer
 * index, a K/V tag, and token_start/token_end COUNTS. There is nothing
 * prompt-derived to strip; the metadata label this tool writes is
 * built solely from those safe fields (plus this trace file's own
 * basename), never from payload content.
 */

/* ggml's GGML_TYPE_F16 enumerator value, per the pinned llama.cpp
 * commit's ggml/include/ggml.h (GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1).
 * Not included from ggml.h directly -- that header only exists when
 * the llama.cpp submodule is checked out, and this tool must build
 * without it; membrane_kv_header_t.dtype is documented (kvdump.h) as
 * an opaque "producer's element type id", and membrane-kv-capture's
 * producer is always ggml. */
# define KVDUMP_DTYPE_F16_GGML		1u

# define DEFAULT_ELEMENTS_PER_BLOCK	128u

typedef struct s_export_opts
{
	const char	*input_path;
	const char	*output_path;
	const char	*label_override;
	long		layer;
	int			tensor;			/* -1 = unspecified, else membrane_kv_tensor_t */
	uint32_t	elements_per_block;
}	export_opts_t;

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-kv-trace-export --input FILE.kvdump "
		"--output FILE.memkv [options]\n"
		"\n"
		"Converts one record of an existing membrane-kv-capture .kvdump\n"
		"file into the portable .memkv trace format\n"
		"membrane-quant-policy-bench consumes via --trace. Builds and\n"
		"runs without llama.cpp -- it only reads an already-captured\n"
		"file.\n"
		"\n"
		"  --input FILE       source .kvdump file (required)\n"
		"  --output FILE      destination .memkv file (required)\n"
		"  --layer N          select this layer's record (default: the\n"
		"                     first F16 K/V record found)\n"
		"  --tensor k|v       select K or V (requires --layer)\n"
		"  --elements-per-block N\n"
		"                     block size, must be a positive multiple\n"
		"                     of 32 (default %u)\n"
		"  --label TEXT       override the trace's metadata label\n"
		"  --help             print this message and exit\n",
		DEFAULT_ELEMENTS_PER_BLOCK);
}

/* strtol/strtoul both return 0 on a non-numeric string with no error
 * indication of their own; callers must check errno and that the
 * whole argument was consumed. */
static int	parse_long(const char *s, long *out)
{
	char	*end;

	errno = 0;
	*out = strtol(s, &end, 10);
	return (errno == 0 && end != s && *end == '\0');
}

static int	parse_ulong(const char *s, unsigned long *out)
{
	char	*end;

	errno = 0;
	*out = strtoul(s, &end, 10);
	return (errno == 0 && end != s && *end == '\0');
}

static int	parse_opts(int argc, char **argv, export_opts_t *o)
{
	int	i;

	o->input_path = NULL;
	o->output_path = NULL;
	o->label_override = NULL;
	o->layer = -1;
	o->tensor = -1;
	o->elements_per_block = DEFAULT_ELEMENTS_PER_BLOCK;
	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--help") == 0)
			return (usage(stdout), 1);
		else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
			o->input_path = argv[++i];
		else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
			o->output_path = argv[++i];
		else if (strcmp(argv[i], "--label") == 0 && i + 1 < argc)
			o->label_override = argv[++i];
		else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc)
		{
			++i;
			if (!parse_long(argv[i], &o->layer) || o->layer < 0)
				return (fprintf(stderr,
						"--layer must be a non-negative integer\n"), -1);
		}
		else if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc)
		{
			++i;
			if (strcmp(argv[i], "k") == 0 || strcmp(argv[i], "K") == 0)
				o->tensor = MEMBRANE_KV_TENSOR_K;
			else if (strcmp(argv[i], "v") == 0 || strcmp(argv[i], "V") == 0)
				o->tensor = MEMBRANE_KV_TENSOR_V;
			else
				return (fprintf(stderr, "--tensor must be k or v\n"), -1);
		}
		else if (strcmp(argv[i], "--elements-per-block") == 0 && i + 1 < argc)
		{
			unsigned long	v;

			++i;
			if (!parse_ulong(argv[i], &v) || v == 0 || v > UINT32_MAX)
				return (fprintf(stderr,
						"--elements-per-block must be a positive integer\n"),
					-1);
			o->elements_per_block = (uint32_t)v;
		}
		else
			return (fprintf(stderr, "unknown option: %s\n", argv[i]), -1);
		i++;
	}
	if (o->input_path == NULL || o->output_path == NULL)
		return (fprintf(stderr, "--input and --output are required\n"), -1);
	if (o->elements_per_block == 0 || o->elements_per_block % 32 != 0)
		return (fprintf(stderr,
				"--elements-per-block must be a positive multiple of 32\n"),
			-1);
	return (0);
}

static int	record_matches(const export_opts_t *o,
				const membrane_kv_header_t *h)
{
	if (h->dtype != KVDUMP_DTYPE_F16_GGML)
		return (0);
	if (o->layer >= 0 && (uint32_t)o->layer != h->layer)
		return (0);
	if (o->tensor >= 0 && (uint32_t)o->tensor != h->tensor_type)
		return (0);
	return (1);
}

static int	find_record(const export_opts_t *o, membrane_kv_header_t *h,
				uint8_t **payload)
{
	FILE				*f;
	membrane_status_t	st;

	f = fopen(o->input_path, "rb");
	if (f == NULL)
		return (fprintf(stderr, "cannot open %s\n", o->input_path), -1);
	st = membrane_kvdump_read_header(f, h);
	while (st == MEMBRANE_OK)
	{
		if (record_matches(o, h))
		{
			st = membrane_kvdump_read_payload(f, h, payload);
			return (fclose(f), st == MEMBRANE_OK ? 0 : -1);
		}
		st = membrane_kvdump_read_payload(f, h, payload);
		if (st != MEMBRANE_OK)
			break ;
		free(*payload);
		st = membrane_kvdump_read_header(f, h);
	}
	fclose(f);
	if (st != MEMBRANE_OK && st != MEMBRANE_ERR_NOT_FOUND)
	{
		fprintf(stderr, "error reading %s: status=%d (truncated or "
			"corrupt file?)\n", o->input_path, (int)st);
		return (-1);
	}
	if (o->layer >= 0)
		fprintf(stderr, "no F16 record matching layer=%ld tensor=%d "
			"found in %s\n", o->layer, o->tensor, o->input_path);
	else
		fprintf(stderr, "no F16 K/V record found in %s\n", o->input_path);
	return (-1);
}

static void	build_label(char *out, size_t out_cap, const export_opts_t *o,
				const membrane_kv_header_t *h)
{
	if (o->label_override != NULL)
	{
		snprintf(out, out_cap, "%s", o->label_override);
		return ;
	}
	snprintf(out, out_cap, "model=%s layer=%u tensor=%s tokens=%u",
		h->model, h->layer, h->tensor_type == MEMBRANE_KV_TENSOR_K ? "K" : "V",
		h->token_end - h->token_start);
}

int	main(int argc, char **argv)
{
	export_opts_t			o;
	membrane_kv_header_t	h;
	uint8_t					*payload;
	FILE					*out_f;
	uint64_t				total_elems;
	uint64_t				block_count;
	char					label[192];
	membrane_status_t		st;
	int						rc;

	rc = parse_opts(argc, argv, &o);
	if (rc > 0)
		return (0);
	if (rc < 0)
		return (usage(stderr), 2);
	if (find_record(&o, &h, &payload) != 0)
		return (1);
	if (h.payload_size % sizeof(uint16_t) != 0)
	{
		fprintf(stderr, "record payload is not a whole number of F16 "
			"elements (%llu bytes) -- corrupt capture?\n",
			(unsigned long long)h.payload_size);
		return (free(payload), 1);
	}
	total_elems = h.payload_size / sizeof(uint16_t);
	block_count = total_elems / o.elements_per_block;
	if (block_count == 0)
	{
		fprintf(stderr, "record has only %llu F16 elements, fewer than "
			"one %u-element block\n", (unsigned long long)total_elems,
			o.elements_per_block);
		return (free(payload), 1);
	}
	if (total_elems % o.elements_per_block != 0)
		fprintf(stderr, "note: %llu trailing element(s) dropped (not a "
			"whole multiple of --elements-per-block %u)\n",
			(unsigned long long)(total_elems % o.elements_per_block),
			o.elements_per_block);
	build_label(label, sizeof(label), &o, &h);
	fprintf(stderr, "selected layer=%u tensor=%s: %llu blocks of %u "
		"elements (%s)\n", h.layer,
		h.tensor_type == MEMBRANE_KV_TENSOR_K ? "K" : "V",
		(unsigned long long)block_count, o.elements_per_block, label);
	out_f = fopen(o.output_path, "wb");
	if (out_f == NULL)
	{
		fprintf(stderr, "cannot create %s\n", o.output_path);
		return (free(payload), 1);
	}
	/* payload is the exact bytes captured from a live x86_64 process
	 * (llama.cpp), so reinterpreting it as an array of host-native
	 * uint16_t is correct on every platform this project targets
	 * (little-endian); membrane_trace_write re-encodes explicitly to
	 * the wire's little-endian format regardless. */
	st = membrane_trace_write(out_f, MEMBRANE_TRACE_DTYPE_F16,
			o.elements_per_block, block_count, (const uint16_t *)payload,
			label, (uint64_t)time(NULL));
	free(payload);
	if (st != MEMBRANE_OK)
	{
		fclose(out_f);
		fprintf(stderr, "write failed: status=%d\n", (int)st);
		return (1);
	}
	if (fclose(out_f) != 0)
	{
		fprintf(stderr, "write failed: error flushing %s\n",
			o.output_path);
		return (1);
	}
	fprintf(stderr, "wrote %s\n", o.output_path);
	return (0);
}
