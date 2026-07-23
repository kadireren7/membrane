#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "membrane/stats.h"

typedef struct s_bench_opts
{
	const char			*input_path;
	const char			*codec_name;
	membrane_codec_t	codec;
	size_t				block_size;
	int					iterations;
}	bench_opts_t;

typedef struct s_bench_ctx
{
	bench_opts_t	opts;
	uint8_t			*input;
	uint8_t			*scratch;
	size_t			input_size;
	size_t			compressed_total;
	uint64_t		compress_ns;
	uint64_t		decompress_ns;
	int				integrity_ok;
}	bench_ctx_t;

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-bench --input FILE [options]\n"
		"\n"
		"Options:\n"
		"  -i, --input FILE       input file to benchmark (required)\n"
		"  -c, --codec NAME       codec: raw, rle (default: rle)\n"
		"  -b, --block-size N     block size in bytes (default: 4096)\n"
		"  -n, --iterations N     benchmark passes (default: 10)\n"
		"  -h, --help             show this help\n"
		"\n"
		"A human-readable summary goes to stderr; a single JSON result\n"
		"line goes to stdout. Exit code is 0 only if integrity is PASS.\n");
}

static int	err_arg(const char *what, const char *arg)
{
	fprintf(stderr, "membrane-bench: %s '%s'\n", what, arg);
	return (-1);
}

static int	opt_apply_num(bench_opts_t *o, int c)
{
	long long	v;

	v = atoll(optarg);
	if (v <= 0)
		return (err_arg("invalid value", optarg));
	if (c == 'b')
		o->block_size = (size_t)v;
	else
		o->iterations = (int)v;
	return (0);
}

static int	opt_apply(bench_opts_t *o, int c)
{
	if (c == 'i')
		o->input_path = optarg;
	else if (c == 'c')
	{
		if (!membrane_codec_from_name(optarg, &o->codec))
			return (err_arg("unknown codec", optarg));
		o->codec_name = optarg;
	}
	else if (c == 'b' || c == 'n')
		return (opt_apply_num(o, c));
	else if (c == 'h')
	{
		usage(stdout);
		exit(0);
	}
	else
		return (-1);
	return (0);
}

static int	parse_opts(int argc, char **argv, bench_opts_t *o)
{
	static struct option	long_opts[] = {
	{"input", required_argument, 0, 'i'},
	{"codec", required_argument, 0, 'c'},
	{"block-size", required_argument, 0, 'b'},
	{"iterations", required_argument, 0, 'n'},
	{"help", no_argument, 0, 'h'},
	{0, 0, 0, 0}};
	int						c;

	o->input_path = NULL;
	o->codec = MEMBRANE_CODEC_RLE;
	o->codec_name = "rle";
	o->block_size = 4096;
	o->iterations = 10;
	c = getopt_long(argc, argv, "i:c:b:n:h", long_opts, NULL);
	while (c != -1)
	{
		if (opt_apply(o, c) != 0)
			return (-1);
		c = getopt_long(argc, argv, "i:c:b:n:h", long_opts, NULL);
	}
	if (o->input_path != NULL)
		return (0);
	usage(stderr);
	return (err_arg("missing required option", "--input"));
}

static uint8_t	*read_file_buf(FILE *f, size_t size, const char *path)
{
	uint8_t	*buf;

	buf = malloc(size);
	if (buf == NULL)
	{
		err_arg("out of memory reading", path);
		return (NULL);
	}
	if (fread(buf, 1, size, f) != size)
	{
		err_arg("short read on", path);
		free(buf);
		return (NULL);
	}
	return (buf);
}

static uint8_t	*read_whole_file(const char *path, size_t *out_size)
{
	FILE	*f;
	long	end;
	uint8_t	*buf;

	f = fopen(path, "rb");
	if (f == NULL)
	{
		err_arg("cannot open", path);
		return (NULL);
	}
	end = -1;
	if (fseek(f, 0, SEEK_END) == 0)
		end = ftell(f);
	if (end <= 0)
	{
		err_arg("cannot read or empty file", path);
		fclose(f);
		return (NULL);
	}
	rewind(f);
	*out_size = (size_t)end;
	buf = read_file_buf(f, *out_size, path);
	fclose(f);
	return (buf);
}

static void	format_bytes(size_t bytes, char *out, size_t out_cap)
{
	if (bytes >= (size_t)1 << 30)
		snprintf(out, out_cap, "%.2f GiB", (double)bytes / (1 << 30));
	else if (bytes >= (size_t)1 << 20)
		snprintf(out, out_cap, "%.2f MiB", (double)bytes / (1 << 20));
	else if (bytes >= (size_t)1 << 10)
		snprintf(out, out_cap, "%.2f KiB", (double)bytes / (1 << 10));
	else
		snprintf(out, out_cap, "%zu B", bytes);
}

static int	bench_verify(bench_ctx_t *ctx, membrane_block_t *block,
				size_t off, size_t len)
{
	size_t		got;
	uint64_t	t0;

	t0 = membrane_now_ns();
	if (membrane_block_read(block, ctx->scratch, len, &got) != MEMBRANE_OK)
		return (-1);
	ctx->decompress_ns += membrane_now_ns() - t0;
	if (got != len || memcmp(ctx->scratch, ctx->input + off, len) != 0)
		return (-1);
	return (0);
}

static int	bench_block(bench_ctx_t *ctx, size_t b, int iter)
{
	membrane_block_t	*block;
	size_t				off;
	size_t				len;
	uint64_t			t0;
	int					rc;

	off = b * ctx->opts.block_size;
	len = ctx->input_size - off;
	if (len > ctx->opts.block_size)
		len = ctx->opts.block_size;
	block = membrane_block_create(b, ctx->opts.codec);
	if (block == NULL)
		return (-1);
	rc = -1;
	t0 = membrane_now_ns();
	if (membrane_block_write(block, ctx->input + off, len) == MEMBRANE_OK)
	{
		ctx->compress_ns += membrane_now_ns() - t0;
		if (iter == 0)
			ctx->compressed_total += block->stored_size;
		rc = bench_verify(ctx, block, off, len);
	}
	membrane_block_destroy(block);
	return (rc);
}

static int	bench_run(bench_ctx_t *ctx)
{
	size_t	nblocks;
	size_t	b;
	int		iter;

	nblocks = (ctx->input_size + ctx->opts.block_size - 1)
		/ ctx->opts.block_size;
	iter = 0;
	while (iter < ctx->opts.iterations)
	{
		b = 0;
		while (b < nblocks)
		{
			if (bench_block(ctx, b, iter) != 0)
			{
				fprintf(stderr,
					"membrane-bench: integrity failure on block %zu\n", b);
				ctx->integrity_ok = 0;
				return (-1);
			}
			b++;
		}
		iter++;
	}
	return (0);
}

static void	fill_result(const bench_ctx_t *ctx, membrane_bench_result_t *r)
{
	double	total;

	total = (double)ctx->input_size * (double)ctx->opts.iterations;
	r->original_bytes = ctx->input_size;
	r->compressed_bytes = ctx->compressed_total;
	r->compression_ratio = 0.0;
	if (ctx->compressed_total > 0)
		r->compression_ratio = (double)ctx->input_size
			/ (double)ctx->compressed_total;
	r->compress_gbps = 0.0;
	if (ctx->compress_ns > 0)
		r->compress_gbps = total / (double)ctx->compress_ns;
	r->decompress_gbps = 0.0;
	if (ctx->decompress_ns > 0)
		r->decompress_gbps = total / (double)ctx->decompress_ns;
	r->integrity_ok = ctx->integrity_ok;
}

static void	print_summary(const bench_ctx_t *ctx,
				const membrane_bench_result_t *r)
{
	char	orig[32];
	char	comp[32];
	char	bs[32];

	format_bytes(ctx->input_size, orig, sizeof(orig));
	format_bytes(ctx->compressed_total, comp, sizeof(comp));
	format_bytes(ctx->opts.block_size, bs, sizeof(bs));
	fprintf(stderr, "Input:               %s\n", orig);
	fprintf(stderr, "Block size:          %s\n", bs);
	fprintf(stderr, "Codec:               %s\n", ctx->opts.codec_name);
	fprintf(stderr, "Stored size:         %s\n", comp);
	fprintf(stderr, "Compression ratio:   %.2fx\n", r->compression_ratio);
	fprintf(stderr, "Compression speed:   %.2f GB/s\n", r->compress_gbps);
	fprintf(stderr, "Decompression speed: %.2f GB/s\n", r->decompress_gbps);
	if (r->integrity_ok)
		fprintf(stderr, "Integrity:           PASS\n");
	else
		fprintf(stderr, "Integrity:           FAIL\n");
}

static int	ctx_setup(bench_ctx_t *ctx, int argc, char **argv)
{
	memset(ctx, 0, sizeof(*ctx));
	ctx->integrity_ok = 1;
	if (parse_opts(argc, argv, &ctx->opts) != 0)
		return (-1);
	ctx->input = read_whole_file(ctx->opts.input_path, &ctx->input_size);
	if (ctx->input == NULL)
		return (-1);
	ctx->scratch = malloc(ctx->opts.block_size);
	if (ctx->scratch == NULL)
	{
		free(ctx->input);
		return (-1);
	}
	return (0);
}

int	main(int argc, char **argv)
{
	bench_ctx_t				ctx;
	membrane_bench_result_t	result;

	if (ctx_setup(&ctx, argc, argv) != 0)
		return (2);
	bench_run(&ctx);
	fill_result(&ctx, &result);
	print_summary(&ctx, &result);
	membrane_bench_result_print_json(&result, ctx.opts.codec_name,
		ctx.opts.block_size, ctx.opts.iterations, stdout);
	free(ctx.scratch);
	free(ctx.input);
	if (ctx.integrity_ok)
		return (0);
	return (1);
}
