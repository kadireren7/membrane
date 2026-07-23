#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "membrane/stats.h"

typedef struct s_bench_opts
{
    const char          *input_path;
    membrane_codec_t    codec;
    const char          *codec_name;
    size_t              block_size;
    int                 iterations;
}   bench_opts_t;

static void usage(FILE *out)
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

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    static struct option long_opts[] = {
        {"input",      required_argument, 0, 'i'},
        {"codec",      required_argument, 0, 'c'},
        {"block-size", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'n'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    int c;

    opts->input_path = NULL;
    opts->codec = MEMBRANE_CODEC_RLE;
    opts->codec_name = "rle";
    opts->block_size = 4096;
    opts->iterations = 10;
    while ((c = getopt_long(argc, argv, "i:c:b:n:h", long_opts, NULL)) != -1)
    {
        switch (c)
        {
            case 'i':
                opts->input_path = optarg;
                break;
            case 'c':
                if (!membrane_codec_from_name(optarg, &opts->codec))
                {
                    fprintf(stderr, "membrane-bench: unknown codec '%s'\n", optarg);
                    return -1;
                }
                opts->codec_name = optarg;
                break;
            case 'b':
            {
                long long v = atoll(optarg);

                if (v <= 0)
                {
                    fprintf(stderr, "membrane-bench: invalid block size '%s'\n", optarg);
                    return -1;
                }
                opts->block_size = (size_t)v;
                break;
            }
            case 'n':
                opts->iterations = atoi(optarg);
                if (opts->iterations < 1)
                {
                    fprintf(stderr, "membrane-bench: invalid iterations '%s'\n", optarg);
                    return -1;
                }
                break;
            case 'h':
                usage(stdout);
                exit(0);
            default:
                usage(stderr);
                return -1;
        }
    }
    if (opts->input_path == NULL)
    {
        fprintf(stderr, "membrane-bench: --input is required\n");
        usage(stderr);
        return -1;
    }
    return 0;
}

static uint8_t *read_whole_file(const char *path, size_t *out_size)
{
    FILE    *f;
    long    end;
    size_t  size;
    uint8_t *buf;

    f = fopen(path, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "membrane-bench: cannot open '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0)
    {
        fprintf(stderr, "membrane-bench: cannot size '%s'\n", path);
        fclose(f);
        return NULL;
    }
    size = (size_t)end;
    if (size == 0)
    {
        fprintf(stderr, "membrane-bench: '%s' is empty\n", path);
        fclose(f);
        return NULL;
    }
    rewind(f);
    buf = malloc(size);
    if (buf == NULL)
    {
        fprintf(stderr, "membrane-bench: out of memory reading '%s'\n", path);
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, size, f) != size)
    {
        fprintf(stderr, "membrane-bench: short read on '%s'\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = size;
    return buf;
}

static void format_bytes(size_t bytes, char *out, size_t out_cap)
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

int main(int argc, char **argv)
{
    bench_opts_t            opts;
    membrane_bench_result_t result;
    uint8_t                 *input;
    uint8_t                 *scratch;
    size_t                  input_size;
    size_t                  nblocks;
    size_t                  compressed_total = 0;
    uint64_t                compress_ns = 0;
    uint64_t                decompress_ns = 0;
    int                     integrity_ok = 1;

    if (parse_opts(argc, argv, &opts) != 0)
        return 2;
    input = read_whole_file(opts.input_path, &input_size);
    if (input == NULL)
        return 2;
    scratch = malloc(opts.block_size);
    if (scratch == NULL)
    {
        fprintf(stderr, "membrane-bench: out of memory\n");
        free(input);
        return 2;
    }
    nblocks = (input_size + opts.block_size - 1) / opts.block_size;
    for (int iter = 0; iter < opts.iterations && integrity_ok; iter++)
    {
        for (size_t b = 0; b < nblocks; b++)
        {
            size_t              off = b * opts.block_size;
            size_t              len = input_size - off < opts.block_size
                                        ? input_size - off : opts.block_size;
            membrane_block_t    *block;
            size_t              got;
            uint64_t            t0;
            uint64_t            t1;
            membrane_status_t   status;

            block = membrane_block_create(b, opts.codec);
            if (block == NULL)
            {
                fprintf(stderr, "membrane-bench: block allocation failed\n");
                integrity_ok = 0;
                break;
            }
            t0 = membrane_now_ns();
            status = membrane_block_write(block, input + off, len);
            t1 = membrane_now_ns();
            compress_ns += t1 - t0;
            if (status != MEMBRANE_OK)
            {
                fprintf(stderr, "membrane-bench: write failed on block %zu (status %d)\n",
                        b, status);
                integrity_ok = 0;
                membrane_block_destroy(block);
                break;
            }
            if (iter == 0)
                compressed_total += block->stored_size;
            t0 = membrane_now_ns();
            status = membrane_block_read(block, scratch, opts.block_size, &got);
            t1 = membrane_now_ns();
            decompress_ns += t1 - t0;
            if (status != MEMBRANE_OK || got != len
                || memcmp(scratch, input + off, len) != 0)
            {
                fprintf(stderr, "membrane-bench: integrity failure on block %zu\n", b);
                integrity_ok = 0;
                membrane_block_destroy(block);
                break;
            }
            membrane_block_destroy(block);
        }
    }
    result.original_bytes = input_size;
    result.compressed_bytes = compressed_total;
    result.compression_ratio = compressed_total > 0
        ? (double)input_size / (double)compressed_total : 0.0;
    result.compress_gbps = compress_ns > 0
        ? (double)input_size * opts.iterations / (double)compress_ns : 0.0;
    result.decompress_gbps = decompress_ns > 0
        ? (double)input_size * opts.iterations / (double)decompress_ns : 0.0;
    result.integrity_ok = integrity_ok;

    {
        char orig_str[32];
        char comp_str[32];
        char bs_str[32];

        format_bytes(input_size, orig_str, sizeof(orig_str));
        format_bytes(compressed_total, comp_str, sizeof(comp_str));
        format_bytes(opts.block_size, bs_str, sizeof(bs_str));
        fprintf(stderr,
                "Input:               %s\n"
                "Block size:          %s\n"
                "Codec:               %s\n"
                "Stored size:         %s\n"
                "Compression ratio:   %.2fx\n"
                "Compression speed:   %.2f GB/s\n"
                "Decompression speed: %.2f GB/s\n"
                "Integrity:           %s\n",
                orig_str, bs_str, opts.codec_name, comp_str,
                result.compression_ratio,
                result.compress_gbps, result.decompress_gbps,
                integrity_ok ? "PASS" : "FAIL");
    }
    membrane_bench_result_print_json(&result, opts.codec_name,
                                     opts.block_size, opts.iterations, stdout);
    free(scratch);
    free(input);
    return integrity_ok ? 0 : 1;
}
