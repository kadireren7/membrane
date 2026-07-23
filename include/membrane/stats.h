#ifndef MEMBRANE_STATS_H
#define MEMBRANE_STATS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Nanosecond timestamp from CLOCK_MONOTONIC_RAW. */
uint64_t membrane_now_ns(void);

typedef struct s_membrane_bench_result
{
    size_t  original_bytes;
    size_t  compressed_bytes;
    double  compression_ratio;   /* original_bytes / compressed_bytes */
    double  compress_gbps;
    double  decompress_gbps;
    int     integrity_ok;        /* 1 = PASS, 0 = FAIL */
}   membrane_bench_result_t;

/* Serializes `r` as a single-line JSON object to `out`. */
void membrane_bench_result_print_json(const membrane_bench_result_t *r,
                                       const char *codec_name,
                                       size_t block_size,
                                       int iterations,
                                       FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* MEMBRANE_STATS_H */
