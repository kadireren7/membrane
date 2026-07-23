/* CLOCK_MONOTONIC_RAW is a Linux extension, hidden under strict C11. */
#define _DEFAULT_SOURCE

#include <time.h>

#include "membrane/stats.h"

uint64_t	membrane_now_ns(void)
{
	struct timespec	ts;

	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

void	membrane_bench_result_print_json(const membrane_bench_result_t *r,
			const char *codec_name, size_t block_size, int iterations,
			FILE *out)
{
	fprintf(out,
		"{\"codec\":\"%s\",\"block_size\":%zu,\"iterations\":%d,"
		"\"original_bytes\":%zu,\"compressed_bytes\":%zu,"
		"\"compression_ratio\":%.4f,"
		"\"compress_gbps\":%.4f,\"decompress_gbps\":%.4f,"
		"\"integrity\":\"%s\"}\n",
		codec_name, block_size, iterations,
		r->original_bytes, r->compressed_bytes,
		r->compression_ratio,
		r->compress_gbps, r->decompress_gbps,
		r->integrity_ok ? "PASS" : "FAIL");
}
