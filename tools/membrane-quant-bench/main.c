#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
# include <x86intrin.h>
# define QBENCH_HAVE_RDTSC 1
#endif

#include "membrane/quant_simd.h"
#include "membrane/f16convert.h"

/*
 * Phase 5.1 item 1 (profile first) / item 10 (benchmark matrix).
 *
 * No `perf` binary is available in this environment (checked via
 * `which perf`), so this tool cannot report real hardware performance
 * counters (retired instructions, IPC, branch misses, cache misses).
 * That gap is disclosed here and in docs/phase5-quant-engine.md rather
 * than faked. What it DOES measure, all directly and honestly:
 *
 *   - wall-clock time (CLOCK_MONOTONIC), the primary metric
 *   - process CPU time (CLOCK_PROCESS_CPUTIME_ID), a utilization sanity
 *     check against wall time (useful for spotting oversubscription in
 *     the parallel arms)
 *   - TSC tick deltas (RDTSC), calibrated against wall clock at
 *     startup -- an APPROXIMATION of cycle counts, not exact retired
 *     cycles (invariant TSC ticks at a fixed nominal rate regardless of
 *     turbo/power-state, so this under/over-counts relative to true
 *     core cycles under frequency scaling; labeled as such in output)
 *   - real allocation/free call counts across the timed region, via
 *     linker --wrap=malloc/free/calloc/realloc -- an actual measurement,
 *     not a code-inspection claim, of item 6's "measure malloc/free
 *     count" requirement
 *   - peak RSS (getrusage ru_maxrss) -- process-lifetime cumulative,
 *     not scoped to one timed region; disclosed as such
 *   - derived throughput: ns/block, GB/s (input bytes/sec), element/s
 */

extern void	*__real_malloc(size_t n);
extern void	__real_free(void *p);
extern void	*__real_calloc(size_t a, size_t b);
extern void	*__real_realloc(void *p, size_t n);

static volatile long	g_alloc_count = 0;
static volatile long	g_free_count = 0;

void	*__wrap_malloc(size_t n)
{
	g_alloc_count++;
	return (__real_malloc(n));
}

void	__wrap_free(void *p)
{
	if (p != NULL)
		g_free_count++;
	__real_free(p);
}

void	*__wrap_calloc(size_t a, size_t b)
{
	g_alloc_count++;
	return (__real_calloc(a, b));
}

void	*__wrap_realloc(void *p, size_t n)
{
	g_alloc_count++;
	return (__real_realloc(p, n));
}

static double	g_ns_per_tick = 0.0;

static void	calibrate_tsc(void)
{
#ifdef QBENCH_HAVE_RDTSC
	struct timespec	t0;
	struct timespec	t1;
	uint64_t		c0;
	uint64_t		c1;
	int64_t			elapsed_ns;

	clock_gettime(CLOCK_MONOTONIC, &t0);
	c0 = __rdtsc();
	t1 = t0;
	while ((t1.tv_sec - t0.tv_sec) * 1000000000LL
			+ (t1.tv_nsec - t0.tv_nsec) < 150000000LL)
		clock_gettime(CLOCK_MONOTONIC, &t1);
	c1 = __rdtsc();
	elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL
		+ (t1.tv_nsec - t0.tv_nsec);
	if (c1 > c0)
		g_ns_per_tick = (double)elapsed_ns / (double)(c1 - c0);
#endif
}

static uint64_t	now_ns(clockid_t clk)
{
	struct timespec	ts;

	clock_gettime(clk, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

static uint64_t	rdtsc_now(void)
{
#ifdef QBENCH_HAVE_RDTSC
	return (__rdtsc());
#else
	return (0);
#endif
}

typedef enum e_qbench_op
{
	QBENCH_Q8_QUANT = 0,
	QBENCH_Q8_DEQUANT,
	QBENCH_Q4_QUANT,
	QBENCH_Q4_DEQUANT,
	QBENCH_OP_COUNT
}	qbench_op_t;

static const char	*op_name(qbench_op_t op)
{
	if (op == QBENCH_Q8_QUANT)
		return ("q8_quantize");
	if (op == QBENCH_Q8_DEQUANT)
		return ("q8_dequantize");
	if (op == QBENCH_Q4_QUANT)
		return ("q4_quantize");
	return ("q4_dequantize");
}

typedef struct s_qbench_result
{
	uint64_t	wall_ns_min;
	uint64_t	wall_ns_mean;
	uint64_t	cpu_ns_min;
	uint64_t	tsc_ticks_min;
	int			threads_used;
	long		alloc_delta;
	long		free_delta;
}	qbench_result_t;

static size_t	packed_bytes(qbench_op_t op, size_t n_elems)
{
	size_t	n_blocks;

	n_blocks = n_elems / MEMBRANE_QSIMD_BLOCK_ELEMS;
	if (op == QBENCH_Q8_QUANT || op == QBENCH_Q8_DEQUANT)
		return (n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
	return (n_blocks * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES);
}

static void	run_op_once(qbench_op_t op, membrane_simd_backend_t backend,
	const uint16_t *x_f16, uint8_t *packed_q8, uint8_t *packed_q4,
	uint16_t *out_f16, size_t n_blocks, size_t elems_per_row,
	int n_threads, int *threads_used)
{
	membrane_status_t	st;

	if (op == QBENCH_Q8_QUANT)
		st = membrane_simd_q8_0_quantize_batch_parallel(backend, x_f16,
				n_blocks, elems_per_row, packed_q8, n_threads,
				threads_used);
	else if (op == QBENCH_Q8_DEQUANT)
		st = membrane_simd_q8_0_dequantize_batch_parallel(backend,
				packed_q8, n_blocks, elems_per_row, out_f16, n_threads,
				threads_used);
	else if (op == QBENCH_Q4_QUANT)
		st = membrane_simd_q4_0_quantize_batch_parallel(backend, x_f16,
				n_blocks, elems_per_row, packed_q4, n_threads,
				threads_used);
	else
		st = membrane_simd_q4_0_dequantize_batch_parallel(backend,
				packed_q4, n_blocks, elems_per_row, out_f16, n_threads,
				threads_used);
	if (st != MEMBRANE_OK)
	{
		fprintf(stderr, "qbench: op %s backend %s failed: status=%d\n",
			op_name(op), membrane_simd_backend_name(backend), (int)st);
		exit(1);
	}
}

static qbench_result_t	bench_one(qbench_op_t op,
	membrane_simd_backend_t backend, const uint16_t *x_f16,
	uint8_t *packed_q8, uint8_t *packed_q4, uint16_t *out_f16,
	size_t n_elems, size_t block_size, int n_threads, int repeats)
{
	qbench_result_t	r;
	size_t				n_blocks;
	int					i;
	uint64_t			w0;
	uint64_t			w1;
	uint64_t			c0;
	uint64_t			c1;
	uint64_t			t0;
	uint64_t			t1;
	uint64_t			wall_sum;
	int					used;

	memset(&r, 0, sizeof(r));
	r.wall_ns_min = UINT64_MAX;
	r.cpu_ns_min = UINT64_MAX;
	r.tsc_ticks_min = UINT64_MAX;
	n_blocks = n_elems / block_size;
	wall_sum = 0;
	i = 0;
	while (i < repeats)
	{
		g_alloc_count = 0;
		g_free_count = 0;
		w0 = now_ns(CLOCK_MONOTONIC);
		c0 = now_ns(CLOCK_PROCESS_CPUTIME_ID);
		t0 = rdtsc_now();
		used = 0;
		run_op_once(op, backend, x_f16, packed_q8, packed_q4, out_f16,
			n_blocks, block_size, n_threads, &used);
		t1 = rdtsc_now();
		c1 = now_ns(CLOCK_PROCESS_CPUTIME_ID);
		w1 = now_ns(CLOCK_MONOTONIC);
		if (w1 - w0 < r.wall_ns_min)
			r.wall_ns_min = w1 - w0;
		if (c1 - c0 < r.cpu_ns_min)
			r.cpu_ns_min = c1 - c0;
		if (t1 - t0 < r.tsc_ticks_min)
			r.tsc_ticks_min = t1 - t0;
		wall_sum += w1 - w0;
		r.threads_used = used;
		r.alloc_delta += g_alloc_count;
		r.free_delta += g_free_count;
		i++;
	}
	r.wall_ns_mean = wall_sum / (uint64_t)repeats;
	return (r);
}

static size_t	parse_csv_sizes(const char *s, size_t *out, size_t max)
{
	size_t	n;
	char	*end;
	long	v;

	n = 0;
	while (*s != '\0' && n < max)
	{
		v = strtol(s, &end, 10);
		if (end == s)
			break;
		out[n++] = (size_t)v;
		s = end;
		if (*s == ',')
			s++;
	}
	return (n);
}

static int	parse_csv_ints(const char *s, int *out, size_t max)
{
	int		n;
	char	*end;
	long	v;

	n = 0;
	while (*s != '\0' && (size_t)n < max)
	{
		v = strtol(s, &end, 10);
		if (end == s)
			break;
		out[n++] = (int)v;
		s = end;
		if (*s == ',')
			s++;
	}
	return (n);
}

static int	backend_probe_ok(membrane_simd_backend_t b)
{
	uint16_t	x[32];
	uint8_t		out[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
	size_t		i;

	i = 0;
	while (i < 32)
	{
		x[i] = (uint16_t)(0x3C00u + (uint16_t)i);
		i++;
	}
	return (membrane_simd_q8_0_quantize(b, x, 32, out) == MEMBRANE_OK);
}

int	main(int argc, char **argv)
{
	size_t				sizes[16];
	size_t				n_sizes;
	int					threads[16];
	int					n_threads_list;
	membrane_simd_backend_t	backends[MEMBRANE_SIMD_COUNT];
	int					n_backends;
	size_t				total_elems;
	int					repeats;
	const char			*csv_path;
	FILE				*csv;
	size_t				si;
	int					ti;
	int					bi;
	int					oi;
	uint16_t			*x_f16;
	uint8_t				*packed_q8;
	uint8_t				*packed_q4;
	uint16_t			*out_f16;
	struct rusage		ru;
	unsigned int		rng;

	total_elems = 16u * 1024u * 1024u;
	repeats = 3;
	csv_path = NULL;
	n_sizes = parse_csv_sizes("32,64,128,256,1024,4096", sizes, 16);
	n_threads_list = parse_csv_ints("1,2,4", threads, 16);
	if ((size_t)threads[n_threads_list - 1]
			!= membrane_simd_default_threads())
		threads[n_threads_list++] = (int)membrane_simd_default_threads();
	{
		int	i;

		i = 1;
		while (i < argc)
		{
			if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc)
				n_sizes = parse_csv_sizes(argv[++i], sizes, 16);
			else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
				n_threads_list = parse_csv_ints(argv[++i], threads, 16);
			else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
				repeats = atoi(argv[++i]);
			else if (strcmp(argv[i], "--total-elems") == 0 && i + 1 < argc)
				total_elems = (size_t)strtoul(argv[++i], NULL, 10);
			else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
				csv_path = argv[++i];
			i++;
		}
	}
	n_backends = 0;
	bi = 0;
	while (bi < MEMBRANE_SIMD_COUNT)
	{
		if (backend_probe_ok((membrane_simd_backend_t)bi))
			backends[n_backends++] = (membrane_simd_backend_t)bi;
		bi++;
	}
	calibrate_tsc();
	csv = NULL;
	if (csv_path != NULL)
	{
		csv = fopen(csv_path, "w");
		if (csv != NULL)
			fprintf(csv, "op,backend,block_size,threads_requested,"
				"threads_used,wall_ns_min,wall_ns_mean,cpu_ns_min,"
				"tsc_ticks_min,ns_per_tick,ns_per_block,gbytes_per_sec,"
				"elems_per_sec,alloc_calls,free_calls\n");
	}
	printf("MEMBRANE Phase 5.1 quant engine benchmark\n");
	printf("total_elems=%zu repeats=%d backends=%d threads_list=%d\n",
		total_elems, repeats, n_backends, n_threads_list);
	if (g_ns_per_tick > 0.0)
		printf("tsc calibration: %.6f ns/tick (~%.3f GHz nominal, "
			"invariant-TSC estimate, NOT hardware perf-counter cycles)\n",
			g_ns_per_tick, 1.0 / g_ns_per_tick);
	else
		printf("tsc calibration unavailable on this platform\n");
	printf("note: no `perf` binary available in this environment -- "
		"IPC, branch-misses, cache-misses are NOT measured here.\n\n");
	printf("%-14s %-8s %6s %5s/%-5s %12s %12s %10s %8s %6s %6s\n",
		"op", "backend", "block", "reqT", "usedT", "ns/block",
		"GB/s(in)", "elem/s", "allocs", "frees", "ticks/e");
	si = 0;
	while (si < n_sizes)
	{
		x_f16 = malloc(total_elems * sizeof(uint16_t));
		packed_q8 = malloc(packed_bytes(QBENCH_Q8_QUANT, total_elems));
		packed_q4 = malloc(packed_bytes(QBENCH_Q4_QUANT, total_elems));
		out_f16 = malloc(total_elems * sizeof(uint16_t));
		if (x_f16 == NULL || packed_q8 == NULL || packed_q4 == NULL
				|| out_f16 == NULL)
		{
			fprintf(stderr, "qbench: allocation failure for block=%zu\n",
				sizes[si]);
			return (1);
		}
		rng = 0xC0FFEEu + (unsigned int)sizes[si];
		{
			size_t	i;
			float	v;
			union { float f; uint32_t u; } cv;

			i = 0;
			while (i < total_elems)
			{
				rng = rng * 1103515245u + 12345u;
				v = ((float)((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f)
					* 4.0f;
				(void)v;
				cv.f = v;
				x_f16[i] = membrane_f32_to_f16(cv.f);
				i++;
			}
		}
		bi = 0;
		while (bi < n_backends)
		{
			membrane_simd_q8_0_quantize_batch(backends[bi], x_f16,
				total_elems / sizes[si], sizes[si], packed_q8, NULL, 0);
			membrane_simd_q4_0_quantize_batch(backends[bi], x_f16,
				total_elems / sizes[si], sizes[si], packed_q4, NULL, 0);
			ti = 0;
			while (ti < n_threads_list)
			{
				oi = 0;
				while (oi < QBENCH_OP_COUNT)
				{
					qbench_result_t	r;
					double				secs;
					double				gbps;
					double				eps;
					double				nspb;
					double				ticks_per_elem;
					size_t				n_blocks_this;

					r = bench_one((qbench_op_t)oi, backends[bi], x_f16,
							packed_q8, packed_q4, out_f16, total_elems,
							sizes[si], threads[ti], repeats);
					secs = (double)r.wall_ns_min / 1e9;
					n_blocks_this = total_elems / sizes[si];
					nspb = (double)r.wall_ns_min / (double)n_blocks_this;
					eps = (double)total_elems / secs;
					gbps = ((double)total_elems * 2.0) / secs / 1e9;
					ticks_per_elem = (double)r.tsc_ticks_min
						/ (double)total_elems;
					printf("%-14s %-8s %6zu %5d/%-5d %12.2f %12.3f "
						"%10.0f %8ld %6ld %6.2f\n",
						op_name((qbench_op_t)oi),
						membrane_simd_backend_name(backends[bi]),
						sizes[si], threads[ti], r.threads_used, nspb,
						gbps, eps, r.alloc_delta, r.free_delta,
						ticks_per_elem);
					if (csv != NULL)
						fprintf(csv, "%s,%s,%zu,%d,%d,%llu,%llu,%llu,"
							"%llu,%.6f,%.4f,%.4f,%.2f,%ld,%ld\n",
							op_name((qbench_op_t)oi),
							membrane_simd_backend_name(backends[bi]),
							sizes[si], threads[ti], r.threads_used,
							(unsigned long long)r.wall_ns_min,
							(unsigned long long)r.wall_ns_mean,
							(unsigned long long)r.cpu_ns_min,
							(unsigned long long)r.tsc_ticks_min,
							g_ns_per_tick, nspb, gbps, eps,
							r.alloc_delta, r.free_delta);
					oi++;
				}
				ti++;
			}
			bi++;
		}
		free(x_f16);
		free(packed_q8);
		free(packed_q4);
		free(out_f16);
		si++;
	}
	if (csv != NULL)
		fclose(csv);
	getrusage(RUSAGE_SELF, &ru);
	printf("\nprocess peak RSS (ru_maxrss, cumulative over whole run, "
		"not scoped per-measurement): %ld KB\n", ru.ru_maxrss);
	return (0);
}
