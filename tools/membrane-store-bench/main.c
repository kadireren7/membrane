#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/backend.h"
#include "membrane/store.h"
#include "membrane/stats.h"

typedef struct s_bench_opts
{
	size_t				logical_size;
	size_t				memory_budget;
	size_t				block_size;
	double				compressible_ratio;
	membrane_codec_t	codec;
	const char			*codec_name;
	const char			*backend_kind;
	const char			*backend_path;
	unsigned int		seed;
}	bench_opts_t;

typedef struct s_bench_result
{
	membrane_store_stats_t	stats;
	double					ingest_gbps;
	uint64_t				verified_blocks;
	uint64_t				evicted_blocks;
	int						integrity_ok;
}	bench_result_t;

static uint32_t	rng_next(uint32_t *state)
{
	uint32_t	x;

	x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return (x);
}

/* Even split: exactly floor(nblocks*ratio) blocks are compressible. */
static int	block_is_compressible(uint64_t i, uint64_t nblocks, double ratio)
{
	uint64_t	c;

	c = (uint64_t)((double)nblocks * ratio);
	if (nblocks == 0)
		return (0);
	return ((i + 1) * c / nblocks > i * c / nblocks);
}

static void	gen_block(const bench_opts_t *o, uint64_t id, uint64_t nblocks,
				uint8_t *buf, size_t len)
{
	uint32_t	state;
	size_t		i;

	if (block_is_compressible(id, nblocks, o->compressible_ratio))
	{
		memset(buf, (int)(uint8_t)(id + 1), len);
		return ;
	}
	state = o->seed ^ (uint32_t)id ^ (uint32_t)(id >> 32);
	if (state == 0)
		state = 1;
	i = 0;
	while (i < len)
	{
		buf[i] = (uint8_t)rng_next(&state);
		i++;
	}
}

static size_t	block_len(const bench_opts_t *o, uint64_t id, uint64_t nblocks)
{
	size_t	off;

	if (id + 1 < nblocks)
		return (o->block_size);
	off = (size_t)id * o->block_size;
	return (o->logical_size - off);
}

static membrane_status_t	run_ingest(membrane_store_t *s,
								const bench_opts_t *o, uint64_t nblocks,
								bench_result_t *r)
{
	uint8_t				*buf;
	uint64_t			id;
	uint64_t			t0;
	membrane_status_t	status;

	buf = malloc(o->block_size);
	if (buf == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	id = 0;
	t0 = membrane_now_ns();
	while (id < nblocks)
	{
		gen_block(o, id, nblocks, buf, block_len(o, id, nblocks));
		status = membrane_store_put(s, id, buf, block_len(o, id, nblocks));
		if (status != MEMBRANE_OK)
			return (free(buf), status);
		id++;
	}
	r->ingest_gbps = (double)o->logical_size / (double)(membrane_now_ns() - t0);
	return (free(buf), MEMBRANE_OK);
}

static int	verify_block(membrane_store_t *s, const bench_opts_t *o,
				uint64_t id, uint64_t nblocks)
{
	uint8_t	*want;
	uint8_t	*got;
	size_t	len;
	int		ok;

	len = block_len(o, id, nblocks);
	want = malloc(len);
	got = malloc(len);
	if (want == NULL || got == NULL)
		return (free(want), free(got), -1);
	gen_block(o, id, nblocks, want, len);
	ok = membrane_store_get(s, id, got, len, &len);
	if (ok == MEMBRANE_ERR_NOT_FOUND)
		return (free(want), free(got), 1);
	if (ok != MEMBRANE_OK || memcmp(want, got, len) != 0)
		return (free(want), free(got), -1);
	return (free(want), free(got), 0);
}

/* Visits every block exactly once but in a scrambled order (an odd stride
 * is coprime to any block count), so read order differs from write order. */
static void	run_verify(membrane_store_t *s, const bench_opts_t *o,
				uint64_t nblocks, bench_result_t *r)
{
	uint64_t	k;
	uint64_t	id;
	int			rc;

	r->integrity_ok = 1;
	r->verified_blocks = 0;
	r->evicted_blocks = 0;
	k = 0;
	while (k < nblocks)
	{
		id = (k * 2654435761u + 12289u) % nblocks;
		rc = verify_block(s, o, id, nblocks);
		if (rc == 1)
			r->evicted_blocks += 1;
		else if (rc == 0)
			r->verified_blocks += 1;
		else
			r->integrity_ok = 0;
		k++;
	}
}

static void	format_bytes(size_t bytes, char *out, size_t cap)
{
	if (bytes >= (size_t)1 << 30)
		snprintf(out, cap, "%.2f GiB", (double)bytes / (1 << 30));
	else if (bytes >= (size_t)1 << 20)
		snprintf(out, cap, "%.2f MiB", (double)bytes / (1 << 20));
	else if (bytes >= (size_t)1 << 10)
		snprintf(out, cap, "%.2f KiB", (double)bytes / (1 << 10));
	else
		snprintf(out, cap, "%zu B", bytes);
}

static void	print_json(const bench_opts_t *o, const bench_result_t *r)
{
	const membrane_store_stats_t	*st;

	st = &r->stats;
	printf("{\"codec\":\"%s\",\"block_size\":%zu,\"compressible_ratio\":%.3f,"
		"\"seed\":%u,\"logical_bytes\":%llu,\"resident_bytes\":%zu,"
		"\"stored_bytes\":%llu,\"budget_bytes\":%zu,\"block_count\":%llu,"
		"\"evictions\":%llu,\"raw_blocks\":%llu,\"compressed_blocks\":%llu,"
		"\"effective_capacity_ratio\":%.4f,\"ingest_gbps\":%.4f,"
		"\"verified_blocks\":%llu,\"evicted_blocks\":%llu,"
		"\"integrity\":\"%s\"}\n",
		o->codec_name, o->block_size, o->compressible_ratio, o->seed,
		(unsigned long long)st->logical_bytes, st->resident_bytes,
		(unsigned long long)st->stored_bytes, st->budget_bytes,
		(unsigned long long)st->block_count,
		(unsigned long long)st->evictions,
		(unsigned long long)st->raw_blocks,
		(unsigned long long)st->compressed_blocks,
		st->effective_capacity_ratio, r->ingest_gbps,
		(unsigned long long)r->verified_blocks,
		(unsigned long long)r->evicted_blocks,
		r->integrity_ok ? "PASS" : "FAIL");
}

static void	print_human(const bench_result_t *r)
{
	const membrane_store_stats_t	*st;
	char							a[32];
	char							b[32];
	char							c[32];

	st = &r->stats;
	format_bytes(st->logical_bytes, a, sizeof(a));
	format_bytes(st->resident_bytes, b, sizeof(b));
	format_bytes(st->budget_bytes, c, sizeof(c));
	fprintf(stderr, "Logical data:        %s\n", a);
	fprintf(stderr, "Resident (budgeted): %s\n", b);
	fprintf(stderr, "Memory budget:       %s\n", c);
	fprintf(stderr, "Blocks:              %llu (%llu raw, %llu compressed)\n",
		(unsigned long long)st->block_count,
		(unsigned long long)st->raw_blocks,
		(unsigned long long)st->compressed_blocks);
	fprintf(stderr, "Evictions:           %llu\n",
		(unsigned long long)st->evictions);
	fprintf(stderr, "Effective capacity:  %.2fx\n",
		st->effective_capacity_ratio);
	fprintf(stderr, "Ingest throughput:   %.2f GB/s\n", r->ingest_gbps);
	fprintf(stderr, "Integrity:           %s\n",
		r->integrity_ok ? "PASS" : "FAIL");
}

static size_t	parse_size(const char *s)
{
	char				*end;
	unsigned long long	v;

	v = strtoull(s, &end, 10);
	if (*end == 'k' || *end == 'K')
		v *= 1024ull;
	else if (*end == 'm' || *end == 'M')
		v *= 1024ull * 1024;
	else if (*end == 'g' || *end == 'G')
		v *= 1024ull * 1024 * 1024;
	return ((size_t)v);
}

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-store-bench [options]\n"
		"  --logical-size N        total logical data (K/M/G suffix ok)\n"
		"  --memory-budget N       resident byte budget\n"
		"  --block-size N          block size (default 64K)\n"
		"  --compressible-ratio F  fraction of blocks that compress (0..1)\n"
		"  --codec NAME            raw or rle (default rle)\n"
		"  --backend KIND          none or file (default none)\n"
		"  --backend-path DIR      directory for the file backend\n"
		"  --seed N                PRNG seed (default 1)\n");
}

static int	opt_apply(bench_opts_t *o, int c)
{
	if (c == 'l')
		o->logical_size = parse_size(optarg);
	else if (c == 'm')
		o->memory_budget = parse_size(optarg);
	else if (c == 'b')
		o->block_size = parse_size(optarg);
	else if (c == 'r')
		o->compressible_ratio = strtod(optarg, NULL);
	else if (c == 's')
		o->seed = (unsigned int)strtoul(optarg, NULL, 10);
	else if (c == 'c')
	{
		if (!membrane_codec_from_name(optarg, &o->codec))
			return (-1);
		o->codec_name = optarg;
	}
	else if (c == 'k')
		o->backend_kind = optarg;
	else if (c == 'p')
		o->backend_path = optarg;
	else
		return (-1);
	return (0);
}

static void	opts_defaults(bench_opts_t *o)
{
	o->logical_size = 0;
	o->memory_budget = 0;
	o->block_size = 64 * 1024;
	o->compressible_ratio = 0.5;
	o->codec = MEMBRANE_CODEC_RLE;
	o->codec_name = "rle";
	o->backend_kind = NULL;
	o->backend_path = NULL;
	o->seed = 1;
}

static int	parse_opts(int argc, char **argv, bench_opts_t *o)
{
	static struct option	lo[] = {
	{"logical-size", required_argument, 0, 'l'},
	{"memory-budget", required_argument, 0, 'm'},
	{"block-size", required_argument, 0, 'b'},
	{"compressible-ratio", required_argument, 0, 'r'},
	{"codec", required_argument, 0, 'c'},
	{"backend", required_argument, 0, 'k'},
	{"backend-path", required_argument, 0, 'p'},
	{"seed", required_argument, 0, 's'},
	{0, 0, 0, 0}};
	int						c;

	opts_defaults(o);
	c = getopt_long(argc, argv, "l:m:b:r:c:k:p:s:", lo, NULL);
	while (c != -1)
	{
		if (opt_apply(o, c) != 0)
			return (-1);
		c = getopt_long(argc, argv, "l:m:b:r:c:k:p:s:", lo, NULL);
	}
	return (0);
}

static int	validate_opts(const bench_opts_t *o)
{
	if (o->logical_size == 0 || o->block_size == 0)
	{
		fprintf(stderr, "membrane-store-bench: "
			"--logical-size and --block-size must be > 0\n");
		return (-1);
	}
	if (o->compressible_ratio < 0.0 || o->compressible_ratio > 1.0)
	{
		fprintf(stderr, "membrane-store-bench: "
			"--compressible-ratio must be in [0, 1]\n");
		return (-1);
	}
	return (0);
}

static membrane_backend_t	*make_backend(const bench_opts_t *o)
{
	if (o->backend_kind == NULL || strcmp(o->backend_kind, "none") == 0)
		return (NULL);
	if (strcmp(o->backend_kind, "file") != 0 || o->backend_path == NULL)
		return (NULL);
	return (membrane_backend_file_create(o->backend_path));
}

static membrane_store_t	*make_store(const bench_opts_t *o,
							membrane_backend_t *be)
{
	membrane_store_config_t	cfg;

	cfg.budget_bytes = o->memory_budget;
	cfg.default_codec = o->codec;
	cfg.index_capacity = 0;
	cfg.backend = be;
	return (membrane_store_create(&cfg));
}

static int	backend_requested(const bench_opts_t *o)
{
	return (o->backend_kind != NULL && strcmp(o->backend_kind, "none") != 0);
}

static int	run_all(const bench_opts_t *o, membrane_store_t *store,
				bench_result_t *r)
{
	uint64_t	nblocks;

	nblocks = (o->logical_size + o->block_size - 1) / o->block_size;
	memset(r, 0, sizeof(*r));
	if (run_ingest(store, o, nblocks, r) != MEMBRANE_OK)
	{
		fprintf(stderr, "ingest failed (budget too small?)\n");
		return (-1);
	}
	run_verify(store, o, nblocks, r);
	membrane_store_get_stats(store, &r->stats);
	return (0);
}

int	main(int argc, char **argv)
{
	bench_opts_t		o;
	bench_result_t		r;
	membrane_store_t	*store;
	membrane_backend_t	*be;
	int					rc;

	if (parse_opts(argc, argv, &o) != 0 || validate_opts(&o) != 0)
		return (usage(stderr), 2);
	be = make_backend(&o);
	if (be == NULL && backend_requested(&o))
		return (fprintf(stderr, "backend create failed\n"), 2);
	store = make_store(&o, be);
	if (store == NULL)
		return (membrane_backend_destroy(be),
			fprintf(stderr, "store create failed\n"), 2);
	rc = run_all(&o, store, &r);
	if (rc == 0)
		(print_human(&r), print_json(&o, &r));
	membrane_store_destroy(store);
	membrane_backend_destroy(be);
	if (rc != 0)
		return (1);
	return (r.integrity_ok ? 0 : 1);
}
