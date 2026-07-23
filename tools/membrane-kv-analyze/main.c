#define _DEFAULT_SOURCE

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "membrane/kvdump.h"
#include "membrane/kvmetrics.h"

# define MAX_META 8
# define N_BLOCK_SIZES 4

static const size_t	g_block_sizes[N_BLOCK_SIZES] = {
	4096, 16384, 65536, 262144
};

typedef struct s_kva_opts
{
	const char	*jsonl_path;
	const char	*csv_path;
	const char	*meta[MAX_META];
	int			meta_count;
	char		**inputs;
	int			input_count;
}	kva_opts_t;

/* Aggregates for the human K-vs-V summary at 64 KiB blocks. */
typedef struct s_kva_totals
{
	uint64_t	raw[2];
	uint64_t	rle[2];
	uint64_t	adaptive[2];
	int			integrity_ok;
}	kva_totals_t;

typedef struct s_kva_out
{
	FILE		*jsonl;
	FILE		*csv;
	kva_totals_t	totals;
}	kva_out_t;

static void	print_meta_json(FILE *f, const kva_opts_t *o)
{
	int	i;

	i = 0;
	while (i < o->meta_count)
	{
		const char	*eq = strchr(o->meta[i], '=');

		if (eq != NULL)
			fprintf(f, ",\"%.*s\":\"%s\"",
				(int)(eq - o->meta[i]), o->meta[i], eq + 1);
		i++;
	}
}

static void	cpu_model(char *out, size_t cap)
{
	FILE	*f;
	char	line[256];
	char	*colon;

	snprintf(out, cap, "unknown");
	f = fopen("/proc/cpuinfo", "r");
	if (f == NULL)
		return ;
	while (fgets(line, sizeof(line), f) != NULL)
	{
		if (strncmp(line, "model name", 10) == 0
			&& (colon = strchr(line, ':')) != NULL)
		{
			line[strcspn(line, "\n")] = '\0';
			snprintf(out, cap, "%s", colon + 2);
			break ;
		}
	}
	fclose(f);
}

static void	print_env(const kva_opts_t *o, kva_out_t *out)
{
	struct utsname	u;
	char			cpu[128];
	long			ram_mb;

	uname(&u);
	cpu_model(cpu, sizeof(cpu));
	ram_mb = sysconf(_SC_PHYS_PAGES) / 1024 * sysconf(_SC_PAGE_SIZE) / 1024;
	fprintf(out->jsonl,
		"{\"record\":\"env\",\"os\":\"%s %s\",\"cpu\":\"%s\","
		"\"ram_mb\":%ld,\"compiler\":\"%s\"", u.sysname, u.release, cpu,
		ram_mb, __VERSION__);
	print_meta_json(out->jsonl, o);
	fprintf(out->jsonl, "}\n");
	fprintf(stderr, "env: %s %s | %s | %ld MB RAM | gcc %s\n",
		u.sysname, u.release, cpu, ram_mb, __VERSION__);
}

static double	ratio_of(uint64_t raw, uint64_t stored)
{
	if (stored == 0)
		return (0.0);
	return ((double)raw / (double)stored);
}

static void	emit_jsonl_metrics(FILE *f, const membrane_kv_metrics_t *m)
{
	fprintf(f,
		"\"blocks\":%llu,\"raw_bytes\":%llu,"
		"\"rle_bytes\":%llu,\"rle_ratio\":%.4f,"
		"\"adaptive_bytes\":%llu,\"adaptive_ratio\":%.4f,"
		"\"adaptive_raw_blocks\":%llu,\"adaptive_rle_blocks\":%llu,"
		"\"zero_ratio\":%.6f,\"entropy_bits\":%.4f,"
		"\"total_runs\":%llu,\"max_run\":%llu,\"mean_run\":%.3f,"
		"\"integrity\":\"%s\"",
		(unsigned long long)m->blocks,
		(unsigned long long)m->raw_bytes,
		(unsigned long long)m->rle_bytes, ratio_of(m->raw_bytes, m->rle_bytes),
		(unsigned long long)m->adaptive_bytes,
		ratio_of(m->raw_bytes, m->adaptive_bytes),
		(unsigned long long)m->adaptive_raw_blocks,
		(unsigned long long)m->adaptive_rle_blocks,
		m->raw_bytes ? (double)m->zero_bytes / (double)m->raw_bytes : 0.0,
		m->entropy,
		(unsigned long long)m->total_runs, (unsigned long long)m->max_run,
		m->total_runs ? (double)m->raw_bytes / (double)m->total_runs : 0.0,
		m->integrity_ok ? "PASS" : "FAIL");
}

static void	emit_jsonl(kva_out_t *out, const kva_opts_t *o, const char *file,
				const membrane_kv_header_t *h, size_t bs,
				const membrane_kv_metrics_t *m)
{
	fprintf(out->jsonl,
		"{\"record\":\"metrics\",\"file\":\"%s\",\"model\":\"%s\","
		"\"layer\":%u,\"tensor\":\"%c\",\"token_start\":%u,\"token_end\":%u,"
		"\"dtype\":%u,\"block_size\":%zu,",
		file, h->model, h->layer, h->tensor_type ? 'V' : 'K',
		h->token_start, h->token_end, h->dtype, bs);
	emit_jsonl_metrics(out->jsonl, m);
	print_meta_json(out->jsonl, o);
	fprintf(out->jsonl, "}\n");
}

static void	emit_csv(kva_out_t *out, const char *file,
				const membrane_kv_header_t *h, size_t bs,
				const membrane_kv_metrics_t *m)
{
	fprintf(out->csv,
		"%s,%s,%u,%c,%u,%u,%u,%zu,%llu,%llu,%llu,%.4f,%llu,%.4f,"
		"%.6f,%.4f,%llu,%s\n",
		file, h->model, h->layer, h->tensor_type ? 'V' : 'K',
		h->token_start, h->token_end, h->dtype, bs,
		(unsigned long long)m->blocks,
		(unsigned long long)m->raw_bytes,
		(unsigned long long)m->rle_bytes, ratio_of(m->raw_bytes, m->rle_bytes),
		(unsigned long long)m->adaptive_bytes,
		ratio_of(m->raw_bytes, m->adaptive_bytes),
		m->raw_bytes ? (double)m->zero_bytes / (double)m->raw_bytes : 0.0,
		m->entropy, (unsigned long long)m->max_run,
		m->integrity_ok ? "PASS" : "FAIL");
}

static void	human_record(const membrane_kv_header_t *h,
				const membrane_kv_metrics_t *m)
{
	fprintf(stderr,
		"  layer %2u %c  tokens %u..%u  raw %7llu B  rle %.3fx  "
		"adaptive %.3fx  H %.3f  %s\n",
		h->layer, h->tensor_type ? 'V' : 'K', h->token_start, h->token_end,
		(unsigned long long)m->raw_bytes,
		ratio_of(m->raw_bytes, m->rle_bytes),
		ratio_of(m->raw_bytes, m->adaptive_bytes), m->entropy,
		m->integrity_ok ? "PASS" : "FAIL");
}

static void	totals_add(kva_totals_t *t, const membrane_kv_header_t *h,
				const membrane_kv_metrics_t *m)
{
	int	idx;

	idx = (h->tensor_type != 0);
	t->raw[idx] += m->raw_bytes;
	t->rle[idx] += m->rle_bytes;
	t->adaptive[idx] += m->adaptive_bytes;
	if (!m->integrity_ok)
		t->integrity_ok = 0;
}

static membrane_status_t	analyze_record(kva_out_t *out,
								const kva_opts_t *o, const char *file,
								const membrane_kv_header_t *h,
								const uint8_t *payload)
{
	membrane_kv_metrics_t	m;
	membrane_status_t		st;
	int						i;

	i = 0;
	while (i < N_BLOCK_SIZES)
	{
		st = membrane_kv_metrics_compute(payload, h->payload_size,
				g_block_sizes[i], &m);
		if (st != MEMBRANE_OK)
			return (st);
		emit_jsonl(out, o, file, h, g_block_sizes[i], &m);
		emit_csv(out, file, h, g_block_sizes[i], &m);
		if (g_block_sizes[i] == 65536)
		{
			human_record(h, &m);
			totals_add(&out->totals, h, &m);
		}
		i++;
	}
	return (MEMBRANE_OK);
}

static membrane_status_t	analyze_file(kva_out_t *out, const kva_opts_t *o,
								const char *path)
{
	FILE					*f;
	membrane_kv_header_t	h;
	uint8_t					*payload;
	membrane_status_t		st;

	f = fopen(path, "rb");
	if (f == NULL)
		return (fprintf(stderr, "cannot open %s\n", path), MEMBRANE_ERR_IO);
	fprintf(stderr, "%s:\n", path);
	st = membrane_kvdump_read_header(f, &h);
	while (st == MEMBRANE_OK)
	{
		st = membrane_kvdump_read_payload(f, &h, &payload);
		if (st != MEMBRANE_OK)
			break ;
		st = analyze_record(out, o, path, &h, payload);
		free(payload);
		if (st != MEMBRANE_OK)
			break ;
		st = membrane_kvdump_read_header(f, &h);
	}
	fclose(f);
	if (st == MEMBRANE_ERR_NOT_FOUND)
		return (MEMBRANE_OK);
	return (fprintf(stderr, "error in %s (status %d)\n", path, st), st);
}

static void	human_totals(const kva_totals_t *t)
{
	fprintf(stderr, "\nK vs V (64 KiB blocks, all inputs):\n");
	fprintf(stderr, "  K: raw %llu B  rle %.3fx  adaptive %.3fx\n",
		(unsigned long long)t->raw[0], ratio_of(t->raw[0], t->rle[0]),
		ratio_of(t->raw[0], t->adaptive[0]));
	fprintf(stderr, "  V: raw %llu B  rle %.3fx  adaptive %.3fx\n",
		(unsigned long long)t->raw[1], ratio_of(t->raw[1], t->rle[1]),
		ratio_of(t->raw[1], t->adaptive[1]));
	fprintf(stderr, "  integrity: %s\n",
		t->integrity_ok ? "PASS" : "FAIL");
}

static int	opt_apply(kva_opts_t *o, int c)
{
	if (c == 'j')
		o->jsonl_path = optarg;
	else if (c == 'v')
		o->csv_path = optarg;
	else if (c == 'x' && o->meta_count < MAX_META)
		o->meta[o->meta_count++] = optarg;
	else if (c != 'x')
		return (-1);
	return (0);
}

static int	parse_opts(int argc, char **argv, kva_opts_t *o)
{
	static struct option	lo[] = {
	{"jsonl", required_argument, 0, 'j'},
	{"csv", required_argument, 0, 'v'},
	{"meta", required_argument, 0, 'x'},
	{0, 0, 0, 0}};
	int						c;

	memset(o, 0, sizeof(*o));
	c = getopt_long(argc, argv, "j:v:x:", lo, NULL);
	while (c != -1)
	{
		if (opt_apply(o, c) != 0)
			return (-1);
		c = getopt_long(argc, argv, "j:v:x:", lo, NULL);
	}
	o->inputs = argv + optind;
	o->input_count = argc - optind;
	if (o->input_count < 1 || o->jsonl_path == NULL || o->csv_path == NULL)
	{
		fprintf(stderr, "Usage: membrane-kv-analyze --jsonl OUT --csv OUT "
			"[--meta k=v]... DUMP...\n");
		return (-1);
	}
	return (0);
}

static int	open_outputs(kva_out_t *out, const kva_opts_t *o)
{
	out->jsonl = fopen(o->jsonl_path, "w");
	out->csv = fopen(o->csv_path, "w");
	if (out->jsonl == NULL || out->csv == NULL)
		return (fprintf(stderr, "cannot open outputs\n"), -1);
	fprintf(out->csv, "file,model,layer,tensor,token_start,token_end,dtype,"
		"block_size,blocks,raw_bytes,rle_bytes,rle_ratio,adaptive_bytes,"
		"adaptive_ratio,zero_ratio,entropy_bits,max_run,integrity\n");
	out->totals.integrity_ok = 1;
	return (0);
}

int	main(int argc, char **argv)
{
	kva_opts_t	o;
	kva_out_t	out;
	int			i;
	int			rc;

	if (parse_opts(argc, argv, &o) != 0)
		return (2);
	memset(&out, 0, sizeof(out));
	if (open_outputs(&out, &o) != 0)
		return (2);
	print_env(&o, &out);
	rc = 0;
	i = 0;
	while (i < o.input_count)
	{
		if (analyze_file(&out, &o, o.inputs[i]) != MEMBRANE_OK)
			rc = 1;
		i++;
	}
	human_totals(&out.totals);
	fclose(out.jsonl);
	fclose(out.csv);
	if (rc == 0 && !out.totals.integrity_ok)
		rc = 1;
	return (rc);
}
