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
# define MAX_LAYERS 64
# define SUMMARY_BLOCK 65536

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

/* Aggregates for the human K-vs-V summary at 64 KiB blocks. Index 0 = K,
 * 1 = V for the per-tensor arrays. */
typedef struct s_kva_totals
{
	uint64_t	raw[2];
	uint64_t	rle[2];
	uint64_t	adaptive[2];
	uint64_t	byteplane[2];
	uint64_t	byteplane_adaptive[2];
	int			integrity_ok;
	int			byteplane_integrity_ok;
}	kva_totals_t;

/* Per-layer byteplane aggregate at 64 KiB blocks, across all prompts. */
typedef struct s_kva_layer
{
	uint64_t	raw;
	uint64_t	byteplane_adaptive;
	int			seen;
}	kva_layer_t;

/* Per-prompt (one input file) aggregate at 64 KiB blocks. */
typedef struct s_kva_file_sum
{
	uint64_t	raw;
	uint64_t	rle;
	uint64_t	adaptive;
	uint64_t	byteplane;
	uint64_t	byteplane_adaptive;
}	kva_file_sum_t;

typedef struct s_kva_out
{
	FILE		*jsonl;
	FILE		*csv;
	kva_totals_t	totals;
	kva_layer_t	layers[MAX_LAYERS];
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

static void	emit_jsonl_byteplane(FILE *f, const membrane_kv_metrics_t *m)
{
	fprintf(f,
		",\"byteplane_bytes\":%llu,\"byteplane_ratio\":%.4f,"
		"\"byteplane_adaptive_bytes\":%llu,\"byteplane_adaptive_ratio\":%.4f,"
		"\"byteplane_raw_blocks\":%llu,\"byteplane_codec_blocks\":%llu,"
		"\"low_entropy_bits\":%.4f,\"high_entropy_bits\":%.4f,"
		"\"low_plane_ratio\":%.4f,\"high_plane_ratio\":%.4f,"
		"\"byteplane_applicable\":%d,\"byteplane_integrity\":\"%s\"",
		(unsigned long long)m->byteplane_bytes,
		ratio_of(m->raw_bytes, m->byteplane_bytes),
		(unsigned long long)m->byteplane_adaptive_bytes,
		ratio_of(m->raw_bytes, m->byteplane_adaptive_bytes),
		(unsigned long long)m->byteplane_raw_blocks,
		(unsigned long long)m->byteplane_codec_blocks,
		m->low_entropy, m->high_entropy,
		ratio_of(m->low_plane_bytes, m->low_plane_rle_bytes),
		ratio_of(m->high_plane_bytes, m->high_plane_rle_bytes),
		m->byteplane_applicable,
		m->byteplane_integrity_ok ? "PASS" : "FAIL");
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
	emit_jsonl_byteplane(out->jsonl, m);
	print_meta_json(out->jsonl, o);
	fprintf(out->jsonl, "}\n");
}

static void	emit_csv(kva_out_t *out, const char *file,
				const membrane_kv_header_t *h, size_t bs,
				const membrane_kv_metrics_t *m)
{
	fprintf(out->csv,
		"%s,%s,%u,%c,%u,%u,%u,%zu,%llu,%llu,%llu,%.4f,%llu,%.4f,"
		"%.6f,%.4f,%llu,%s,"
		"%llu,%.4f,%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%s\n",
		file, h->model, h->layer, h->tensor_type ? 'V' : 'K',
		h->token_start, h->token_end, h->dtype, bs,
		(unsigned long long)m->blocks,
		(unsigned long long)m->raw_bytes,
		(unsigned long long)m->rle_bytes, ratio_of(m->raw_bytes, m->rle_bytes),
		(unsigned long long)m->adaptive_bytes,
		ratio_of(m->raw_bytes, m->adaptive_bytes),
		m->raw_bytes ? (double)m->zero_bytes / (double)m->raw_bytes : 0.0,
		m->entropy, (unsigned long long)m->max_run,
		m->integrity_ok ? "PASS" : "FAIL",
		(unsigned long long)m->byteplane_bytes,
		ratio_of(m->raw_bytes, m->byteplane_bytes),
		(unsigned long long)m->byteplane_adaptive_bytes,
		ratio_of(m->raw_bytes, m->byteplane_adaptive_bytes),
		m->low_entropy, m->high_entropy,
		ratio_of(m->low_plane_bytes, m->low_plane_rle_bytes),
		ratio_of(m->high_plane_bytes, m->high_plane_rle_bytes),
		m->byteplane_applicable,
		m->byteplane_integrity_ok ? "PASS" : "FAIL");
}

static void	human_record(const membrane_kv_header_t *h,
				const membrane_kv_metrics_t *m)
{
	fprintf(stderr,
		"  layer %2u %c  tokens %u..%u  raw %7llu B  adaptive %.3fx  "
		"byteplane %.3fx (adaptive %.3fx)  Hlo %.3f Hhi %.3f  %s\n",
		h->layer, h->tensor_type ? 'V' : 'K', h->token_start, h->token_end,
		(unsigned long long)m->raw_bytes,
		ratio_of(m->raw_bytes, m->adaptive_bytes),
		ratio_of(m->raw_bytes, m->byteplane_bytes),
		ratio_of(m->raw_bytes, m->byteplane_adaptive_bytes),
		m->low_entropy, m->high_entropy,
		(m->integrity_ok && m->byteplane_integrity_ok) ? "PASS" : "FAIL");
}

static void	totals_add(kva_out_t *out, const membrane_kv_header_t *h,
				const membrane_kv_metrics_t *m)
{
	kva_totals_t	*t;
	int				idx;

	t = &out->totals;
	idx = (h->tensor_type != 0);
	t->raw[idx] += m->raw_bytes;
	t->rle[idx] += m->rle_bytes;
	t->adaptive[idx] += m->adaptive_bytes;
	t->byteplane[idx] += m->byteplane_bytes;
	t->byteplane_adaptive[idx] += m->byteplane_adaptive_bytes;
	if (!m->integrity_ok)
		t->integrity_ok = 0;
	if (!m->byteplane_integrity_ok || !m->byteplane_applicable)
		t->byteplane_integrity_ok = 0;
	if (h->layer < MAX_LAYERS)
	{
		out->layers[h->layer].raw += m->raw_bytes;
		out->layers[h->layer].byteplane_adaptive += m->byteplane_adaptive_bytes;
		out->layers[h->layer].seen = 1;
	}
}

static void	file_sum_add(kva_file_sum_t *fs, const membrane_kv_metrics_t *m)
{
	fs->raw += m->raw_bytes;
	fs->rle += m->rle_bytes;
	fs->adaptive += m->adaptive_bytes;
	fs->byteplane += m->byteplane_bytes;
	fs->byteplane_adaptive += m->byteplane_adaptive_bytes;
}

static membrane_status_t	analyze_record(kva_out_t *out,
								const kva_opts_t *o, const char *file,
								const membrane_kv_header_t *h,
								const uint8_t *payload, kva_file_sum_t *fs)
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
		if (g_block_sizes[i] == SUMMARY_BLOCK)
		{
			human_record(h, &m);
			totals_add(out, h, &m);
			file_sum_add(fs, &m);
		}
		i++;
	}
	return (MEMBRANE_OK);
}

static void	human_file_sum(const char *path, const kva_file_sum_t *fs)
{
	fprintf(stderr,
		"  prompt total: raw %llu B  adaptive %.3fx  byteplane %.3fx"
		" (adaptive %.3fx)\n",
		(unsigned long long)fs->raw, ratio_of(fs->raw, fs->adaptive),
		ratio_of(fs->raw, fs->byteplane),
		ratio_of(fs->raw, fs->byteplane_adaptive));
	(void)path;
}

static membrane_status_t	analyze_file(kva_out_t *out, const kva_opts_t *o,
								const char *path)
{
	FILE					*f;
	membrane_kv_header_t	h;
	uint8_t					*payload;
	kva_file_sum_t			fs;
	membrane_status_t		st;

	f = fopen(path, "rb");
	if (f == NULL)
		return (fprintf(stderr, "cannot open %s\n", path), MEMBRANE_ERR_IO);
	fprintf(stderr, "%s:\n", path);
	memset(&fs, 0, sizeof(fs));
	st = membrane_kvdump_read_header(f, &h);
	while (st == MEMBRANE_OK)
	{
		st = membrane_kvdump_read_payload(f, &h, &payload);
		if (st != MEMBRANE_OK)
			break ;
		st = analyze_record(out, o, path, &h, payload, &fs);
		free(payload);
		if (st != MEMBRANE_OK)
			break ;
		st = membrane_kvdump_read_header(f, &h);
	}
	fclose(f);
	if (st == MEMBRANE_ERR_NOT_FOUND)
		return (human_file_sum(path, &fs), MEMBRANE_OK);
	return (fprintf(stderr, "error in %s (status %d)\n", path, st), st);
}

static void	human_kv_line(const kva_totals_t *t, int idx, char label)
{
	fprintf(stderr,
		"  %c: raw %llu B  rle %.3fx  adaptive %.3fx  byteplane %.3fx"
		" (adaptive %.3fx)\n",
		label, (unsigned long long)t->raw[idx],
		ratio_of(t->raw[idx], t->rle[idx]),
		ratio_of(t->raw[idx], t->adaptive[idx]),
		ratio_of(t->raw[idx], t->byteplane[idx]),
		ratio_of(t->raw[idx], t->byteplane_adaptive[idx]));
}

static void	human_layers(const kva_out_t *out)
{
	int	i;

	fprintf(stderr, "\nPer-layer byteplane adaptive (64 KiB blocks,"
		" K+V, all prompts):\n");
	i = 0;
	while (i < MAX_LAYERS)
	{
		if (out->layers[i].seen)
			fprintf(stderr, "  layer %2d: raw %llu B  byteplane %.3fx\n",
				i, (unsigned long long)out->layers[i].raw,
				ratio_of(out->layers[i].raw,
					out->layers[i].byteplane_adaptive));
		i++;
	}
}

static void	human_totals(const kva_out_t *out)
{
	const kva_totals_t	*t;

	t = &out->totals;
	fprintf(stderr, "\nK vs V (64 KiB blocks, all inputs):\n");
	human_kv_line(t, 0, 'K');
	human_kv_line(t, 1, 'V');
	human_layers(out);
	fprintf(stderr, "\n  integrity: adaptive %s  byteplane %s\n",
		t->integrity_ok ? "PASS" : "FAIL",
		t->byteplane_integrity_ok ? "PASS" : "FAIL");
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
		"adaptive_ratio,zero_ratio,entropy_bits,max_run,integrity,"
		"byteplane_bytes,byteplane_ratio,byteplane_adaptive_bytes,"
		"byteplane_adaptive_ratio,low_entropy_bits,high_entropy_bits,"
		"low_plane_ratio,high_plane_ratio,byteplane_applicable,"
		"byteplane_integrity\n");
	out->totals.integrity_ok = 1;
	out->totals.byteplane_integrity_ok = 1;
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
	human_totals(&out);
	fclose(out.jsonl);
	fclose(out.csv);
	if (rc == 0 && (!out.totals.integrity_ok
			|| !out.totals.byteplane_integrity_ok))
		rc = 1;
	return (rc);
}
