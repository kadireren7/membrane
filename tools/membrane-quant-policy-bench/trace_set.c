#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "trace_set.h"

/* ------------------------------------------------------------------ */
/* Discovery                                                           */
/* ------------------------------------------------------------------ */

static int	has_memkv_suffix(const char *name)
{
	size_t	len;

	len = strlen(name);
	return (len > 6 && strcmp(name + len - 6, ".memkv") == 0);
}

/* Parses "layer-NNN-k.memkv" / "layer-NNN-v.memkv" exactly -- anything
 * else (extra characters, missing digits, wrong separators) leaves
 * has_layer/has_tensor false rather than guessing. */
static void	parse_batch_name(const char *name, membrane_trace_set_file_t *f)
{
	const char	*p;
	char		*end;
	long		layer;
	char		tensor_ch;

	f->has_layer = 0;
	f->has_tensor = 0;
	if (strncmp(name, "layer-", 6) != 0)
		return ;
	p = name + 6;
	if (*p == '\0' || !(*p >= '0' && *p <= '9'))
		return ;
	errno = 0;
	layer = strtol(p, &end, 10);
	if (errno != 0 || layer < 0 || end == p)
		return ;
	if (end[0] != '-' || (end[1] != 'k' && end[1] != 'v')
		|| strcmp(end + 2, ".memkv") != 0)
		return ;
	tensor_ch = end[1];
	f->has_layer = 1;
	f->layer = (uint32_t)layer;
	f->has_tensor = 1;
	f->tensor = (char)(tensor_ch - 32);	/* 'k'->'K', 'v'->'V' */
}

static int	cmp_dirent_name(const void *a, const void *b)
{
	return (strcmp(*(const char *const *)a, *(const char *const *)b));
}

static int	conflicts(const membrane_trace_set_file_t *files, size_t n,
				const membrane_trace_set_file_t *cand)
{
	size_t	i;

	if (!cand->has_layer || !cand->has_tensor)
		return (0);
	i = 0;
	while (i < n)
	{
		if (files[i].has_layer && files[i].has_tensor
			&& files[i].layer == cand->layer
			&& files[i].tensor == cand->tensor)
			return (1);
		i++;
	}
	return (0);
}

static membrane_status_t	list_candidates(const char *dir, char ***out_names,
								size_t *out_count, char *err_buf,
								size_t err_cap)
{
	DIR				*d;
	struct dirent	*ent;
	char			**names;
	size_t			count;
	char			entpath[MEMBRANE_TRACE_SET_PATH_CAP];
	struct stat		st;

	d = opendir(dir);
	if (d == NULL)
	{
		snprintf(err_buf, err_cap, "cannot open directory %s", dir);
		return (MEMBRANE_ERR_IO);
	}
	names = malloc(MEMBRANE_TRACE_SET_MAX_FILES * sizeof(*names));
	if (names == NULL)
	{
		closedir(d);
		snprintf(err_buf, err_cap, "out of memory");
		return (MEMBRANE_ERR_ALLOC_FAILED);
	}
	count = 0;
	while ((ent = readdir(d)) != NULL)
	{
		if (!has_memkv_suffix(ent->d_name))
			continue ;
		if (snprintf(entpath, sizeof(entpath), "%s/%s", dir, ent->d_name)
			>= (int)sizeof(entpath))
		{
			snprintf(err_buf, err_cap, "path too long: %s/%s", dir,
				ent->d_name);
			goto fail;
		}
		if (lstat(entpath, &st) != 0)
		{
			snprintf(err_buf, err_cap, "cannot stat %s", entpath);
			goto fail;
		}
		if (S_ISLNK(st.st_mode))
		{
			snprintf(err_buf, err_cap, "refusing to follow symlink in "
				"trace directory: %s", ent->d_name);
			goto fail;
		}
		if (!S_ISREG(st.st_mode))
			continue ;
		if (count >= MEMBRANE_TRACE_SET_MAX_FILES)
		{
			snprintf(err_buf, err_cap, "more than %u .memkv files in %s "
				"(cap)", MEMBRANE_TRACE_SET_MAX_FILES, dir);
			goto fail;
		}
		names[count] = strdup(ent->d_name);
		if (names[count] == NULL)
		{
			snprintf(err_buf, err_cap, "out of memory");
			goto fail;
		}
		count++;
	}
	closedir(d);
	qsort(names, count, sizeof(*names), cmp_dirent_name);
	*out_names = names;
	*out_count = count;
	return (MEMBRANE_OK);
fail:
	closedir(d);
	while (count > 0)
	{
		count--;
		free(names[count]);
	}
	free(names);
	return (MEMBRANE_ERR_INVALID_ARG);
}

membrane_status_t	membrane_trace_set_discover(const char *dir,
						membrane_trace_set_file_t **out, size_t *out_count,
						uint64_t *out_total_blocks, char *err_buf,
						size_t err_cap)
{
	char						**names;
	size_t						n;
	membrane_trace_set_file_t	*files;
	membrane_trace_reader_t		*r;
	membrane_trace_info_t		info;
	uint64_t					total;
	size_t						i;
	membrane_status_t			st;

	if (err_buf != NULL && err_cap > 0)
		err_buf[0] = '\0';
	if (dir == NULL || out == NULL || out_count == NULL
		|| out_total_blocks == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	st = list_candidates(dir, &names, &n, err_buf, err_cap);
	if (st != MEMBRANE_OK)
		return (st);
	files = malloc((n > 0 ? n : 1) * sizeof(*files));
	if (files == NULL)
	{
		i = 0;
		while (i < n)
			free(names[i++]);
		free(names);
		snprintf(err_buf, err_cap, "out of memory");
		return (MEMBRANE_ERR_ALLOC_FAILED);
	}
	total = 0;
	i = 0;
	while (i < n)
	{
		snprintf(files[i].path, sizeof(files[i].path), "%s/%s", dir,
			names[i]);
		parse_batch_name(names[i], &files[i]);
		if (conflicts(files, i, &files[i]))
		{
			snprintf(err_buf, err_cap, "duplicate/conflicting trace for "
				"layer=%u tensor=%c: %s", files[i].layer, files[i].tensor,
				names[i]);
			st = MEMBRANE_ERR_INVALID_ARG;
			break ;
		}
		st = membrane_trace_open(files[i].path, &r);
		if (st != MEMBRANE_OK)
		{
			snprintf(err_buf, err_cap, "%s failed validation (status=%d) "
				"-- corrupt or unsupported trace", names[i], (int)st);
			break ;
		}
		membrane_trace_info(r, &info);
		membrane_trace_close(r);
		total += info.block_count;
		if (total > MEMBRANE_TRACE_SET_MAX_TOTAL_BLOCKS)
		{
			snprintf(err_buf, err_cap, "total block count across %s "
				"exceeds the trace-set cap (%llu)", dir,
				(unsigned long long)MEMBRANE_TRACE_SET_MAX_TOTAL_BLOCKS);
			st = MEMBRANE_ERR_INVALID_ARG;
			break ;
		}
		i++;
	}
	i = 0;
	while (i < n)
		free(names[i++]);
	free(names);
	if (st != MEMBRANE_OK)
	{
		free(files);
		return (st);
	}
	if (n == 0)
	{
		free(files);
		snprintf(err_buf, err_cap, "no .memkv files found in %s", dir);
		return (MEMBRANE_ERR_NOT_FOUND);
	}
	*out = files;
	*out_count = n;
	*out_total_blocks = total;
	return (MEMBRANE_OK);
}

/* ------------------------------------------------------------------ */
/* Run + aggregate                                                     */
/* ------------------------------------------------------------------ */

static double	safe_ratio(uint64_t num, uint64_t den)
{
	if (den == 0)
		return (0.0);
	return ((double)num / (double)den);
}

static int	cmp_double(const void *a, const void *b)
{
	double	da;
	double	db;

	da = *(const double *)a;
	db = *(const double *)b;
	if (da < db)
		return (-1);
	if (da > db)
		return (1);
	return (0);
}

static void	compute_aggregate(const membrane_trace_set_item_t *items,
				size_t n, membrane_trace_set_aggregate_t *agg)
{
	double		*ratios;
	size_t		n_adaptive;
	double		q4_num;
	double		q8_num;
	size_t		i;
	const membrane_bench_result_t	*r;

	memset(agg, 0, sizeof(*agg));
	ratios = malloc((n > 0 ? n : 1) * sizeof(*ratios));
	n_adaptive = 0;
	q4_num = 0.0;
	q8_num = 0.0;
	agg->min_q4_ratio = 2.0;
	agg->max_q4_ratio = -1.0;
	i = 0;
	while (i < n)
	{
		if (items[i].result.config.policy == MEMBRANE_BENCH_POLICY_ADAPTIVE)
		{
			r = &items[i].result;
			agg->traces++;
			agg->total_blocks += (uint64_t)r->config.blocks;
			agg->q4_blocks += r->q4_blocks;
			agg->q8_blocks += r->q8_blocks;
			agg->baseline_bytes += r->baseline_bytes;
			agg->encoded_bytes += r->encoded_bytes;
			agg->decode_failures += r->decode_failures;
			agg->total_measured_seconds += r->median_seconds;
			q4_num += r->q4_mean_rel_l2_error * (double)r->q4_blocks;
			q8_num += r->q8_mean_rel_l2_error * (double)r->q8_blocks;
			if (r->q4_max_rel_l2_error > agg->q4_max_rel_l2_error)
				agg->q4_max_rel_l2_error = r->q4_max_rel_l2_error;
			if (r->q8_max_rel_l2_error > agg->q8_max_rel_l2_error)
				agg->q8_max_rel_l2_error = r->q8_max_rel_l2_error;
			if (r->q4_blocks > 0)
				agg->traces_any_q4++;
			else
				agg->traces_all_q8++;
			ratios[n_adaptive] = safe_ratio(r->q4_blocks,
					r->q4_blocks + r->q8_blocks);
			if (ratios[n_adaptive] < agg->min_q4_ratio)
			{
				agg->min_q4_ratio = ratios[n_adaptive];
				snprintf(agg->min_q4_ratio_trace,
					sizeof(agg->min_q4_ratio_trace), "%s", r->workload_name);
			}
			if (ratios[n_adaptive] > agg->max_q4_ratio)
			{
				agg->max_q4_ratio = ratios[n_adaptive];
				snprintf(agg->max_q4_ratio_trace,
					sizeof(agg->max_q4_ratio_trace), "%s", r->workload_name);
			}
			n_adaptive++;
		}
		i++;
	}
	agg->q4_ratio = safe_ratio(agg->q4_blocks, agg->total_blocks);
	agg->storage_reduction_ratio = safe_ratio(
			agg->baseline_bytes - agg->encoded_bytes, agg->baseline_bytes);
	agg->q4_mean_rel_l2_error = agg->q4_blocks > 0
		? q4_num / (double)agg->q4_blocks : 0.0;
	agg->q8_mean_rel_l2_error = agg->q8_blocks > 0
		? q8_num / (double)agg->q8_blocks : 0.0;
	agg->weighted_mean_rel_l2_error = (agg->q4_blocks + agg->q8_blocks) > 0
		? (q4_num + q8_num) / (double)(agg->q4_blocks + agg->q8_blocks)
		: 0.0;
	agg->blocks_per_second = agg->total_measured_seconds > 0.0
		? (double)agg->total_blocks / agg->total_measured_seconds : 0.0;
	if (n_adaptive == 0)
	{
		agg->min_q4_ratio = 0.0;
		agg->max_q4_ratio = 0.0;
	}
	else
	{
		qsort(ratios, n_adaptive, sizeof(*ratios), cmp_double);
		if (n_adaptive % 2 == 1)
			agg->median_q4_ratio = ratios[n_adaptive / 2];
		else
			agg->median_q4_ratio = (ratios[n_adaptive / 2 - 1]
					+ ratios[n_adaptive / 2]) / 2.0;
	}
	free(ratios);
}

membrane_status_t	membrane_trace_set_run(
						const membrane_bench_config_t *base,
						const membrane_trace_set_file_t *files,
						size_t n_files, membrane_trace_set_item_t **out,
						size_t *out_count, membrane_trace_set_aggregate_t *out_agg)
{
	membrane_trace_set_item_t	*items;
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n_cells;
	size_t						i;
	size_t						k;
	size_t						pos;
	membrane_status_t			st;

	if (base == NULL || files == NULL || n_files == 0 || out == NULL
		|| out_count == NULL || out_agg == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	items = malloc(n_files * 3 * sizeof(*items));
	if (items == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	pos = 0;
	i = 0;
	while (i < n_files)
	{
		cfg = *base;
		cfg.trace_path = files[i].path;
		st = membrane_bench_run_matrix(&cfg, rs, MEMBRANE_BENCH_MATRIX_CELLS,
				&n_cells);
		if (st != MEMBRANE_OK)
			return (free(items), st);
		k = 0;
		while (k < n_cells)
		{
			items[pos].result = rs[k];
			items[pos].has_layer = files[i].has_layer;
			items[pos].layer = files[i].layer;
			items[pos].has_tensor = files[i].has_tensor;
			items[pos].tensor = files[i].tensor;
			pos++;
			k++;
		}
		i++;
	}
	compute_aggregate(items, pos, out_agg);
	*out = items;
	*out_count = pos;
	return (MEMBRANE_OK);
}

/* ------------------------------------------------------------------ */
/* Printers                                                            */
/* ------------------------------------------------------------------ */

void	membrane_trace_set_print_human(const membrane_trace_set_item_t *items,
			size_t n, const membrane_trace_set_aggregate_t *agg, FILE *f)
{
	size_t	i;

	fprintf(f, "MEMBRANE multi-trace KV analysis (one real capture, not "
		"general LLM behavior; timing is local CPU only)\n\n");
	fprintf(f, "%-8s %-6s %-10s %8s %7s %7s %10s %10s\n",
		"Layer", "Tensor", "Policy", "Blocks", "Q4%", "Q8%", "Storage%",
		"MeanErr");
	i = 0;
	while (i < n)
	{
		char	layer_buf[16];
		char	tensor_buf[4];
		uint64_t	total;

		if (items[i].has_layer)
			snprintf(layer_buf, sizeof(layer_buf), "%u", items[i].layer);
		else
			snprintf(layer_buf, sizeof(layer_buf), "?");
		snprintf(tensor_buf, sizeof(tensor_buf), "%c",
			items[i].has_tensor ? items[i].tensor : '?');
		total = items[i].result.q4_blocks + items[i].result.q8_blocks;
		fprintf(f, "%-8s %-6s %-10s %8u %6.1f%% %6.1f%% %9.2f%% %10.4f\n",
			layer_buf, tensor_buf,
			membrane_bench_policy_name(items[i].result.config.policy),
			items[i].result.config.blocks,
			safe_ratio(items[i].result.q4_blocks, total) * 100.0,
			safe_ratio(items[i].result.q8_blocks, total) * 100.0,
			items[i].result.reduction_ratio * 100.0,
			membrane_bench_weighted_mean_rel_l2_error(&items[i].result));
		i++;
	}
	fprintf(f, "\nAdaptive totals (this one capture only)\n");
	fprintf(f, "  traces                 %u\n", agg->traces);
	fprintf(f, "  blocks                 %llu\n",
		(unsigned long long)agg->total_blocks);
	fprintf(f, "  Q4 blocks              %llu\n",
		(unsigned long long)agg->q4_blocks);
	fprintf(f, "  Q8 blocks              %llu\n",
		(unsigned long long)agg->q8_blocks);
	fprintf(f, "  Q4 ratio               %.4f\n", agg->q4_ratio);
	fprintf(f, "  storage reduction      %.2f%%\n",
		agg->storage_reduction_ratio * 100.0);
	fprintf(f, "  weighted mean rel-L2   %.6f\n",
		agg->weighted_mean_rel_l2_error);
	fprintf(f, "  min Q4 ratio           %.4f (%s)\n", agg->min_q4_ratio,
		agg->min_q4_ratio_trace);
	fprintf(f, "  max Q4 ratio           %.4f (%s)\n", agg->max_q4_ratio,
		agg->max_q4_ratio_trace);
	fprintf(f, "  median Q4 ratio        %.4f\n", agg->median_q4_ratio);
	fprintf(f, "  traces selecting any Q4 %llu / %u\n",
		(unsigned long long)agg->traces_any_q4, agg->traces);
	fprintf(f, "  traces at 100%% Q8      %llu / %u\n",
		(unsigned long long)agg->traces_all_q8, agg->traces);
}

static void	print_item_json(const membrane_trace_set_item_t *it, FILE *f)
{
	const membrane_bench_result_t	*r;

	r = &it->result;
	fprintf(f, "{\"trace\":\"");
	membrane_bench_json_escape(f, r->workload_name);
	fprintf(f, "\",\"workload_kind\":\"%s\",\"layer\":", r->workload_kind_label);
	if (it->has_layer)
		fprintf(f, "%u", it->layer);
	else
		fprintf(f, "null");
	fprintf(f, ",\"tensor\":");
	if (it->has_tensor)
		fprintf(f, "\"%c\"", it->tensor);
	else
		fprintf(f, "null");
	fprintf(f, ",\"policy\":\"%s\",\"blocks\":%u,\"elements_per_block\":%u,"
		"\"storage\":{\"baseline_bytes\":%llu,\"encoded_bytes\":%llu,"
		"\"storage_reduction_ratio\":%.6f},"
		"\"precision_counts\":{\"q4\":%llu,\"q8\":%llu},"
		"\"accuracy\":{\"q4_mean_rel_l2_error\":%.6f,"
		"\"q4_max_rel_l2_error\":%.6f,\"q8_mean_rel_l2_error\":%.6f,"
		"\"q8_max_rel_l2_error\":%.6f,\"decode_failures\":%llu,"
		"\"validation_pass\":%s},"
		"\"timing\":{\"median_seconds\":%.6f,\"blocks_per_second\":%.3f}}",
		membrane_bench_policy_name(r->config.policy), r->config.blocks,
		r->elems_per_block, (unsigned long long)r->baseline_bytes,
		(unsigned long long)r->encoded_bytes, r->reduction_ratio,
		(unsigned long long)r->q4_blocks, (unsigned long long)r->q8_blocks,
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error,
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error,
		(unsigned long long)r->decode_failures,
		r->validation_pass ? "true" : "false", r->median_seconds,
		r->blocks_per_second);
}

void	membrane_trace_set_print_json(const membrane_trace_set_item_t *items,
			size_t n, const membrane_trace_set_aggregate_t *agg, FILE *f)
{
	size_t	i;

	fprintf(f, "{\"schema_version\":%d,\"mode\":\"trace_set\",\"results\":[",
		MEMBRANE_TRACE_SET_SCHEMA_VERSION);
	i = 0;
	while (i < n)
	{
		if (i > 0)
			fprintf(f, ",");
		print_item_json(&items[i], f);
		i++;
	}
	fprintf(f, "],\"aggregate\":{\"policy\":\"adaptive\",\"traces\":%u,"
		"\"blocks\":%llu,\"q4_blocks\":%llu,\"q8_blocks\":%llu,"
		"\"q4_ratio\":%.6f,\"baseline_bytes\":%llu,\"encoded_bytes\":%llu,"
		"\"storage_reduction_ratio\":%.6f,\"q4_mean_rel_l2_error\":%.6f,"
		"\"q4_max_rel_l2_error\":%.6f,\"q8_mean_rel_l2_error\":%.6f,"
		"\"q8_max_rel_l2_error\":%.6f,"
		"\"weighted_mean_rel_l2_error\":%.6f,\"decode_failures\":%llu,"
		"\"total_measured_seconds\":%.6f,\"blocks_per_second\":%.3f,"
		"\"insight\":{\"min_q4_ratio\":%.6f,\"min_q4_ratio_trace\":\"",
		agg->traces, (unsigned long long)agg->total_blocks,
		(unsigned long long)agg->q4_blocks,
		(unsigned long long)agg->q8_blocks, agg->q4_ratio,
		(unsigned long long)agg->baseline_bytes,
		(unsigned long long)agg->encoded_bytes,
		agg->storage_reduction_ratio, agg->q4_mean_rel_l2_error,
		agg->q4_max_rel_l2_error, agg->q8_mean_rel_l2_error,
		agg->q8_max_rel_l2_error, agg->weighted_mean_rel_l2_error,
		(unsigned long long)agg->decode_failures,
		agg->total_measured_seconds, agg->blocks_per_second,
		agg->min_q4_ratio);
	membrane_bench_json_escape(f, agg->min_q4_ratio_trace);
	fprintf(f, "\",\"max_q4_ratio\":%.6f,\"max_q4_ratio_trace\":\"",
		agg->max_q4_ratio);
	membrane_bench_json_escape(f, agg->max_q4_ratio_trace);
	fprintf(f, "\",\"median_q4_ratio\":%.6f,\"traces_any_q4\":%llu,"
		"\"traces_all_q8\":%llu}}}\n", agg->median_q4_ratio,
		(unsigned long long)agg->traces_any_q4,
		(unsigned long long)agg->traces_all_q8);
}

void	membrane_trace_set_print_csv(const membrane_trace_set_item_t *items,
			size_t n, FILE *f)
{
	size_t	i;
	const membrane_bench_result_t	*r;

	fprintf(f, "trace,workload_kind,layer,tensor,policy,blocks,"
		"elements_per_block,baseline_bytes,encoded_bytes,"
		"storage_reduction_ratio,q4_blocks,q8_blocks,"
		"q4_mean_rel_l2_error,q4_max_rel_l2_error,q8_mean_rel_l2_error,"
		"q8_max_rel_l2_error,decode_failures,validation_pass,"
		"median_seconds,blocks_per_second,simd_backend\n");
	i = 0;
	while (i < n)
	{
		r = &items[i].result;
		membrane_bench_csv_escape(f, r->workload_name);
		fprintf(f, ",%s,", r->workload_kind_label);
		if (items[i].has_layer)
			fprintf(f, "%u", items[i].layer);
		fprintf(f, ",");
		if (items[i].has_tensor)
			fprintf(f, "%c", items[i].tensor);
		fprintf(f, ",%s,%u,%u,%llu,%llu,%.6f,%llu,%llu,%.6f,%.6f,%.6f,%.6f,"
			"%llu,%s,%.6f,%.3f,%s\n",
			membrane_bench_policy_name(r->config.policy), r->config.blocks,
			r->elems_per_block, (unsigned long long)r->baseline_bytes,
			(unsigned long long)r->encoded_bytes, r->reduction_ratio,
			(unsigned long long)r->q4_blocks,
			(unsigned long long)r->q8_blocks, r->q4_mean_rel_l2_error,
			r->q4_max_rel_l2_error, r->q8_mean_rel_l2_error,
			r->q8_max_rel_l2_error, (unsigned long long)r->decode_failures,
			r->validation_pass ? "PASS" : "FAIL", r->median_seconds,
			r->blocks_per_second, r->simd_backend_name);
		i++;
	}
}
