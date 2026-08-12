#include "kv_store_telemetry.h"

#include <string.h>
#include <sys/resource.h>

uint64_t	membrane_kv_store_total_bytes(const membrane_kv_store_bytes_t *b)
{
	if (b == NULL)
		return (0);
	return (b->n_layer * b->kv_size
		* (b->bytes_per_token_k + b->bytes_per_token_v));
}

/*
 * /proc/self/status lines look like "VmRSS:	  12345 kB\n". Parses the
 * two we care about; leaves proc_status_ok at 0 if the file can't be
 * opened or either field is never found (e.g. non-Linux), so callers
 * can tell a genuine 0 kB apart from "couldn't measure".
 */
static void	parse_proc_status(membrane_kv_store_rss_t *out)
{
	FILE			*f;
	char			line[256];
	int				have_rss;
	int				have_hwm;
	unsigned long long	kb;

	have_rss = 0;
	have_hwm = 0;
	f = fopen("/proc/self/status", "r");
	if (f == NULL)
		return ;
	while (fgets(line, sizeof(line), f) != NULL)
	{
		if (!have_rss && sscanf(line, "VmRSS: %llu kB", &kb) == 1)
		{
			out->vm_rss_kb = (uint64_t)kb;
			have_rss = 1;
		}
		else if (!have_hwm && sscanf(line, "VmHWM: %llu kB", &kb) == 1)
		{
			out->vm_hwm_kb = (uint64_t)kb;
			have_hwm = 1;
		}
		if (have_rss && have_hwm)
			break ;
	}
	fclose(f);
	out->proc_status_ok = have_rss && have_hwm;
}

void	membrane_kv_store_read_rss(membrane_kv_store_rss_t *out)
{
	struct rusage	ru;

	if (out == NULL)
		return ;
	memset(out, 0, sizeof(*out));
	parse_proc_status(out);
	if (getrusage(RUSAGE_SELF, &ru) == 0)
		out->ru_maxrss_kb = (uint64_t)ru.ru_maxrss;
}

void	membrane_kv_store_rss_max(const membrane_kv_store_rss_t *a,
			const membrane_kv_store_rss_t *b, membrane_kv_store_rss_t *out)
{
	if (a == NULL || b == NULL || out == NULL)
		return ;
	out->vm_rss_kb = a->vm_rss_kb > b->vm_rss_kb ? a->vm_rss_kb : b->vm_rss_kb;
	out->vm_hwm_kb = a->vm_hwm_kb > b->vm_hwm_kb ? a->vm_hwm_kb : b->vm_hwm_kb;
	out->ru_maxrss_kb = a->ru_maxrss_kb > b->ru_maxrss_kb
		? a->ru_maxrss_kb : b->ru_maxrss_kb;
	out->proc_status_ok = a->proc_status_ok && b->proc_status_ok;
}

static void	print_rss_human(const char *label, const membrane_kv_store_rss_t *r,
			FILE *out)
{
	fprintf(out, "  %-24s VmRSS=%llu kB  VmHWM=%llu kB  ru_maxrss=%llu kB%s\n",
		label, (unsigned long long)r->vm_rss_kb,
		(unsigned long long)r->vm_hwm_kb,
		(unsigned long long)r->ru_maxrss_kb,
		r->proc_status_ok ? "" : "  (/proc unavailable)");
}

void	membrane_kv_store_print_human(const membrane_kv_store_telemetry_t *t,
			FILE *out)
{
	if (t == NULL)
		return ;
	fprintf(out, "kv_store_mode: %s  ctx_size: %u  generated_tokens: "
		"%llu\n", t->kv_store_mode_name, t->ctx_size,
		(unsigned long long)t->generated_tokens);
	fprintf(out, "native_kv_allocated_bytes:     %llu\n",
		(unsigned long long)t->native_kv_allocated_bytes);
	fprintf(out, "compressed_kv_allocated_bytes: %llu\n",
		(unsigned long long)t->compressed_kv_allocated_bytes);
	fprintf(out, "metadata_bytes:                %llu\n",
		(unsigned long long)t->metadata_bytes);
	fprintf(out, "scratch_peak_bytes:            %llu\n",
		(unsigned long long)t->scratch_peak_bytes);
	print_rss_human("rss_after_model_load", &t->rss_after_model_load, out);
	print_rss_human("rss_after_context", &t->rss_after_context, out);
	print_rss_human("rss_after_prompt", &t->rss_after_prompt, out);
	print_rss_human("rss_final", &t->rss_final, out);
	print_rss_human("rss_peak", &t->rss_peak, out);
	fprintf(out, "prompt_tok_per_s: %.3f  generation_tok_per_s: %.3f\n",
		t->prompt_tok_per_s, t->generation_tok_per_s);
	fprintf(out, "encode_seconds: %.6f  decode_seconds: %.6f\n",
		t->encode_seconds, t->decode_seconds);
	if (t->quality_available)
		fprintf(out, "token_identity: %s  first_divergence: %d  "
			"logit_rel_l2: %.6f  top1_preservation: %.4f  "
			"delta_nll: %.6f\n",
			t->token_identity ? "identical" : "diverged",
			t->first_divergence, t->logit_rel_l2, t->top1_preservation,
			t->delta_nll);
	fprintf(out, "q4_blocks: %llu  q8_blocks: %llu  encoded_blocks: %llu\n",
		(unsigned long long)t->q4_blocks, (unsigned long long)t->q8_blocks,
		(unsigned long long)t->encoded_blocks);
	fprintf(out, "no_fallback_occurred: %s\n",
		t->no_fallback_occurred ? "true" : "false");
}

static void	print_rss_json(const char *key, const membrane_kv_store_rss_t *r,
			FILE *out)
{
	fprintf(out, "\"%s\":{\"vm_rss_kb\":%llu,\"vm_hwm_kb\":%llu,"
		"\"ru_maxrss_kb\":%llu,\"proc_status_ok\":%s}", key,
		(unsigned long long)r->vm_rss_kb, (unsigned long long)r->vm_hwm_kb,
		(unsigned long long)r->ru_maxrss_kb,
		r->proc_status_ok ? "true" : "false");
}

void	membrane_kv_store_print_json(const membrane_kv_store_telemetry_t *t,
			FILE *out)
{
	if (t == NULL)
		return ;
	fprintf(out, "{\"kv_store_mode\":\"%s\",\"ctx_size\":%u,"
		"\"generated_tokens\":%llu,",
		t->kv_store_mode_name, t->ctx_size,
		(unsigned long long)t->generated_tokens);
	fprintf(out, "\"storage\":{\"native_kv_allocated_bytes\":%llu,"
		"\"compressed_kv_allocated_bytes\":%llu,\"metadata_bytes\":%llu,"
		"\"scratch_peak_bytes\":%llu},",
		(unsigned long long)t->native_kv_allocated_bytes,
		(unsigned long long)t->compressed_kv_allocated_bytes,
		(unsigned long long)t->metadata_bytes,
		(unsigned long long)t->scratch_peak_bytes);
	fprintf(out, "\"memory\":{");
	print_rss_json("rss_after_model_load", &t->rss_after_model_load, out);
	fprintf(out, ",");
	print_rss_json("rss_after_context", &t->rss_after_context, out);
	fprintf(out, ",");
	print_rss_json("rss_after_prompt", &t->rss_after_prompt, out);
	fprintf(out, ",");
	print_rss_json("rss_final", &t->rss_final, out);
	fprintf(out, ",");
	print_rss_json("peak_rss", &t->rss_peak, out);
	fprintf(out, "},");
	fprintf(out, "\"performance\":{\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f,\"encode_seconds\":%.6f,"
		"\"decode_seconds\":%.6f},",
		t->prompt_tok_per_s, t->generation_tok_per_s, t->encode_seconds,
		t->decode_seconds);
	fprintf(out, "\"quality\":{\"available\":%s,\"token_identity\":%s,"
		"\"first_divergence\":%d,\"logit_rel_l2\":%.6f,"
		"\"top1_preservation\":%.6f,\"delta_nll\":%.6f},",
		t->quality_available ? "true" : "false",
		t->token_identity ? "true" : "false", t->first_divergence,
		t->logit_rel_l2, t->top1_preservation, t->delta_nll);
	fprintf(out, "\"compression\":{\"q4_blocks\":%llu,\"q8_blocks\":%llu,"
		"\"encoded_blocks\":%llu},",
		(unsigned long long)t->q4_blocks, (unsigned long long)t->q8_blocks,
		(unsigned long long)t->encoded_blocks);
	fprintf(out, "\"no_fallback_occurred\":%s}",
		t->no_fallback_occurred ? "true" : "false");
}
