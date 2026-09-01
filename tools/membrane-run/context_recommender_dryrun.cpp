/*
 * Phase 33, Section 30: a REAL, dev-only evidence-gathering harness --
 * NOT a product CLI surface (no --ctx auto, nothing installed, never
 * referenced by product_cli.h). Feeds real GGUF metadata (gpu_device.h's
 * membrane_gpu_estimate_model(), now extended with model_max_context),
 * real device facts (membrane_gpu_list_devices()), and real
 * ggml_row_size()-based KV-byte arithmetic (the exact same formula
 * main.cpp's own native_kv_bytes()/q8_kv_bytes()/q5_kv_bytes() use)
 * into the new, pure context_recommender.h core, and prints one JSON
 * object with the real result -- used to produce results/context-
 * recommendation/core-validation.json's real_dry_runs entries. Never
 * runs generation, never loads the model past GGUF metadata.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "context_recommender.h"
#include "gpu_device.h"
#include "kv_store_telemetry.h"

#include "ggml.h"

static uint64_t	real_kv_bytes(const membrane_gpu_model_estimate_t &m,
					uint64_t ctx, enum ggml_type type)
{
	int64_t	n_embd_gqa = (m.n_head > 0)
			? (int64_t)(m.n_embd / m.n_head) * m.n_head_kv : 0;
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)m.n_layer;
	b.kv_size = ctx;
	b.bytes_per_token_k = ggml_row_size(type, n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(type, n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static void	print_evaluated(const membrane_ctxrec_evaluated_t *ev)
{
	printf("      {\"ctx\":%llu,\"feasible\":%s,\"reason_code\":\"%s\"",
		(unsigned long long)ev->ctx, ev->feasible ? "true" : "false",
		ev->reason_code);
	if (ev->feasible)
		printf(",\"gpu_layers\":%d,\"kv_precision\":%d,"
			"\"kv_placement\":%d,\"host_resident\":%s",
			ev->selected_gpu_layers, ev->selected_kv_precision,
			ev->selected_kv_placement, ev->host_resident ? "true" : "false");
	printf("}");
}

int	main(int argc, char **argv)
{
	membrane_gpu_model_estimate_t	m;
	membrane_gpu_device_info_t		devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t							n_devices;
	size_t							i;
	int								gpu_index = -1;

	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
		return (1);
	}
	if (!membrane_gpu_estimate_model(argv[1], &m))
	{
		printf("{\"model_path\":\"%s\",\"error\":"
			"\"membrane_gpu_estimate_model failed\"}\n", argv[1]);
		return (1);
	}
	n_devices = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	for (i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
		{
			gpu_index = (int)i;
			break ;
		}

	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	uint64_t					ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t						n_ctxs;

	memset(&req, 0, sizeof(req));
	req.n_layer_all = m.n_layer;
	req.bytes_per_layer = m.bytes_per_layer;
	req.output_role_bytes = m.output_role_bytes;
	req.arch_name = m.arch_name;
	req.n_embd = m.n_embd;
	req.n_head = m.n_head;
	req.n_head_kv = m.n_head_kv;
	req.model_max_context = m.model_max_context;
	req.model_max_context_known = m.model_max_context_available;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	if (gpu_index >= 0)
	{
		/* Real GPU present: mirrors bare `--auto` (adaptive precision,
		 * auto gpu-layers). */
		req.precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
		req.gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
		req.device_free_bytes = devices[gpu_index].memory_free;
		req.device_total_bytes = devices[gpu_index].memory_total;
	}
	else
	{
		/* No GPU device: Section 9 of this phase's own test suite
		 * documents that adaptive precision has no CPU-only fallback in
		 * the existing, unchanged joint planner at zero GPU budget --
		 * explicit native precision is the real, working CPU-only path
		 * (matches main.cpp's own gpu_layers==0 short-circuit, which
		 * never reaches the joint planner with an implicit adaptive
		 * request either). */
		req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
		req.gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
		req.device_free_bytes = 0;
		req.device_total_bytes = 0;
	}
	n_ctxs = membrane_ctxrec_generate_candidates(req.model_max_context, 0,
			ctxs, MEMBRANE_CTXREC_MAX_CANDIDATES);
	for (i = 0; i < n_ctxs; ++i)
	{
		req.candidates[i].ctx = ctxs[i];
		req.candidates[i].kv_bytes_native = real_kv_bytes(m, ctxs[i],
				GGML_TYPE_F16);
		req.candidates[i].kv_bytes_q8 = real_kv_bytes(m, ctxs[i],
				GGML_TYPE_Q8_0);
		req.candidates[i].kv_bytes_q5 = real_kv_bytes(m, ctxs[i],
				GGML_TYPE_Q5_1);
	}
	req.candidate_count = n_ctxs;
	membrane_ctxrec_resolve(&req, &res);

	printf("{\n");
	printf("  \"model_path\":\"%s\",\n", argv[1]);
	printf("  \"arch_name\":\"%s\",\n", m.arch_name);
	printf("  \"hparams_available\":%s,\n",
		m.hparams_available ? "true" : "false");
	printf("  \"model_max_context_available\":%s,\n",
		m.model_max_context_available ? "true" : "false");
	printf("  \"model_max_context\":%llu,\n",
		(unsigned long long)m.model_max_context);
	printf("  \"gpu_device_found\":%s,\n", gpu_index >= 0 ? "true" : "false");
	if (gpu_index >= 0)
		printf("  \"gpu_device_name\":\"%s\",\n", devices[gpu_index].name);
	printf("  \"device_free_bytes\":%llu,\n",
		(unsigned long long)req.device_free_bytes);
	printf("  \"device_total_bytes\":%llu,\n",
		(unsigned long long)req.device_total_bytes);
	printf("  \"precision_request\":%d,\n", req.precision_request);
	printf("  \"candidates\":[");
	for (i = 0; i < n_ctxs; ++i)
		printf("%s%llu", i == 0 ? "" : ",", (unsigned long long)ctxs[i]);
	printf("],\n");
	printf("  \"status\":\"%s\",\n", res.status);
	printf("  \"ok\":%s,\n", res.ok ? "true" : "false");
	printf("  \"hardware_fit_context\":%llu,\n",
		(unsigned long long)res.hardware_fit_context);
	printf("  \"recommended_context\":%llu,\n",
		(unsigned long long)res.recommended_context);
	printf("  \"recommendation_policy\":\"%s\",\n", res.recommendation_policy);
	printf("  \"host_memory_unvalidated\":%s,\n",
		res.host_memory_unvalidated ? "true" : "false");
	printf("  \"explanation\":\"%s\",\n", res.explanation);
	printf("  \"evaluated\":[\n");
	for (i = 0; i < res.evaluated_count; ++i)
	{
		print_evaluated(&res.evaluated[i]);
		printf("%s\n", i + 1 < res.evaluated_count ? "," : "");
	}
	printf("  ]\n");
	printf("}\n");
	return (0);
}
