#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kv_residency_policy.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

static void	test_default_mode_produces_no_plan(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_DEFAULT,
			30, 6 * MIB, 4 * GIB, 4 * GIB, 500 * MIB, 0, &r) == 1,
		"default mode always resolves ok");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_KV_PLACEMENT_REASON_DEFAULT_UNCHANGED)
		== 0, "default mode reports the unchanged reason");
	TEST_ASSERT(r.gpu_kv_layers == 0 && r.cpu_kv_layers == 0,
		"default mode never populates a layer split -- caller must not "
		"consult it");
}

static void	test_all_gpu_fits(void)
{
	membrane_kv_residency_result_t	r;

	/* 4 GiB device, 3.5 GiB free, 500 MiB weights, 30 layers * 2 MiB
	 * KV = 60 MiB total KV -- comfortably fits after the ~614 MiB
	 * reserve (15% of 4 GiB) and small margin. */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			30, 2 * MIB, (uint64_t)(3.5 * GIB), 4 * GIB, 500 * MIB, 0,
			&r) == 1, "gpu mode resolves when KV comfortably fits");
	TEST_ASSERT(r.ok == 1, "ok reflects success");
	TEST_ASSERT(r.gpu_kv_layers == 30 && r.cpu_kv_layers == 0,
		"all 30 layers placed on GPU");
	TEST_ASSERT(r.gpu_kv_bytes == 60 * MIB, "gpu_kv_bytes is exact");
	TEST_ASSERT(r.cpu_kv_bytes == 0, "cpu_kv_bytes is zero");
	TEST_ASSERT(r.total_kv_bytes == r.gpu_kv_bytes + r.cpu_kv_bytes,
		"GPU_KV_BYTES + CPU_KV_BYTES == TOTAL_KV_BYTES");
	TEST_ASSERT(r.layer_on_gpu[0] == 1 && r.layer_on_gpu[29] == 1,
		"every layer marked GPU-resident in the map");
}

static void	test_split_fit_deterministic_order(void)
{
	membrane_kv_residency_result_t	r;

	/* Constrained device: force a partial split and check the
	 * deterministic ascending-layer-index-first order (Section 9). */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			30, 8 * MIB, 900 * MIB, 2 * GIB, 100 * MIB, 0, &r) == 1,
		"auto resolves under a constrained budget");
	TEST_ASSERT(r.ok == 1, "ok reflects success");
	TEST_ASSERT(r.gpu_kv_layers > 0 && r.gpu_kv_layers < 30,
		"a genuine partial split occurred (test setup sanity)");
	TEST_ASSERT(r.gpu_kv_layers + r.cpu_kv_layers == 30,
		"gpu+cpu layer counts sum to n_layer");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_KV_PLACEMENT_REASON_AUTO_SPLIT)
		== 0, "reason reflects a real split");
	{
		int32_t	i;
		int32_t	last_gpu = -1;
		int32_t	first_cpu = -1;

		i = 0;
		while (i < 30)
		{
			if (r.layer_on_gpu[i])
				last_gpu = i;
			else if (first_cpu < 0)
				first_cpu = i;
			i++;
		}
		TEST_ASSERT(last_gpu < first_cpu,
			"every GPU-resident layer has a lower index than every "
			"CPU-resident layer (ascending, GPU-first order)");
	}
	TEST_ASSERT(r.gpu_kv_bytes + r.cpu_kv_bytes == r.total_kv_bytes,
		"byte accounting exact under a split");
}

static void	test_all_cpu_required(void)
{
	membrane_kv_residency_result_t	r;

	/* Weights alone already consume essentially the whole budget --
	 * AUTO must fall back to all-CPU-KV, not fail. */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			30, 8 * MIB, 620 * MIB, 2 * GIB, 600 * MIB, 0, &r) == 1,
		"auto resolves even when no GPU KV budget remains");
	TEST_ASSERT(r.ok == 1, "ok reflects success (never a failure for "
		"lack of GPU KV room)");
	TEST_ASSERT(r.gpu_kv_layers == 0 && r.cpu_kv_layers == 30,
		"falls back to all-CPU-KV");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_KV_PLACEMENT_REASON_CPU_FULL)
		== 0, "reason reflects the all-CPU outcome");
}

static void	test_impossible_gpu_budget_fails_closed(void)
{
	membrane_kv_residency_result_t	r;

	/* Explicit --kv-placement gpu, but there is no room at all. */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			30, 8 * MIB, 620 * MIB, 2 * GIB, 600 * MIB, 0, &r) == 0,
		"gpu mode fails closed rather than silently placing a partial "
		"plan");
	TEST_ASSERT(r.ok == 0, "ok reflects failure");
	TEST_ASSERT(strstr(r.reason,
			MEMBRANE_KV_PLACEMENT_REASON_BUDGET_INSUFFICIENT) != NULL,
		"reason names the stable BUDGET_INSUFFICIENT code");
}

static void	test_exact_boundary(void)
{
	membrane_kv_residency_result_t	r;
	uint64_t	reserve;
	uint64_t	margin;
	uint64_t	free_bytes;

	/* Hand-computed exact boundary: 2 GiB total -> 15%% pct reserve is
	 * only ~300 MiB, LESS than the 512 MiB fixed floor, so reserve is
	 * the 512 MiB floor (matches gpu_policy.c's reserve_for() -- pct
	 * only wins once total device memory exceeds 512MiB/0.15 ~= 3.4
	 * GiB). weights=0, kv/layer=1 MiB, 10 layers -> total KV=10 MiB;
	 * margin=max(64MiB fixed,5%% of the 10 MiB GPU-mode candidate=0.5
	 * MiB)=64 MiB fixed floor. Compute free so budget_for_kv_bytes
	 * lands EXACTLY on 10 MiB: free = reserve + margin + 10MiB. */
	reserve = (uint64_t)512 * MIB;
	margin = MEMBRANE_KV_RESIDENCY_MARGIN_FIXED_BYTES;
	free_bytes = reserve + margin + 10 * MIB;
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			10, 1 * MIB, free_bytes, 2 * GIB, 0, 0, &r) == 1,
		"gpu mode succeeds exactly at the computed boundary");
	TEST_ASSERT(r.gpu_kv_layers == 10, "all layers fit at the exact "
		"boundary");
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			10, 1 * MIB, free_bytes - 1, 2 * GIB, 0, 0, &r) == 0,
		"gpu mode fails one byte below the exact boundary");
}

static void	test_reserve_matches_gpu_policy_convention(void)
{
	membrane_kv_residency_result_t	r;

	/* 4 GiB total: 15%% = 614400 KiB = 600 MiB > 512 MiB fixed floor,
	 * so the percentage reserve applies (same formula as
	 * gpu_policy.c's reserve_for()). */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_CPU,
			1, 1 * MIB, 1 * GIB, 4 * GIB, 0, 0, &r) == 1, "resolves ok");
	TEST_ASSERT(r.safety_reserve_bytes == (uint64_t)4 * GIB / 100 * 15,
		"reserve follows the documented max(fixed, pct) formula");
}

static void	test_runtime_margin_applied(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			1, 1 * MIB, 1 * GIB, 4 * GIB, 0, 0, &r) == 1, "resolves ok");
	TEST_ASSERT(r.runtime_margin_bytes >=
		MEMBRANE_KV_RESIDENCY_MARGIN_FIXED_BYTES,
		"runtime margin is never below the documented fixed floor");
}

static void	test_single_layer_model(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			1, 1 * MIB, (uint64_t)(3.5 * GIB), 4 * GIB, 0, 0, &r) == 1,
		"a 1-layer model resolves");
	TEST_ASSERT(r.gpu_kv_layers == 1 && r.cpu_kv_layers == 0,
		"the single layer is placed GPU-resident");
}

static void	test_zero_layers_rejected(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			0, 1 * MIB, 4 * GIB, 4 * GIB, 0, 0, &r) == 0,
		"zero layers is an invalid config, not an empty-but-ok plan");
	TEST_ASSERT(strstr(r.reason,
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG) != NULL,
		"reason names INVALID_CONFIG");
}

static void	test_negative_layers_rejected(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			-1, 1 * MIB, 4 * GIB, 4 * GIB, 0, 0, &r) == 0,
		"negative layer count is rejected");
}

static void	test_over_max_layers_rejected(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			(int32_t)MEMBRANE_KV_RESIDENCY_MAX_LAYERS + 1, 1 * MIB,
			4 * GIB, 4 * GIB, 0, 0, &r) == 0,
		"a layer count above the fixed map size is rejected, not "
		"silently truncated");
}

static void	test_invalid_placement_mode_rejected(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(99, 10, 1 * MIB, 4 * GIB,
			4 * GIB, 0, 0, &r) == 0, "an unknown placement mode is "
		"rejected");
	TEST_ASSERT(strstr(r.reason,
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG) != NULL,
		"reason names INVALID_CONFIG");
}

static void	test_deterministic_repeat(void)
{
	membrane_kv_residency_result_t	r1;
	membrane_kv_residency_result_t	r2;

	membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO, 30, 8 * MIB,
		900 * MIB, 2 * GIB, 100 * MIB, 0, &r1);
	membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO, 30, 8 * MIB,
		900 * MIB, 2 * GIB, 100 * MIB, 0, &r2);
	TEST_ASSERT(memcmp(&r1, &r2, sizeof(r1)) == 0,
		"identical inputs produce a byte-identical result (deterministic "
		"map, no hidden state)");
}

static void	test_zero_kv_bytes_per_layer_treated_as_free(void)
{
	membrane_kv_residency_result_t	r;

	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			30, 0, 1 * GIB, 4 * GIB, 0, 0, &r) == 1,
		"a zero-byte-per-layer input resolves without a divide-by-zero");
	TEST_ASSERT(r.gpu_kv_layers == 30, "all layers placed GPU-resident "
		"when KV costs nothing");
}

static void	test_cpu_only_backend_scenario(void)
{
	membrane_kv_residency_result_t	r;

	/* CPU-only build/host: device_free/total are both 0 (no GPU
	 * queried at all) -- CPU mode must still succeed trivially, GPU
	 * mode must fail closed rather than divide-by-zero or crash. */
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_CPU,
			30, 8 * MIB, 0, 0, 0, 0, &r) == 1,
		"cpu mode succeeds with zero device memory (CPU-only host)");
	TEST_ASSERT(r.cpu_kv_layers == 30, "all layers CPU-resident");
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_GPU,
			30, 8 * MIB, 0, 0, 0, 0, &r) == 0,
		"gpu mode fails closed with zero device memory, no crash");
}

static void	test_null_out_pointer_is_safe(void)
{
	TEST_ASSERT(membrane_kv_residency_resolve(MEMBRANE_KV_PLACEMENT_AUTO,
			30, 8 * MIB, 4 * GIB, 4 * GIB, 0, 0, NULL) == 0,
		"a NULL out pointer is rejected, not dereferenced");
}

static void	test_mode_name_strings(void)
{
	TEST_ASSERT(strcmp(membrane_kv_placement_mode_name(
			MEMBRANE_KV_PLACEMENT_DEFAULT), "default") == 0, "default name");
	TEST_ASSERT(strcmp(membrane_kv_placement_mode_name(
			MEMBRANE_KV_PLACEMENT_GPU), "gpu") == 0, "gpu name");
	TEST_ASSERT(strcmp(membrane_kv_placement_mode_name(
			MEMBRANE_KV_PLACEMENT_CPU), "cpu") == 0, "cpu name");
	TEST_ASSERT(strcmp(membrane_kv_placement_mode_name(
			MEMBRANE_KV_PLACEMENT_AUTO), "auto") == 0, "auto name");
}

int	main(void)
{
	test_default_mode_produces_no_plan();
	test_all_gpu_fits();
	test_split_fit_deterministic_order();
	test_all_cpu_required();
	test_impossible_gpu_budget_fails_closed();
	test_exact_boundary();
	test_reserve_matches_gpu_policy_convention();
	test_runtime_margin_applied();
	test_single_layer_model();
	test_zero_layers_rejected();
	test_negative_layers_rejected();
	test_over_max_layers_rejected();
	test_invalid_placement_mode_rejected();
	test_deterministic_repeat();
	test_zero_kv_bytes_per_layer_treated_as_free();
	test_cpu_only_backend_scenario();
	test_null_out_pointer_is_safe();
	test_mode_name_strings();
	printf("test_kv_residency_policy: all tests passed\n");
	return (0);
}
