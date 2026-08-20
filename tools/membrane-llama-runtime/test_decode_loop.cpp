#include <cstdio>

#include "decode_loop.h"
#include "test_helpers.h"

/* Phase 12H review fix: kv_placement_dev_override_cb must never
 * dereference a NULL layer_on_gpu, even for a non-NULL map with a
 * positive n_layer -- no real call site currently produces that
 * combination, but the callback itself must stay safe against it
 * (llama_context_params.kv_dev_override is installed once at
 * construction time and any crash there takes the whole process
 * down). */
static void	test_null_map_falls_back_to_default(void)
{
	ggml_backend_dev_t	fake_default;

	fake_default = (ggml_backend_dev_t)(void *)0x1;
	TEST_ASSERT(kv_placement_dev_override_cb(0, fake_default, NULL)
		== fake_default, "NULL map falls back to default_dev");
}

static void	test_null_layer_on_gpu_falls_back_to_default(void)
{
	membrane_kv_placement_map_t	m;
	ggml_backend_dev_t				fake_default;

	m.n_layer = 4;
	m.layer_on_gpu = NULL;
	fake_default = (ggml_backend_dev_t)(void *)0x2;
	TEST_ASSERT(kv_placement_dev_override_cb(0, fake_default, &m)
		== fake_default,
		"non-NULL map with NULL layer_on_gpu falls back to default_dev "
		"instead of crashing");
	TEST_ASSERT(kv_placement_dev_override_cb(3, fake_default, &m)
		== fake_default,
		"same fallback at the last in-range layer index");
}

static void	test_out_of_range_index_falls_back_to_default(void)
{
	uint8_t							layer_on_gpu[2] = {1, 0};
	membrane_kv_placement_map_t	m;
	ggml_backend_dev_t				fake_default;

	m.n_layer = 2;
	m.layer_on_gpu = layer_on_gpu;
	fake_default = (ggml_backend_dev_t)(void *)0x3;
	TEST_ASSERT(kv_placement_dev_override_cb(-1, fake_default, &m)
		== fake_default, "negative index falls back to default_dev");
	TEST_ASSERT(kv_placement_dev_override_cb(2, fake_default, &m)
		== fake_default,
		"index == n_layer (out of range) falls back to default_dev");
}

static void	test_gpu_resident_layer_returns_default_dev(void)
{
	uint8_t							layer_on_gpu[2] = {1, 0};
	membrane_kv_placement_map_t	m;
	ggml_backend_dev_t				fake_default;

	m.n_layer = 2;
	m.layer_on_gpu = layer_on_gpu;
	fake_default = (ggml_backend_dev_t)(void *)0x4;
	TEST_ASSERT(kv_placement_dev_override_cb(0, fake_default, &m)
		== fake_default,
		"layer_on_gpu[0]=1 (GPU-resident) returns default_dev unchanged, "
		"a genuine no-op");
}

static void	test_cpu_resident_layer_returns_cpu_device(void)
{
	uint8_t							layer_on_gpu[2] = {1, 0};
	membrane_kv_placement_map_t	m;
	ggml_backend_dev_t				fake_default;
	ggml_backend_dev_t				result;

	m.n_layer = 2;
	m.layer_on_gpu = layer_on_gpu;
	fake_default = (ggml_backend_dev_t)(void *)0x5;
	result = kv_placement_dev_override_cb(1, fake_default, &m);
	TEST_ASSERT(result != fake_default,
		"layer_on_gpu[1]=0 (CPU-resident) does not return default_dev");
	TEST_ASSERT(result == ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
		"layer_on_gpu[1]=0 returns the real CPU backend device");
}

int	main(void)
{
	test_null_map_falls_back_to_default();
	test_null_layer_on_gpu_falls_back_to_default();
	test_out_of_range_index_falls_back_to_default();
	test_gpu_resident_layer_returns_default_dev();
	test_cpu_resident_layer_returns_cpu_device();
	printf("test_decode_loop: all tests passed\n");
	return (0);
}
