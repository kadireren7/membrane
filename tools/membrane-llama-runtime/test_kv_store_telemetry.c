#include "kv_store_telemetry.h"
#include "test_helpers.h"

static void	test_total_bytes_null_is_zero(void)
{
	TEST_ASSERT(membrane_kv_store_total_bytes(NULL) == 0,
		"NULL input returns 0, never garbage");
}

/* F16 (native): 2 bytes/element, no block structure -- bytes_per_token
 * for a 64-wide head is simply 64*2 = 128. */
static void	test_total_bytes_native_f16(void)
{
	membrane_kv_store_bytes_t	b;
	uint64_t					total;

	b.n_layer = 30;
	b.kv_size = 2048;
	b.bytes_per_token_k = 128;	/* 64 elems * 2 bytes (F16) */
	b.bytes_per_token_v = 128;
	total = membrane_kv_store_total_bytes(&b);
	TEST_ASSERT(total == 30ULL * 2048ULL * 256ULL,
		"native bytes = n_layer * kv_size * (k_row + v_row)");
}

/* Q8_0: 32-element blocks, 34 bytes/block (2-byte F16 scale + 32 int8
 * values) -- a 64-wide head is 2 blocks = 68 bytes/token. */
static void	test_total_bytes_q8(void)
{
	membrane_kv_store_bytes_t	b;
	uint64_t					total;

	b.n_layer = 30;
	b.kv_size = 2048;
	b.bytes_per_token_k = 68;	/* 2 blocks * 34 bytes (Q8_0) */
	b.bytes_per_token_v = 68;
	total = membrane_kv_store_total_bytes(&b);
	TEST_ASSERT(total == 30ULL * 2048ULL * 136ULL,
		"q8 bytes = n_layer * kv_size * (k_row + v_row)");
}

/* The whole point of Phase 7: q8 storage must be measurably smaller
 * than native for identical layer/kv_size/head-width parameters --
 * this is the byte-accounting side of that claim (real RSS is the
 * other, independently measured side, see main.cpp). */
static void	test_q8_smaller_than_native_same_shape(void)
{
	membrane_kv_store_bytes_t	native;
	membrane_kv_store_bytes_t	q8;
	uint64_t					native_total;
	uint64_t					q8_total;
	double						ratio;

	native.n_layer = 30;
	native.kv_size = 2048;
	native.bytes_per_token_k = 128;
	native.bytes_per_token_v = 128;
	q8 = native;
	q8.bytes_per_token_k = 68;
	q8.bytes_per_token_v = 68;
	native_total = membrane_kv_store_total_bytes(&native);
	q8_total = membrane_kv_store_total_bytes(&q8);
	TEST_ASSERT(q8_total < native_total,
		"q8 allocated bytes strictly less than native for the same "
		"layer/kv_size/head shape");
	ratio = (double)native_total / (double)q8_total;
	TEST_ASSERT(ratio > 1.8 && ratio < 1.95,
		"compression ratio matches the known Q8_0-vs-F16 ratio "
		"(34/32 bytes/elem vs 2 bytes/elem, ~1.88x), not some other "
		"unrelated number");
}

/* Phase 10C: Q5_1, 32-element blocks, 24 bytes/block (2x ggml_half
 * scale+min + 4-byte qh[] 5th-bit group + 16 bytes packed nibbles) --
 * a 64-wide head is 2 blocks = 48 bytes/token. */
static void	test_total_bytes_q5(void)
{
	membrane_kv_store_bytes_t	b;
	uint64_t					total;

	b.n_layer = 30;
	b.kv_size = 2048;
	b.bytes_per_token_k = 48;	/* 2 blocks * 24 bytes (Q5_1) */
	b.bytes_per_token_v = 48;
	total = membrane_kv_store_total_bytes(&b);
	TEST_ASSERT(total == 30ULL * 2048ULL * 96ULL,
		"q5 bytes = n_layer * kv_size * (k_row + v_row)");
}

/* Phase 10C Section 21: q5 < q8 < native ordering, for the same
 * layer/kv_size/head shape -- proves the byte totals are strictly
 * ordered, not accidentally aliased (e.g. a copy-paste bug giving q5
 * the same byte-per-token constant as q8). */
static void	test_q5_between_q8_and_native_same_shape(void)
{
	membrane_kv_store_bytes_t	shape;
	uint64_t					native_total;
	uint64_t					q8_total;
	uint64_t					q5_total;

	shape.n_layer = 30;
	shape.kv_size = 2048;

	shape.bytes_per_token_k = shape.bytes_per_token_v = 128;
	native_total = membrane_kv_store_total_bytes(&shape);
	shape.bytes_per_token_k = shape.bytes_per_token_v = 68;
	q8_total = membrane_kv_store_total_bytes(&shape);
	shape.bytes_per_token_k = shape.bytes_per_token_v = 48;
	q5_total = membrane_kv_store_total_bytes(&shape);

	TEST_ASSERT(q5_total < q8_total && q8_total < native_total,
		"q5 < q8 < native for the same layer/kv_size/head shape, no "
		"two modes accidentally alias the same bytes");
}

static void	test_total_bytes_scales_with_ctx_size(void)
{
	membrane_kv_store_bytes_t	small;
	membrane_kv_store_bytes_t	big;

	small.n_layer = 30;
	small.kv_size = 512;
	small.bytes_per_token_k = 68;
	small.bytes_per_token_v = 68;
	big = small;
	big.kv_size = 4096;
	TEST_ASSERT(membrane_kv_store_total_bytes(&big)
			== membrane_kv_store_total_bytes(&small) * 8,
		"bytes scale linearly with kv_size (context size) -- "
		"512 -> 4096 is exactly 8x");
}

static void	test_read_rss_populates_ru_maxrss(void)
{
	membrane_kv_store_rss_t	r;

	membrane_kv_store_read_rss(&r);
	TEST_ASSERT(r.ru_maxrss_kb > 0,
		"getrusage(RUSAGE_SELF).ru_maxrss is always available (POSIX) "
		"regardless of /proc availability");
}

static void	test_rss_max_is_componentwise(void)
{
	membrane_kv_store_rss_t	a;
	membrane_kv_store_rss_t	b;
	membrane_kv_store_rss_t	out;

	a.vm_rss_kb = 100;
	a.vm_hwm_kb = 500;
	a.ru_maxrss_kb = 300;
	a.proc_status_ok = 1;
	b.vm_rss_kb = 900;
	b.vm_hwm_kb = 200;
	b.ru_maxrss_kb = 700;
	b.proc_status_ok = 1;
	membrane_kv_store_rss_max(&a, &b, &out);
	TEST_ASSERT(out.vm_hwm_kb == 500,
		"max picks the larger vm_hwm_kb component-wise");
	TEST_ASSERT(out.ru_maxrss_kb == 700,
		"max picks the larger ru_maxrss_kb component-wise");
}

static void	test_rss_max_null_safe(void)
{
	membrane_kv_store_rss_t	a;
	membrane_kv_store_rss_t	out;

	membrane_kv_store_read_rss(&a);
	membrane_kv_store_rss_max(NULL, &a, &out);
	membrane_kv_store_rss_max(&a, NULL, &out);
	membrane_kv_store_rss_max(&a, &a, NULL);
	TEST_ASSERT(1, "NULL args never crash membrane_kv_store_rss_max");
}

int	main(void)
{
	test_total_bytes_null_is_zero();
	test_total_bytes_native_f16();
	test_total_bytes_q8();
	test_q8_smaller_than_native_same_shape();
	test_total_bytes_q5();
	test_q5_between_q8_and_native_same_shape();
	test_total_bytes_scales_with_ctx_size();
	test_read_rss_populates_ru_maxrss();
	test_rss_max_is_componentwise();
	test_rss_max_null_safe();
	printf("test_kv_store_telemetry: all tests passed\n");
	return (0);
}
