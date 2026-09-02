#include <stdint.h>
#include <string.h>

#include "host_memory_guard.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

static void	fill_base(membrane_host_guard_request_t *req)
{
	memset(req, 0, sizeof(*req));
	req->host_total_bytes = 8 * GIB;
	req->host_available_bytes = 4 * GIB;
	req->host_available_known = 1;
}

/* 1. Zero host requirement -- trivially ok, no reserve applied. */
static void	test_zero_requirement(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"zero requirement is trivially ok");
	TEST_ASSERT(res.required_bytes == 0, "required_bytes is 0");
	TEST_ASSERT(res.reserve_bytes == 0, "no reserve applied to nothing");
	TEST_ASSERT(strcmp(res.reason_code,
			MEMBRANE_HOST_GUARD_REASON_ZERO_REQUIREMENT) == 0,
		"ZERO_REQUIREMENT reason code");
}

/* 2. Weight-only requirement. */
static void	test_weight_only(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_weight_bytes = 200 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"weight-only requirement fits ample host RAM");
	TEST_ASSERT(res.required_bytes == 200 * MIB, "required == weight bytes");
	TEST_ASSERT(res.reserve_bytes == MEMBRANE_HOST_RESERVE_FIXED_BYTES,
		"fixed floor dominates 10% of 200 MiB (20 MiB < 64 MiB)");
}

/* 3. KV-only requirement. */
static void	test_kv_only(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_kv_bytes = 50 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"KV-only requirement fits");
	TEST_ASSERT(res.required_bytes == 50 * MIB, "required == kv bytes");
}

/* 4. Weights + KV combined, and the percentage term dominating the
 * fixed floor for a large-enough combined requirement (2.7 GiB * 10%
 * = 270 MiB > 256 MiB fixed floor). */
static void	test_weights_plus_kv_pct_dominates(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;
	uint64_t						combined = 2500 * MIB + 200 * MIB;

	fill_base(&req);
	req.host_weight_bytes = 2500 * MIB;
	req.host_kv_bytes = 200 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"2.7 GiB combined requirement fits 4 GiB available");
	TEST_ASSERT(res.required_bytes == combined, "required == weight + kv");
	TEST_ASSERT(res.reserve_bytes == combined / 10,
		"10%% of 2.7 GiB (270 MiB) exceeds the 256 MiB fixed floor");
}

/* 5. Enough memory, ample headroom. */
static void	test_enough_memory(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_weight_bytes = 100 * MIB;
	req.host_kv_bytes = 10 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"resolves ok");
	TEST_ASSERT(res.available_after_reserve
			== req.host_available_bytes - res.reserve_bytes,
		"available_after_reserve arithmetic is exact");
}

/* 6. Exact boundary: required + reserve == available exactly. */
static void	test_exact_boundary(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;
	uint64_t						required = 100 * MIB;
	uint64_t						reserve =
			membrane_host_memory_reserve_bytes(required);

	fill_base(&req);
	req.host_weight_bytes = required;
	req.host_available_bytes = required + reserve;	/* exactly enough */
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"required + reserve == available is still a fit (<=, not <)");
	req.host_available_bytes = required + reserve - 1;	/* one byte short */
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 0,
		"one byte short of required + reserve is NOT a fit");
}

/* 7. Insufficient memory. */
static void	test_insufficient_memory(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_weight_bytes = 3900 * MIB;	/* close to the 4 GiB available */
	req.host_kv_bytes = 200 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 0,
		"insufficient memory fails closed");
	TEST_ASSERT(strcmp(res.reason_code,
			MEMBRANE_HOST_GUARD_REASON_INSUFFICIENT) == 0,
		"HOST_MEMORY_INSUFFICIENT reason code");
}

/* 8. Reserve alone (added to a small requirement) exceeds available. */
static void	test_reserve_larger_than_available(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_available_bytes = 32 * MIB;	/* below the 64 MiB fixed floor */
	req.host_weight_bytes = 1 * MIB;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 0,
		"tiny available memory cannot even cover the reserve floor");
	TEST_ASSERT(res.available_after_reserve == 0,
		"available_after_reserve clamps to 0, never underflows");
}

/* 9. Unknown availability -- never assumed infinite. */
static void	test_unknown_availability_fails_closed(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_available_known = 0;
	req.host_weight_bytes = 1 * MIB;	/* tiny -- would trivially fit */
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 0,
		"unknown availability fails closed even for a tiny requirement");
	TEST_ASSERT(strcmp(res.reason_code, MEMBRANE_HOST_GUARD_REASON_UNKNOWN)
		== 0, "HOST_MEMORY_UNKNOWN reason code");
}

/* 9b. Unknown availability with zero requirement is still trivially ok
 * -- there is nothing to validate against the unknown value at all. */
static void	test_unknown_availability_zero_requirement_still_ok(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	req.host_available_known = 0;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 1,
		"zero requirement never consults availability at all");
}

/* 10. Arithmetic overflow -- checked, saturating, never wraps. */
static void	test_overflow_never_wraps(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);	/* host_available_bytes stays a realistic 4 GiB */
	req.host_weight_bytes = UINT64_MAX;
	req.host_kv_bytes = UINT64_MAX;
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, &res) == 0,
		"saturated required_bytes (UINT64_MAX) cannot fit a realistic "
		"4 GiB available -- fails closed, does not crash, and (critically)"
		" does not wrap to a small value that would incorrectly fit");
	TEST_ASSERT(res.required_bytes == UINT64_MAX,
		"required_bytes saturates rather than wrapping to a small value");

	TEST_ASSERT(membrane_host_memory_reserve_bytes(UINT64_MAX) > 0,
		"reserve formula on UINT64_MAX does not wrap to 0");
}

/* 11. Deterministic repeat. */
static void	test_deterministic_repeat(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res1;
	membrane_host_guard_result_t	res2;

	fill_base(&req);
	req.host_weight_bytes = 300 * MIB;
	req.host_kv_bytes = 20 * MIB;
	membrane_host_memory_guard_resolve(&req, &res1);
	membrane_host_memory_guard_resolve(&req, &res2);
	TEST_ASSERT(memcmp(&res1, &res2, sizeof(res1)) == 0,
		"identical inputs produce a byte-for-byte identical result");
}

static void	test_null_safety(void)
{
	membrane_host_guard_request_t	req;
	membrane_host_guard_result_t	res;

	fill_base(&req);
	TEST_ASSERT(membrane_host_memory_guard_resolve(NULL, &res) == 0,
		"NULL request never crashes, just fails");
	TEST_ASSERT(strcmp(res.reason_code,
			MEMBRANE_HOST_GUARD_REASON_INVALID_CONFIG) == 0,
		"INVALID_CONFIG reason code for a NULL request");
	TEST_ASSERT(membrane_host_memory_guard_resolve(&req, NULL) == 0,
		"NULL out pointer never crashes, just fails");
}

int	main(void)
{
	test_zero_requirement();
	test_weight_only();
	test_kv_only();
	test_weights_plus_kv_pct_dominates();
	test_enough_memory();
	test_exact_boundary();
	test_insufficient_memory();
	test_reserve_larger_than_available();
	test_unknown_availability_fails_closed();
	test_unknown_availability_zero_requirement_still_ok();
	test_overflow_never_wraps();
	test_deterministic_repeat();
	test_null_safety();
	printf("test_host_memory_guard: all tests passed\n");
	return (0);
}
