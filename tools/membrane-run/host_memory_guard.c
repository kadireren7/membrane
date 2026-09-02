#include "host_memory_guard.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Same checked/saturating uint64 arithmetic convention as
 * joint_planner.c's own sat_mul_u64()/sat_add_u64() -- duplicated here
 * (not shared via a common header) because both modules are
 * deliberately small, dependency-free, single-file pure-arithmetic
 * units, matching this project's existing *_policy.h convention. */
static uint64_t	sat_add_u64(uint64_t a, uint64_t b)
{
	if (a > UINT64_MAX - b)
		return (UINT64_MAX);
	return (a + b);
}

static uint64_t	sat_mul_u64(uint64_t a, uint64_t b)
{
	if (a != 0 && b > UINT64_MAX / a)
		return (UINT64_MAX);
	return (a * b);
}

uint64_t	membrane_host_memory_reserve_bytes(uint64_t host_resident_bytes)
{
	uint64_t	pct_bytes;

	if (host_resident_bytes == 0)
		return (0);
	pct_bytes = sat_mul_u64(host_resident_bytes, MEMBRANE_HOST_RESERVE_PCT)
			/ 100;
	return (pct_bytes > MEMBRANE_HOST_RESERVE_FIXED_BYTES
			? pct_bytes : MEMBRANE_HOST_RESERVE_FIXED_BYTES);
}

static void	set_result(membrane_host_guard_result_t *out, int ok,
				const char *code, const char *detail)
{
	out->ok = ok;
	snprintf(out->reason_code, sizeof(out->reason_code), "%s", code);
	if (detail != NULL && detail[0] != '\0')
		snprintf(out->reason, sizeof(out->reason), "%s: %s", code, detail);
	else
		snprintf(out->reason, sizeof(out->reason), "%s", code);
}

int	membrane_host_memory_guard_resolve(
			const membrane_host_guard_request_t *req,
			membrane_host_guard_result_t *out)
{
	uint64_t	required;
	uint64_t	reserve;
	uint64_t	needed;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	if (req == NULL)
	{
		set_result(out, 0, MEMBRANE_HOST_GUARD_REASON_INVALID_CONFIG,
			"null request");
		return (0);
	}
	required = sat_add_u64(req->host_weight_bytes, req->host_kv_bytes);
	out->required_bytes = required;
	if (required == 0)
	{
		set_result(out, 1, MEMBRANE_HOST_GUARD_REASON_ZERO_REQUIREMENT,
			"nothing host-resident in this guard's scope -- see "
			"host_memory_guard.h's own top comment on what this does "
			"not cover");
		return (1);
	}
	reserve = membrane_host_memory_reserve_bytes(required);
	out->reserve_bytes = reserve;
	if (!req->host_available_known)
	{
		set_result(out, 0, MEMBRANE_HOST_GUARD_REASON_UNKNOWN,
			"host memory availability could not be read -- never "
			"assumed infinite");
		return (0);
	}
	needed = sat_add_u64(required, reserve);
	out->available_after_reserve = req->host_available_bytes > reserve
			? req->host_available_bytes - reserve : 0;
	if (needed > req->host_available_bytes)
	{
		char	detail[192];

		snprintf(detail, sizeof(detail), "required=%llu reserve=%llu "
			"available=%llu", (unsigned long long)required,
			(unsigned long long)reserve,
			(unsigned long long)req->host_available_bytes);
		set_result(out, 0, MEMBRANE_HOST_GUARD_REASON_INSUFFICIENT, detail);
		return (0);
	}
	set_result(out, 1, MEMBRANE_HOST_GUARD_REASON_FIT, "");
	return (1);
}
