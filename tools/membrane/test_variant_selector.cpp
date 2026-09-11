#include <cstdio>

#include "variant_selector.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D2: pure unit tests for variant_selector.h --
 * synthetic hardware facts (no real /proc read), the same testable-
 * without-a-real-host pattern host_memory_guard.h's own test suite
 * already established.
 */

static membrane_catalog_family_t	fake_family(void)
{
	membrane_catalog_family_t	f;

	f.name = "fake-family";
	f.arch = "llama";
	membrane_catalog_variant_t	q4;

	q4.quant = "Q4_K_M";
	q4.size_bytes = 100ull * 1024 * 1024;	/* 100 MiB */
	f.variants.push_back(q4);

	membrane_catalog_variant_t	q8;

	q8.quant = "Q8_0";
	q8.size_bytes = 150ull * 1024 * 1024;	/* 150 MiB */
	f.variants.push_back(q8);

	membrane_catalog_variant_t	f16;

	f16.quant = "F16";
	f16.size_bytes = 280ull * 1024 * 1024;	/* 280 MiB */
	f.variants.push_back(f16);
	return (f);
}

static void	test_selects_largest_fitting_variant(void)
{
	membrane_catalog_family_t	f = fake_family();
	membrane_variant_selector_input_t	hw;

	/* Real host_memory_guard.h reserve policy: 256 MiB fixed + 10% of
	 * resident bytes. Plenty of headroom for every variant here. */
	hw.host_total_bytes = 8ull * 1024 * 1024 * 1024;
	hw.host_available_bytes = 4ull * 1024 * 1024 * 1024;	/* 4 GiB free */
	hw.host_available_known = true;

	std::vector<membrane_variant_fit_t>	considered;
	const membrane_catalog_variant_t	*best
			= membrane_select_variant(f, hw, &considered);

	TEST_ASSERT(best != NULL, "a variant is selected");
	TEST_ASSERT(best->quant == "F16",
		"the LARGEST fitting variant is preferred, never the smallest");
	TEST_ASSERT(considered.size() == 3, "every variant is reported");
	for (const auto &c : considered)
		TEST_ASSERT(c.fits, "every variant fits under 4 GiB available");
}

static void	test_selects_smaller_variant_under_tight_memory(void)
{
	membrane_catalog_family_t	f = fake_family();
	membrane_variant_selector_input_t	hw;

	/* Just enough for Q4_K_M (100 MiB + 256 MiB reserve + 10% ~ 366
	 * MiB) but not Q8_0 (150 MiB + 256 + 15% ~ 421 MiB) or F16. */
	hw.host_total_bytes = 8ull * 1024 * 1024 * 1024;
	hw.host_available_bytes = 400ull * 1024 * 1024;
	hw.host_available_known = true;

	std::vector<membrane_variant_fit_t>	considered;
	const membrane_catalog_variant_t	*best
			= membrane_select_variant(f, hw, &considered);

	TEST_ASSERT(best != NULL, "a variant is selected");
	TEST_ASSERT(best->quant == "Q4_K_M",
		"only the smallest variant fits under tight real memory");
}

static void	test_no_variant_fits_returns_null_with_reasons(void)
{
	membrane_catalog_family_t	f = fake_family();
	membrane_variant_selector_input_t	hw;

	hw.host_total_bytes = 512ull * 1024 * 1024;
	hw.host_available_bytes = 10ull * 1024 * 1024;	/* 10 MiB -- nothing fits */
	hw.host_available_known = true;

	std::vector<membrane_variant_fit_t>	considered;
	const membrane_catalog_variant_t	*best
			= membrane_select_variant(f, hw, &considered);

	TEST_ASSERT(best == NULL, "no variant is selected -- never picks an "
		"oversized one");
	TEST_ASSERT(considered.size() == 3,
		"every variant's own real fit/no-fit reason is still reported "
		"(Section 11: fail with alternatives)");
	for (const auto &c : considered)
	{
		TEST_ASSERT(!c.fits, "every variant correctly reports not fitting");
		TEST_ASSERT(!c.reason_code.empty(), "a real reason code is present");
	}
}

static void	test_unknown_available_memory_fails_closed(void)
{
	membrane_catalog_family_t	f = fake_family();
	membrane_variant_selector_input_t	hw;

	hw.host_total_bytes = 0;
	hw.host_available_bytes = 0;
	hw.host_available_known = false;	/* /proc unreadable */

	std::vector<membrane_variant_fit_t>	considered;
	const membrane_catalog_variant_t	*best
			= membrane_select_variant(f, hw, &considered);

	TEST_ASSERT(best == NULL, "unknown available memory never assumed "
		"to fit -- fails closed, matching host_memory_guard.h's own "
		"HOST_MEMORY_UNKNOWN contract");
	for (const auto &c : considered)
		TEST_ASSERT(c.reason_code == "HOST_MEMORY_UNKNOWN",
			"reason code is HOST_MEMORY_UNKNOWN for every variant");
}

static void	test_deterministic(void)
{
	membrane_catalog_family_t	f = fake_family();
	membrane_variant_selector_input_t	hw;

	hw.host_total_bytes = 8ull * 1024 * 1024 * 1024;
	hw.host_available_bytes = 1ull * 1024 * 1024 * 1024;
	hw.host_available_known = true;

	const membrane_catalog_variant_t	*r1
			= membrane_select_variant(f, hw, NULL);
	const membrane_catalog_variant_t	*r2
			= membrane_select_variant(f, hw, NULL);

	TEST_ASSERT(r1 != NULL && r2 != NULL && r1->quant == r2->quant,
		"identical inputs always produce an identical result");
}

/*
 * Post-v1 product-polish, PR 1: regression coverage for the real
 * v1.0.0 user-session bug (`membrane use` previewed a variant as
 * HOST_MEMORY_FIT, then its own internal re-dispatch of `membrane
 * model install` got HOST_MEMORY_INSUFFICIENT for the SAME variant and
 * silently treated that as "the user forced it") -- see variant_
 * selector.h's own membrane_variant_install_decide() top comment for
 * the full root-cause narrative. Pure booleans in, no real host meminfo
 * or catalog/network involved.
 */
static void	test_install_decide_proceeds_when_it_fits(void)
{
	TEST_ASSERT(membrane_variant_install_decide(true, true)
			== MEMBRANE_VARIANT_INSTALL_PROCEED,
		"an auto-selected variant that still fits proceeds with no "
		"warning");
	TEST_ASSERT(membrane_variant_install_decide(true, false)
			== MEMBRANE_VARIANT_INSTALL_PROCEED,
		"a manually-typed --quant that fits proceeds with no warning");
}

static void	test_install_decide_flags_stale_recommendation(void)
{
	TEST_ASSERT(membrane_variant_install_decide(false, true)
			== MEMBRANE_VARIANT_INSTALL_STALE,
		"an auto-selected variant that no longer fits is a STALE "
		"recommendation (available memory changed since `membrane use` "
		"checked it) -- never silently downgraded to \"the user forced "
		"it\"");
}

static void	test_install_decide_preserves_explicit_override(void)
{
	TEST_ASSERT(membrane_variant_install_decide(false, false)
			== MEMBRANE_VARIANT_INSTALL_FORCED,
		"a real, manually-typed --quant override that does not fit is "
		"still honored (FORCED, warn-and-proceed) -- explicit user "
		"override semantics are preserved exactly");
}

int	main(void)
{
	test_selects_largest_fitting_variant();
	test_selects_smaller_variant_under_tight_memory();
	test_no_variant_fits_returns_null_with_reasons();
	test_unknown_available_memory_fails_closed();
	test_deterministic();
	test_install_decide_proceeds_when_it_fits();
	test_install_decide_flags_stale_recommendation();
	test_install_decide_preserves_explicit_override();
	printf("test_variant_selector: all tests passed\n");
	return (0);
}
