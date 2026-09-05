#include <cstdio>

#include "model_catalog.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D1: pure unit tests for model_catalog.h -- no
 * network, no filesystem, exercises only the compiled-in catalog data
 * and the search/resolve/find_variant functions.
 */

static void	test_load_has_real_families(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();

	TEST_ASSERT(cat.schema_version == 1, "schema_version is 1");
	TEST_ASSERT(cat.families.size() >= 3, "at least 3 catalog families");
}

static void	test_resolve_by_canonical_name(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, "qwen2.5-1.5b-instruct");

	TEST_ASSERT(f != NULL, "resolves by canonical name");
	TEST_ASSERT(f->arch == "qwen2", "arch is qwen2");
}

static void	test_resolve_by_alias_case_insensitive(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, "QWEN2.5:1.5B");

	TEST_ASSERT(f != NULL, "resolves by alias, case-insensitively");
	TEST_ASSERT(f->name == "qwen2.5-1.5b-instruct",
		"resolves to the correct canonical family");
}

static void	test_resolve_unknown_returns_null(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, "definitely-not-a-real-model");

	TEST_ASSERT(f == NULL, "unknown name resolves to NULL, never a guess");
}

static void	test_search_empty_query_returns_everything(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	auto	results = membrane_catalog_search(cat, "");

	TEST_ASSERT(results.size() == cat.families.size(),
		"empty query returns every family");
}

static void	test_search_by_provider(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	auto	results = membrane_catalog_search(cat, "unsloth");

	TEST_ASSERT(results.size() == 2,
		"searching by provider 'unsloth' finds exactly the 2 SmolLM2 families");
}

static void	test_find_variant(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, "smollm2-135m-instruct");

	TEST_ASSERT(f != NULL, "family found");

	const membrane_catalog_variant_t	*v
			= membrane_catalog_find_variant(*f, "q4_k_m");

	TEST_ASSERT(v != NULL, "variant found, case-insensitively");
	TEST_ASSERT(v->size_bytes == 105454144, "size_bytes matches the real, "
		"verified upstream Content-Length");
	TEST_ASSERT(v->sha256.size() == 64, "sha256 is a real 64-hex-char digest");
	TEST_ASSERT(v->download_url.rfind("https://", 0) == 0,
		"download_url is always https://");
}

static void	test_find_variant_unknown_returns_null(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, "smollm2-135m-instruct");

	TEST_ASSERT(f != NULL, "family found");
	TEST_ASSERT(membrane_catalog_find_variant(*f, "Q999_BOGUS") == NULL,
		"unknown quant resolves to NULL");
}

static void	test_every_variant_has_required_fields(void)
{
	membrane_catalog_t	cat = membrane_catalog_load();

	for (const auto &f : cat.families)
	{
		TEST_ASSERT(!f.name.empty(), "family name is never empty");
		TEST_ASSERT(!f.license.empty(), "license is never empty "
			"(Section 4: every catalog model records license metadata)");
		TEST_ASSERT(!f.repo_url.empty()
			&& f.repo_url.rfind("https://", 0) == 0,
			"repo_url is always a real https:// URL");
		TEST_ASSERT(!f.variants.empty(),
			"every family has at least one variant");
		for (const auto &v : f.variants)
		{
			TEST_ASSERT(v.size_bytes > 0, "every variant has a nonzero "
				"recorded size");
			TEST_ASSERT(v.download_url.rfind("https://", 0) == 0,
				"every variant download_url is https://");
		}
	}
}

int	main(void)
{
	test_load_has_real_families();
	test_resolve_by_canonical_name();
	test_resolve_by_alias_case_insensitive();
	test_resolve_unknown_returns_null();
	test_search_empty_query_returns_everything();
	test_search_by_provider();
	test_find_variant();
	test_find_variant_unknown_returns_null();
	test_every_variant_has_required_fields();
	printf("test_model_catalog: all tests passed\n");
	return (0);
}
