#include <cstdio>
#include <string>

#ifndef _WIN32
# include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "plan_cmd.h"
#include "registry_core.h"
#include "product_cli.h"
#include "plan_v2_resolver.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone G1: `membrane plan`'s own tests -- driven directly
 * (in-process, no subprocess), same isolated-temp-directory-env-var
 * convention as test_use_cmd.cpp. Uses the real repository fixture
 * models/stories15M.gguf (Section: "lightweight real hardware probes",
 * no multi-GB download, no generation) for the installed-model path --
 * membrane_gpu_estimate_model() only reads GGUF metadata, never loads
 * tensor data, so this stays cheap on a memory-constrained host.
 */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-plan-cmd-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	rmdir_recursive(const std::string &dir)
{
	std::string	cmd = "rm -rf '" + dir + "'";
	int			rc = system(cmd.c_str());

	if (rc != 0)
		fprintf(stderr, "warning: cleanup of %s may have failed (rc=%d)\n",
			dir.c_str(), rc);
}

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		dir = make_temp_dir();
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_MODELS_INSTALL_DIR", (dir + "/models").c_str(), 1);
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_MODELS_INSTALL_DIR");
		rmdir_recursive(dir);
	}
};

static void	register_entry_as(const std::string &registry_path,
				const std::string &name, const std::string &path,
				const std::string &basename)
{
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	membrane_registry_load(registry_path, &reg, &err);
	entry.name = name;
	entry.path = path;
	entry.basename = basename;
	entry.arch_name = "";
	entry.model_max_context = 0;
	entry.file_size_bytes = 1;
	entry.file_mtime_ns = 1;
	entry.added_at_unix = 1700000000;
	membrane_registry_add(&reg, entry, &err);
	membrane_registry_save(registry_path, reg, &err);
}

static void	register_entry(const std::string &registry_path,
				const std::string &name, const std::string &path)
{
	register_entry_as(registry_path, name, path, "stories15M.gguf");
}

static std::string	real_fixture_path(void)
{
	/* Same repo-root-relative fixture every other real-GGUF test in
	 * this project uses (test_context_recommender.c's own top comment,
	 * context_recommender_dryrun.cpp). Tests run from the build
	 * directory, so this resolves relative to the source tree. */
	return (std::string(MEMBRANE_TEST_SOURCE_DIR) + "/models/stories15M.gguf");
}

/* Milestone G2: the repo's own real installed fixture that IS a known
 * catalog family member -- smollm2-135m-instruct's real F16 variant
 * (see tools/membrane/model_catalog.cpp's own catalog entry). Used to
 * exercise the real cross-variant sibling-scaling join (Part 2 of the
 * G2 task) end to end, not just with synthetic fixtures. The on-disk
 * repo fixture is named with a lowercase "f16" -- registered here under
 * the catalog's own exact-case filename ("...-F16.gguf") so
 * find_installed_catalog_variant()'s real exact-match lookup succeeds,
 * same real-world requirement plan_cmd.cpp's own top comment documents. */
static std::string	real_smollm2_f16_fixture_path(void)
{
	return (std::string(MEMBRANE_TEST_SOURCE_DIR)
			+ "/models/SmolLM2-135M-Instruct-f16.gguf");
}

static std::string	capture_stdout_of_plan_dispatch(
				const std::vector<std::string> &args, bool want_json,
				int *out_rc)
{
	fflush(stdout);
	char	tmpl[] = "/tmp/membrane-plan-cmd-test-capture-XXXXXX";
	int		fd = mkstemp(tmpl);

	TEST_ASSERT(fd >= 0, "could create a temp file to capture stdout into");
	int	saved_fd = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*out_rc = membrane_plan_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved_fd, STDOUT_FILENO);
	close(saved_fd);

	FILE		*f = fopen(tmpl, "r");
	std::string	captured;
	char		buf[4096];
	size_t		n;

	TEST_ASSERT(f != NULL, "could reopen the captured-stdout temp file");
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		captured.append(buf, n);
	fclose(f);
	remove(tmpl);
	return (captured);
}

static std::string	read_file(const std::string &path)
{
	FILE	*f = fopen(path.c_str(), "rb");

	if (f == NULL)
		return ("");

	std::string	out;
	char		buf[4096];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	fclose(f);
	return (out);
}

/* ------------------------------------------------------------------ */
/* G: read-only guarantee                                             */
/* ------------------------------------------------------------------ */

static void	test_plan_never_creates_a_registry_file(void)
{
	s_isolated_env	env;
	int				rc;

	capture_stdout_of_plan_dispatch({"unknown-model-xyz"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown name is CLI_ERROR");

	FILE	*f = fopen((env.dir + "/models.json").c_str(), "r");

	TEST_ASSERT(f == NULL, "`membrane plan` on a fresh env never creates "
		"a registry file (never mutates the registry)");
}

static void	test_plan_never_mutates_an_existing_registry(void)
{
	s_isolated_env	env;
	std::string		registry_path = env.dir + "/models.json";

	register_entry(registry_path, "stories15m", real_fixture_path());

	std::string	before = read_file(registry_path);
	int			rc;

	capture_stdout_of_plan_dispatch({"stories15m"}, false, &rc);
	capture_stdout_of_plan_dispatch({"stories15m", "--ctx", "128"}, false,
		&rc);
	capture_stdout_of_plan_dispatch({"stories15m"}, true, &rc);

	std::string	after = read_file(registry_path);

	TEST_ASSERT(before == after, "the registry file is byte-for-byte "
		"unchanged after several `membrane plan` calls");

	FILE	*install_dir = fopen((env.dir + "/models").c_str(), "r");

	TEST_ASSERT(install_dir == NULL, "no install directory was created");
}

/* ------------------------------------------------------------------ */
/* I: installed model uses real GGUF metadata                         */
/* ------------------------------------------------------------------ */

static void	test_installed_model_plan_json_shape(void)
{
	s_isolated_env	env;

	register_entry(env.dir + "/models.json", "stories15m",
		real_fixture_path());

	int			rc;
	std::string	out = capture_stdout_of_plan_dispatch({"stories15m"}, true,
			&rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "plan --json output is valid JSON");
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "a computed plan is always "
		"exit-success, feasible or not");
	TEST_ASSERT(j["schema_version"].get<int>() == MEMBRANE_PLAN_V2_SCHEMA_VERSION,
		"schema_version == 2 (Milestone G2's own schema bump)");
	TEST_ASSERT(j["mode"] == "plan", "mode is 'plan'");
	TEST_ASSERT(j["identity"]["installed"] == true,
		"identity.installed reflects the real registry entry");
	TEST_ASSERT(j["identity"]["model_path"] == real_fixture_path(),
		"identity.model_path is the real registered path");
	TEST_ASSERT(j.contains("hardware"), "hardware section present");
	TEST_ASSERT(j.contains("feasibility"), "feasibility section present");
	TEST_ASSERT(j.contains("reasons") && j["reasons"].is_array(),
		"reasons is a structured array, not one text blob");
	TEST_ASSERT(j["hardware"]["host_available_known"].get<bool>()
		|| !j["hardware"]["host_available_known"].get<bool>(),
		"host_available_known is a real boolean (present either way)");
}

/* ------------------------------------------------------------------ */
/* D: explicit override preserved end to end through the real CLI     */
/* ------------------------------------------------------------------ */

static void	test_explicit_ctx_preserved_end_to_end(void)
{
	s_isolated_env	env;

	register_entry(env.dir + "/models.json", "stories15m",
		real_fixture_path());

	int			rc;
	std::string	out = capture_stdout_of_plan_dispatch(
			{"stories15m", "--ctx", "128"}, true, &rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "valid JSON");
	TEST_ASSERT(j["workload"]["requested_context"].get<uint64_t>() == 128,
		"the explicit --ctx value is echoed back exactly, never silently "
		"replaced");
	TEST_ASSERT(j["workload"]["context_source"] == "explicit_user",
		"context_source reflects the explicit flag");
}

/* ------------------------------------------------------------------ */
/* H: catalog-only model discloses estimate-only planning             */
/* ------------------------------------------------------------------ */

static void	test_catalog_only_model_json_discloses_estimate(void)
{
	s_isolated_env	env;
	int				rc;
	std::string		out = capture_stdout_of_plan_dispatch(
			{"smollm2-135m-instruct"}, true, &rc);
	json			j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "valid JSON for a catalog-only model");
	TEST_ASSERT(j["identity"]["installed"] == false,
		"identity.installed is false for a catalog-only model");
	TEST_ASSERT(j["identity"]["model_path"] == "",
		"no model_path is fabricated for an uninstalled model");
	TEST_ASSERT(j["decisions"].is_null(),
		"no context/GPU/KV decision exists without real model metadata");

	bool	found_disclosure = false;

	for (const auto &r : j["reasons"])
		if (r["code"] == "MODEL_METADATA_UNAVAILABLE")
			found_disclosure = true;
	TEST_ASSERT(found_disclosure, "the estimate-only limitation is "
		"disclosed via a structured reason code");
}

static void	test_unknown_model_returns_not_found(void)
{
	s_isolated_env	env;
	int				rc;

	capture_stdout_of_plan_dispatch({"this-model-does-not-exist-anywhere"},
		false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "an unknown name (neither "
		"registered nor a catalog id/alias) is a clean CLI_ERROR");
}

/* ------------------------------------------------------------------ */
/* Milestone G2: real cross-variant sibling join, end to end through   */
/* the actual CLI dispatch (not just plan_v2_resolver's own synthetic  */
/* fixtures) -- Part 16's own lightweight smoke, kept as a permanent   */
/* regression test.                                                    */
/* ------------------------------------------------------------------ */

static void	test_real_multi_variant_join_json_schema(void)
{
	s_isolated_env	env;

	register_entry_as(env.dir + "/models.json", "smollm2-135m-instruct",
		real_smollm2_f16_fixture_path(), "SmolLM2-135M-Instruct-F16.gguf");

	int			rc;
	std::string	out = capture_stdout_of_plan_dispatch(
			{"smollm2-135m-instruct"}, true, &rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "valid JSON");
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "computed plan is exit-success");
	TEST_ASSERT(j["schema_version"].get<int>() == MEMBRANE_PLAN_V2_SCHEMA_VERSION,
		"schema_version == 2");
	TEST_ASSERT(j.contains("policy_version") && j["policy_version"].is_string(),
		"policy_version is present (Part 8's own deterministic policy)");
	TEST_ASSERT(j.contains("variants_evaluated")
		&& j["variants_evaluated"].is_array(),
		"variants_evaluated is a structured array");
	/* Real catalog family: Q4_K_M, Q5_K_M, Q8_0, F16 -- all 4 must be
	 * evaluated through the real join (Part 2 of the G2 task), not just
	 * the one installed variant. */
	TEST_ASSERT(j["variants_evaluated"].size() == 4,
		"all 4 real catalog variants were evaluated, not pre-filtered");

	bool	found_installed_real = false;
	bool	found_estimate_sibling = false;

	for (const auto &v : j["variants_evaluated"])
	{
		if (v["variant"] == "F16" && v["estimate_only"] == false)
			found_installed_real = true;
		if (v["variant"] != "F16" && v["estimate_only"] == true
			&& v["source"] == "catalog_metadata")
			found_estimate_sibling = true;
	}
	TEST_ASSERT(found_installed_real, "the real installed F16 variant is "
		"marked as a real (non-estimate) candidate");
	TEST_ASSERT(found_estimate_sibling, "a sibling variant is marked as a "
		"catalog_metadata-sourced estimate, never claimed exact");
	TEST_ASSERT(j.contains("alternatives") && j["alternatives"].is_array(),
		"alternatives is a bounded, structured array");
	TEST_ASSERT(j.contains("selected_variant"), "selected_variant is present");
}

static void	test_explicit_quant_selects_exact_sibling(void)
{
	s_isolated_env	env;

	register_entry_as(env.dir + "/models.json", "smollm2-135m-instruct",
		real_smollm2_f16_fixture_path(), "SmolLM2-135M-Instruct-F16.gguf");

	int			rc;
	std::string	out = capture_stdout_of_plan_dispatch(
			{"smollm2-135m-instruct", "--quant", "Q4_K_M"}, true, &rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "valid JSON");
	TEST_ASSERT(j["variants_evaluated"].size() == 1,
		"an explicit --quant restricts evaluation to exactly that variant, "
		"never silently considering others");
	TEST_ASSERT(j["identity"]["variant"] == "Q4_K_M",
		"the explicit variant is the one reflected in the top-level plan");
	TEST_ASSERT(j["identity"]["variant_source"] == "explicit_user",
		"provenance marks this variant as an explicit user override");
}

static void	test_explicit_huge_ctx_infeasible_across_all_variants(void)
{
	s_isolated_env	env;

	register_entry_as(env.dir + "/models.json", "smollm2-135m-instruct",
		real_smollm2_f16_fixture_path(), "SmolLM2-135M-Instruct-F16.gguf");

	int			rc;
	std::string	out = capture_stdout_of_plan_dispatch(
			{"smollm2-135m-instruct", "--ctx", "999999999"}, true, &rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), "valid JSON");
	TEST_ASSERT(j["feasible"] == false, "an impossible explicit ctx is "
		"honestly reported infeasible, never silently reduced");
	TEST_ASSERT(j["variants_evaluated"].size() == 4,
		"every variant is still evaluated (and reported) even though none "
		"is feasible");
	for (const auto &v : j["variants_evaluated"])
		TEST_ASSERT(v["feasible"] == false, "every variant is honestly "
			"infeasible under the same impossible explicit constraint");
}

int	main(void)
{
	test_plan_never_creates_a_registry_file();
	test_plan_never_mutates_an_existing_registry();
	test_installed_model_plan_json_shape();
	test_explicit_ctx_preserved_end_to_end();
	test_catalog_only_model_json_discloses_estimate();
	test_unknown_model_returns_not_found();
	test_real_multi_variant_join_json_schema();
	test_explicit_quant_selects_exact_sibling();
	test_explicit_huge_ctx_infeasible_across_all_variants();
	printf("all plan_cmd tests passed\n");
	return (0);
}
