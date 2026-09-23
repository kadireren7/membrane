#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
# include <unistd.h>
#endif

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "plan_cmd.h"
#include "product_cli.h"
#include "registry_core.h"
#include "runtime_adapter.h"
#include "runtime_capabilities.h"
#include "runtime_ollama.h"
#include "runtime_plan_assessment.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone H3: runtime-aware plan assessment tests. Two halves:
 *
 *  1. Pure library tests of membrane_runtime_recommend_plan() against the
 *     REAL membrane-native matrix (runtime_capabilities.c) and the REAL
 *     Ollama matrix (membrane_ollama_capabilities()), with hand-built
 *     membrane_plan_t fixtures -- same pattern as test_runtime_
 *     capabilities.c.
 *  2. `membrane plan MODEL --runtime ID` end to end: membrane-native on
 *     the repo's real SmolLM2-135M F16 fixture (GGUF metadata only, no
 *     generation), and ollama against an in-process mock server that logs
 *     every request. No real Ollama is needed or contacted.
 */

/* ================================================================== */
/* Fixtures                                                           */
/* ================================================================== */

static membrane_plan_t	exact_native_plan(void)
{
	membrane_plan_t	p;

	memset(&p, 0, sizeof(p));
	snprintf(p.identity.model_name, sizeof(p.identity.model_name),
		"smollm2-135m-instruct");
	p.identity.installed = 1;
	p.identity.variant_known = 1;
	snprintf(p.identity.variant, sizeof(p.identity.variant), "Q4_K_M");
	p.identity.variant_source = MEMBRANE_PLAN_SOURCE_MODEL_METADATA;
	p.decisions.has_decisions = 1;
	p.decisions.context = 16384;
	p.decisions.context_source = MEMBRANE_PLAN_SOURCE_PLANNER_DECISION;
	p.decisions.gpu_layers = 24;
	p.decisions.gpu_layers_source = MEMBRANE_PLAN_SOURCE_PLANNER_DECISION;
	p.decisions.kv_precision = MEMBRANE_JOINT_KV_Q8;
	p.decisions.kv_precision_source = MEMBRANE_PLAN_SOURCE_PLANNER_DECISION;
	p.decisions.kv_placement = MEMBRANE_JOINT_PLACEMENT_AUTO;
	p.decisions.kv_placement_source = MEMBRANE_PLAN_SOURCE_PLANNER_DECISION;
	p.feasibility.feasible = 1;
	return (p);
}

static membrane_runtime_descriptor_t	native_runtime(void)
{
	membrane_runtime_descriptor_t	d;

	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE, &d),
		"membrane-native describable");
	return (d);
}

/* The real Ollama capability contract, as an AVAILABLE runtime -- no
 * network: availability is set by hand, exactly what a healthy probe
 * would produce. */
static membrane_runtime_descriptor_t	ollama_runtime(void)
{
	membrane_runtime_descriptor_t	d;

	memset(&d, 0, sizeof(d));
	snprintf(d.id, sizeof(d.id), "%s", MEMBRANE_RUNTIME_ID_OLLAMA);
	d.type = MEMBRANE_RUNTIME_TYPE_EXTERNAL;
	d.execution_mode = MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL;
	d.availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	d.health = MEMBRANE_RUNTIME_HEALTH_HEALTHY;
	d.capability_provenance = MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT;
	membrane_ollama_capabilities(&d.capabilities);
	return (d);
}

static const membrane_assessment_dimension_result_t	&dim(
		const membrane_runtime_plan_assessment_t &a,
		membrane_assessment_dimension_t d)
{
	return (a.dimensions[d]);
}

static bool	has_reason(const membrane_runtime_plan_assessment_t &a,
				const char *code)
{
	for (size_t i = 0; i < a.reason_count; ++i)
		if (strcmp(a.reasons[i].code, code) == 0)
			return (true);
	return (false);
}

/* ================================================================== */
/* 1. Library                                                         */
/* ================================================================== */

/* A + O: native, exact plan -> fully actionable */
static void	test_native_fully_actionable(void)
{
	membrane_runtime_descriptor_t		rt = native_runtime();
	membrane_plan_t						plan = exact_native_plan();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;
	membrane_negotiation_outcome_t		neg;

	in.runtime = &rt;
	in.plan = &plan;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE,
		"native + exact plan -> fully actionable");
	TEST_ASSERT(a.planning_level == MEMBRANE_PLANNING_LEVEL_PLANNER_EXACT,
		"planner_exact");
	TEST_ASSERT(a.required_count == 5 && a.controllable_count == 5,
		"all 5 planner dimensions required and controllable");
	TEST_ASSERT(!dim(a, MEMBRANE_ASSESS_DIM_CONCURRENCY).required
		&& dim(a, MEMBRANE_ASSESS_DIM_CONCURRENCY).applicability
			== MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE,
		"native concurrency stays partial but is not a plan dimension");
	TEST_ASSERT(strcmp(dim(a, MEMBRANE_ASSESS_DIM_CONTEXT).value, "16384") == 0
		&& strcmp(dim(a, MEMBRANE_ASSESS_DIM_KV_PRECISION).value, "q8") == 0
		&& strcmp(dim(a, MEMBRANE_ASSESS_DIM_KV_PLACEMENT).value, "auto") == 0
		&& strcmp(dim(a, MEMBRANE_ASSESS_DIM_GPU_LAYERS).value, "24") == 0,
		"planner values carried verbatim");
	neg = membrane_runtime_negotiate_plan(&rt.capabilities, &plan);
	TEST_ASSERT(neg.result == MEMBRANE_NEGOTIATION_FULLY_SUPPORTED,
		"agrees with H1 negotiation");
}

/* B, C, D, E, F, G: same plan against the real Ollama matrix */
static void	test_ollama_partially_actionable(void)
{
	membrane_runtime_descriptor_t		rt = ollama_runtime();
	membrane_plan_t						plan = exact_native_plan();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;
	membrane_negotiation_outcome_t		neg;

	in.runtime = &rt;
	in.plan = &plan;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_PARTIALLY_ACTIONABLE,
		"B: derived partially actionable (not hardcoded)");
	TEST_ASSERT(a.required_count == 5 && a.controllable_count == 1,
		"only context is controllable");
	neg = membrane_runtime_negotiate_plan(&rt.capabilities, &plan);
	TEST_ASSERT(neg.result == MEMBRANE_NEGOTIATION_PARTIAL
		&& neg.unsupported_count == 4, "agrees with H1 negotiation");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_KV_PRECISION).applicability
		== MEMBRANE_APPLICABILITY_UNSUPPORTED && strcmp(dim(a,
		MEMBRANE_ASSESS_DIM_KV_PRECISION).reason_code,
		"KV_PRECISION_CONTROL_UNSUPPORTED") == 0, "C: KV precision");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_KV_PLACEMENT).applicability
		== MEMBRANE_APPLICABILITY_UNSUPPORTED && strcmp(dim(a,
		MEMBRANE_ASSESS_DIM_KV_PLACEMENT).reason_code,
		"KV_PLACEMENT_CONTROL_UNSUPPORTED") == 0, "D: KV placement");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_GPU_LAYERS).applicability
		== MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE && strcmp(dim(a,
		MEMBRANE_ASSESS_DIM_GPU_LAYERS).reason_code,
		"GPU_LAYERS_CONTROL_PARTIAL") == 0, "E: GPU layers partial");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_CONTEXT).applicability
		== MEMBRANE_APPLICABILITY_CONTROLLABLE && strcmp(dim(a,
		MEMBRANE_ASSESS_DIM_CONTEXT).reason_code,
		"CONTEXT_CONTROL_SUPPORTED") == 0, "F: context controllable");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_QUANT_VARIANT).applicability
		== MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE,
		"quant is only partially controllable (tag-dependent)");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_DEVICE_SELECTION).applicability
		== MEMBRANE_APPLICABILITY_UNKNOWN
		&& !dim(a, MEMBRANE_ASSESS_DIM_DEVICE_SELECTION).required
		&& !dim(a, MEMBRANE_ASSESS_DIM_DEVICE_SELECTION).planner_dimension,
		"G: device selection unknown and never a plan dimension");
}

/* H + I: capability-only -- a valid result, not an error */
static void	test_capability_only(void)
{
	membrane_runtime_descriptor_t		rt = ollama_runtime();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;
	size_t								i;

	in.runtime = &rt;
	in.model_known = 1;
	in.runtime_model_id = "qwen2.5:7b";
	in.model_quantization = "Q4_K_M";
	in.model_max_context_known = 1;
	in.model_max_context = 32768;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_ADVISORY_ONLY,
		"H: no plan, no request -> advisory only");
	TEST_ASSERT(a.planning_level == MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY,
		"H: planning level capability_only");
	TEST_ASSERT(has_reason(a, MEMBRANE_ASSESS_REASON_NO_PLANNER_PLAN)
		&& has_reason(a, MEMBRANE_ASSESS_REASON_EXACT_MEMORY_PLAN_UNAVAILABLE)
		&& has_reason(a, MEMBRANE_ASSESS_REASON_QUANT_FIXED_BY_RUNTIME_MODEL)
		&& !has_reason(a, MEMBRANE_ASSESS_REASON_RUNTIME_UNAVAILABLE),
		"H: explained, and not reported as a failure");
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_QUANT_VARIANT).value_source
		== MEMBRANE_ASSESS_VALUE_RUNTIME_METADATA
		&& !dim(a, MEMBRANE_ASSESS_DIM_QUANT_VARIANT).required,
		"the tag's quant is a reported fact, not a recommendation");
	for (i = MEMBRANE_ASSESS_DIM_CONTEXT; i < MEMBRANE_ASSESS_DIM_COUNT; ++i)
		TEST_ASSERT(!a.dimensions[i].value_known,
			"I: no context/GPU/KV/memory value is fabricated");
	TEST_ASSERT(a.dimensions[MEMBRANE_ASSESS_DIM_CONTEXT].applicability
		== MEMBRANE_APPLICABILITY_CONTROLLABLE,
		"capability information is still reported");
}

static void	test_capability_only_with_requests(void)
{
	membrane_runtime_descriptor_t		rt = ollama_runtime();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;

	in.runtime = &rt;
	in.model_known = 1;
	in.runtime_model_id = "qwen2.5:7b";
	in.model_quantization = "Q4_K_M";
	in.model_max_context_known = 1;
	in.model_max_context = 32768;
	in.requested_context_known = 1;
	in.requested_context = 8192;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE,
		"only a controllable dimension requested -> fully actionable");
	TEST_ASSERT(a.planning_level == MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY
		&& dim(a, MEMBRANE_ASSESS_DIM_CONTEXT).value_source
			== MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST,
		"a request is still not a Planner v2 plan");
	in.requested_kv_precision_known = 1;
	in.requested_kv_precision = MEMBRANE_JOINT_KV_Q8;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_PARTIALLY_ACTIONABLE,
		"context + KV precision -> partially actionable");
	in.requested_context_known = 0;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_ADVISORY_ONLY,
		"only an unsupported dimension requested -> advisory only");
	in.requested_context_known = 1;
	in.requested_context = 65536;
	in.requested_quant = "Q8_0";
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(has_reason(a, MEMBRANE_ASSESS_REASON_CONTEXT_EXCEEDS_MODEL_MAX)
		&& has_reason(a, MEMBRANE_ASSESS_REASON_REQUESTED_QUANT_DIFFERS),
		"requests beyond the model's reported facts are flagged");
	in.requested_quant = "q4_k_m";
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(!has_reason(a, MEMBRANE_ASSESS_REASON_REQUESTED_QUANT_DIFFERS),
		"a quant equal to the tag's up to case is not flagged");
	in.requested_quant = "Q4_K";
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(has_reason(a, MEMBRANE_ASSESS_REASON_REQUESTED_QUANT_DIFFERS),
		"a prefix of the tag's quant is still a different quant");
}

static void	test_unavailable_and_unknown(void)
{
	membrane_runtime_descriptor_t		rt = ollama_runtime();
	membrane_plan_t						plan = exact_native_plan();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;

	rt.availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
	in.runtime = &rt;
	in.plan = &plan;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_UNSUPPORTED
		&& a.planning_level == MEMBRANE_PLANNING_LEVEL_UNAVAILABLE
		&& has_reason(a, MEMBRANE_ASSESS_REASON_RUNTIME_UNAVAILABLE)
		&& a.required_count == 0, "unavailable runtime -> unsupported");
	rt.availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	in.plan = NULL;
	TEST_ASSERT(membrane_runtime_recommend_plan(&in, &a)
		== MEMBRANE_ACTIONABILITY_UNSUPPORTED
		&& has_reason(a, MEMBRANE_ASSESS_REASON_MODEL_UNKNOWN),
		"no plan and no model -> unsupported");
	TEST_ASSERT(membrane_runtime_recommend_plan(NULL, &a)
		== MEMBRANE_ACTIONABILITY_UNSUPPORTED, "NULL input is safe");
}

static void	test_estimate_and_observable_only(void)
{
	membrane_runtime_descriptor_t		rt = native_runtime();
	membrane_plan_t						plan = exact_native_plan();
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;

	plan.identity.variant_estimate_only = 1;
	in.runtime = &rt;
	in.plan = &plan;
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(a.planning_level == MEMBRANE_PLANNING_LEVEL_PLANNER_ESTIMATE
		&& has_reason(a, MEMBRANE_ASSESS_REASON_PLAN_ESTIMATE_ONLY),
		"a sibling-scaled plan is an estimate");
	plan.decisions.has_decisions = 0;
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(a.required_count == 1 && a.actionability
		== MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE,
		"a catalog-only plan relies on the variant only");
	/* Synthetic matrix: context not controllable, but reported. */
	rt.capabilities.context_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	rt.capabilities.active_context = MEMBRANE_CAPABILITY_SUPPORTED;
	plan = exact_native_plan();
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_CONTEXT).applicability
		== MEMBRANE_APPLICABILITY_OBSERVABLE_ONLY,
		"unsupported control + supported telemetry -> observable only");
	rt.capabilities.active_context = MEMBRANE_CAPABILITY_UNSUPPORTED;
	membrane_runtime_recommend_plan(&in, &a);
	TEST_ASSERT(dim(a, MEMBRANE_ASSESS_DIM_CONTEXT).applicability
		== MEMBRANE_APPLICABILITY_UNSUPPORTED, "no telemetry -> unsupported");
}

/* K (library) + L (library): deterministic, never mutates its inputs */
static void	test_library_deterministic_and_pure(void)
{
	membrane_runtime_descriptor_t		rt = ollama_runtime();
	membrane_runtime_descriptor_t		rt_copy = rt;
	membrane_plan_t						plan = exact_native_plan();
	membrane_plan_t						plan_copy = plan;
	membrane_runtime_recommend_input_t	in = {};
	membrane_runtime_plan_assessment_t	a;
	membrane_runtime_plan_assessment_t	b;

	in.runtime = &rt;
	in.plan = &plan;
	membrane_runtime_recommend_plan(&in, &a);
	membrane_runtime_recommend_plan(&in, &b);
	TEST_ASSERT(memcmp(&a, &b, sizeof(a)) == 0, "byte-identical results");
	TEST_ASSERT(memcmp(&plan, &plan_copy, sizeof(plan)) == 0
		&& memcmp(&rt, &rt_copy, sizeof(rt)) == 0,
		"the plan and runtime descriptor are never modified");
}

/* ================================================================== */
/* 2. CLI                                                             */
/* ================================================================== */

static const char	*g_show_fixture = R"({
  "parameters": "stop \"<|im_end|>\"",
  "template": "{{ .Prompt }}",
  "details": {"format": "gguf", "family": "qwen2", "families": ["qwen2"],
    "parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
  "model_info": {"general.architecture": "qwen2",
    "general.parameter_count": 7615616512, "qwen2.context_length": 32768,
    "qwen2.block_count": 28},
  "capabilities": ["completion", "tools"],
  "modified_at": "2026-09-01T10:00:00+03:00"
})";

static httplib::Server			*g_mock;
static std::thread				*g_mock_thread;
static int						g_mock_port;
static std::mutex				g_mock_mutex;
static std::vector<std::string>	g_mock_log;
static int						g_show_status = 200;

static void	trap(const httplib::Request &, httplib::Response &res)
{
	res.status = 500;
	res.set_content("{\"error\":\"trap\"}", "application/json");
}

static void	start_mock_server(void)
{
	g_mock = new httplib::Server();
	g_mock->set_logger([](const httplib::Request &req,
			const httplib::Response &)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_mock_log.push_back(req.method + " " + req.path);
	});
	g_mock->Get("/api/version", [](const httplib::Request &,
			httplib::Response &res)
	{
		res.set_content("{\"version\":\"0.34.3\"}", "application/json");
	});
	g_mock->Post("/api/show", [](const httplib::Request &,
			httplib::Response &res)
	{
		res.status = g_show_status;
		res.set_content(g_show_status == 200 ? g_show_fixture
			: "{\"error\":\"model not found\"}", "application/json");
	});
	for (const char *p : {"/api/generate", "/api/chat", "/api/embed",
			"/api/pull", "/api/push", "/api/create", "/api/copy",
			"/v1/chat/completions", "/v1/completions"})
		g_mock->Post(p, trap);
	g_mock->Delete("/api/delete", trap);
	g_mock->Get("/api/ps", trap);
	g_mock->Get("/api/tags", trap);
	g_mock_port = g_mock->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_mock_port > 0, "mock bound");
	g_mock_thread = new std::thread([]() { g_mock->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_mock->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	TEST_ASSERT(g_mock->is_running(), "mock ready");
}

static void	stop_mock_server(void)
{
	g_mock->stop();
	g_mock_thread->join();
	delete g_mock_thread;
	delete g_mock;
}

static std::vector<std::string>	take_mock_log(void)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);
	std::vector<std::string>	out = g_mock_log;

	g_mock_log.clear();
	return (out);
}

static void	use_mock(void)
{
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(g_mock_port)).c_str(), 1);
}

static std::string	read_and_remove(const char *path)
{
	std::ifstream		f(path);
	std::stringstream	ss;

	ss << f.rdbuf();
	remove(path);
	return (ss.str());
}

static std::string	run_plan(const std::vector<std::string> &args,
				bool want_json, int *rc)
{
	char	out_tmpl[] = "/tmp/membrane-runtime-plan-out-XXXXXX";
	char	err_tmpl[] = "/tmp/membrane-runtime-plan-err-XXXXXX";
	int		out_fd = mkstemp(out_tmpl);
	int		err_fd = mkstemp(err_tmpl);
	int		saved_out;
	int		saved_err;

	TEST_ASSERT(out_fd >= 0 && err_fd >= 0, "capture files");
	fflush(stdout);
	fflush(stderr);
	saved_out = dup(STDOUT_FILENO);
	saved_err = dup(STDERR_FILENO);
	dup2(out_fd, STDOUT_FILENO);
	dup2(err_fd, STDERR_FILENO);
	close(out_fd);
	close(err_fd);
	*rc = membrane_plan_cmd_dispatch(args, want_json);
	fflush(stdout);
	fflush(stderr);
	dup2(saved_out, STDOUT_FILENO);
	dup2(saved_err, STDERR_FILENO);
	close(saved_out);
	close(saved_err);
	read_and_remove(err_tmpl);
	return (read_and_remove(out_tmpl));
}

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		char	tmpl[] = "/tmp/membrane-runtime-plan-test-XXXXXX";

		TEST_ASSERT(mkdtemp(tmpl) != NULL, "mkdtemp");
		dir = tmpl;
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_MODELS_INSTALL_DIR", (dir + "/models").c_str(), 1);
		setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(),
			1);
	}
	~s_isolated_env()
	{
		std::string	cmd = "rm -rf '" + dir + "'";

		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_MODELS_INSTALL_DIR");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		if (system(cmd.c_str()) != 0)
			fprintf(stderr, "warning: cleanup of %s failed\n", dir.c_str());
	}
};

static void	register_smollm2_fixture(const std::string &registry_path,
				const std::string &name)
{
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	entry;

	membrane_registry_load(registry_path, &reg, &err);
	entry.name = name;
	entry.path = std::string(MEMBRANE_TEST_SOURCE_DIR)
		+ "/models/SmolLM2-135M-Instruct-f16.gguf";
	entry.basename = "SmolLM2-135M-Instruct-F16.gguf";
	entry.arch_name = "";
	entry.model_max_context = 0;
	entry.file_size_bytes = 1;
	entry.file_mtime_ns = 1;
	entry.added_at_unix = 1700000000;
	TEST_ASSERT(membrane_registry_add(&reg, entry, &err), "registry add");
	TEST_ASSERT(membrane_registry_save(registry_path, reg, &err),
		"registry save");
}

static std::string	file_bytes(const std::string &path)
{
	std::ifstream		f(path, std::ios::binary);
	std::stringstream	ss;

	ss << f.rdbuf();
	return (ss.str());
}

/* Each `membrane plan` call takes its own live /proc/meminfo snapshot,
 * so host-memory figures legitimately differ between two calls -- strip
 * exactly those live fields before comparing two runs. */
static json	without_live_hardware(json j)
{
	if (j.is_object())
	{
		j.erase("hardware");
		j.erase("feasibility");
		j.erase("explanation");
		for (auto &el : j.items())
			el.value() = without_live_hardware(el.value());
	}
	else if (j.is_array())
		for (auto &el : j)
			el = without_live_hardware(el);
	return (j);
}

/* A + N + O (CLI): native installed model, exact plan */
static void	test_cli_native_exact(void)
{
	s_isolated_env	env;
	std::string		registry = env.dir + "/models.json";
	std::string		before;
	std::string		out;
	json			plain;
	json			j;
	int				rc;

	register_smollm2_fixture(registry, "smollm2-135m-instruct");
	before = file_bytes(registry);
	out = run_plan({"smollm2-135m-instruct"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "plain plan exits 0");
	plain = json::parse(out);
	TEST_ASSERT(plain["policy_version"] == "plan-v2-variant-joint-v1"
		&& !plain.contains("planning_level"),
		"N: `membrane plan` without --runtime is unchanged");
	out = run_plan({"smollm2-135m-instruct", "--runtime", "membrane-native"},
		true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "native assessment exits 0");
	j = json::parse(out);
	TEST_ASSERT(j["schema_version"] == MEMBRANE_RUNTIME_ASSESSMENT_SCHEMA_VERSION
		&& j["mode"] == "runtime_plan_assessment"
		&& j["mutates_state"] == false, "schema header");
	TEST_ASSERT(j["runtime"]["id"] == "membrane-native", "runtime id");
	TEST_ASSERT(j["model"]["identity_namespace"] == "membrane_registry"
		&& j["model"]["runtime_model_id"] == "smollm2-135m-instruct",
		"native identity is the registry name");
	TEST_ASSERT(j["planning_level"] == "planner_exact",
		"a real installed GGUF -> planner_exact");
	TEST_ASSERT(j["assessment"]["actionability"] == "fully_actionable",
		"A: native -> fully actionable");
	TEST_ASSERT(without_live_hardware(j["planner_plan"])
		== without_live_hardware(plain),
		"N: Planner v2's own output is embedded verbatim");
	TEST_ASSERT(j["dimensions"].size() == MEMBRANE_ASSESS_DIM_COUNT
		&& j["dimensions"][1]["name"] == "context"
		&& j["dimensions"][1]["value_source"] == "planner_v2"
		&& j["dimensions"][1]["value"] == j["planner_plan"]["decisions"]
			["context"].dump(),
		"dimension values come from the embedded Planner v2 plan");
	TEST_ASSERT(without_live_hardware(json::parse(run_plan({
		"smollm2-135m-instruct", "--runtime", "membrane-native"}, true, &rc)))
		== without_live_hardware(j), "K: deterministic JSON (apart from "
		"the live host-memory snapshot)");
	TEST_ASSERT(file_bytes(registry) == before, "L: registry untouched");
	out = run_plan({"smollm2-135m-instruct", "--runtime", "membrane-native"},
		false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS
		&& out.find("Fully actionable") != std::string::npos
		&& out.find("No settings were changed.") != std::string::npos
		&& out.find("Applied") == std::string::npos
		&& out.find("Activated") == std::string::npos,
		"human output: recommendation wording only");
}

/* B, H, J, L, M (CLI): ollama, capability-only via the mock */
static void	test_cli_ollama(void)
{
	s_isolated_env				env;
	std::string					registry = env.dir + "/models.json";
	std::string					before;
	std::string					out;
	std::vector<std::string>	log;
	json						j;
	int							rc;

	/* J: a native registry model with the SAME name must not be used. */
	register_smollm2_fixture(registry, "qwen2.5:7b");
	before = file_bytes(registry);
	use_mock();
	take_mock_log();
	out = run_plan({"qwen2.5:7b", "--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "ollama assessment exits 0");
	j = json::parse(out);
	TEST_ASSERT(j["planning_level"] == "capability_only"
		&& j["assessment"]["actionability"] == "advisory_only"
		&& j["planner_plan"].is_null(),
		"H: capability-only, not an error, no fabricated planner plan");
	TEST_ASSERT(j["model"]["identity_namespace"] == "runtime_inventory"
		&& j["model"]["quantization"] == "Q4_K_M"
		&& j["model"]["architecture"] == "qwen2"
		&& j["model"]["max_context_length"] == 32768,
		"J: identity + facts are the runtime's own, not the registry's");
	for (const auto &d : j["dimensions"])
		if (d["name"] != "quant_variant")
			TEST_ASSERT(d["value"].is_null(),
				"I: no exact context/GPU/KV value for Ollama");
	TEST_ASSERT(j["mutates_state"] == false, "mutates_state false");
	log = take_mock_log();
	TEST_ASSERT(log == std::vector<std::string>({"GET /api/version",
		"POST /api/show"}), "M: only H2's read calls, no new route");
	TEST_ASSERT(run_plan({"qwen2.5:7b", "--runtime", "ollama"}, true, &rc)
		== out, "K: the ollama assessment JSON is byte-identical");
	take_mock_log();

	out = run_plan({"qwen2.5:7b", "--runtime", "ollama", "--ctx", "8192",
		"--kv", "q8", "--gpu-layers", "20"}, true, &rc);
	j = json::parse(out);
	TEST_ASSERT(j["assessment"]["actionability"] == "partially_actionable"
		&& j["assessment"]["required_dimensions"] == 3
		&& j["assessment"]["controllable_dimensions"] == 1,
		"B: requested ctx/kv/gpu-layers -> partially actionable");
	out = run_plan({"qwen2.5:7b", "--runtime", "ollama", "--ctx", "8192",
		"--kv", "q8"}, false, &rc);
	TEST_ASSERT(out.find("Partially actionable") != std::string::npos
		&& out.find("Runtime can control\n  Context") != std::string::npos
		&& out.find("Runtime cannot control\n  KV precision")
			!= std::string::npos
		&& out.find("not a MEMBRANE registry model") != std::string::npos
		&& out.find("No settings were changed.") != std::string::npos,
		"human output groups dimensions by applicability");
	log = take_mock_log();
	for (const auto &line : log)
	{
		size_t	sp = line.find(' ');

		TEST_ASSERT(membrane_ollama_request_allowed(line.substr(0, sp),
			line.substr(sp + 1)), "M: every request is on H2's allowlist");
		TEST_ASSERT(line.find("/api/chat") == std::string::npos
			&& line.find("/api/generate") == std::string::npos
			&& line.find("/api/ps") == std::string::npos,
			"M: no inference/control/new route");
	}
	TEST_ASSERT(file_bytes(registry) == before,
		"L: registry untouched by the external path");
	TEST_ASSERT(access((env.dir + "/server.json").c_str(), F_OK) != 0,
		"L: no config file written");
}

static void	test_cli_ollama_errors(void)
{
	std::string	out;
	int			rc;

	use_mock();
	take_mock_log();
	out = run_plan({"gpt-oss:20b-cloud", "--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_MODEL_ERROR && json::parse(out)["error"]
		["code"] == MEMBRANE_RUNTIME_ERR_CLOUD_REFUSED,
		"cloud refs refused");
	TEST_ASSERT(take_mock_log() == std::vector<std::string>({
		"GET /api/version"}), "no /api/show for a cloud ref");
	g_show_status = 404;
	out = run_plan({"nope:1b", "--runtime", "ollama"}, true, &rc);
	g_show_status = 200;
	TEST_ASSERT(rc == MEMBRANE_EXIT_MODEL_ERROR && json::parse(out)["error"]
		["code"] == MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND, "missing model");
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	out = run_plan({"qwen2.5:7b", "--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR
		&& json::parse(out)["planning_level"] == "unavailable"
		&& json::parse(out)["assessment"]["actionability"] == "unsupported",
		"unreachable -> printed assessment, unsupported, exit 4");
	run_plan({"x", "--runtime", "vllm"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "vllm stays reserved");
	run_plan({"x", "--runtime", "nope"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown runtime");
}

int	main(void)
{
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	test_native_fully_actionable();
	test_ollama_partially_actionable();
	test_capability_only();
	test_capability_only_with_requests();
	test_unavailable_and_unknown();
	test_estimate_and_observable_only();
	test_library_deterministic_and_pure();
	start_mock_server();
	test_cli_native_exact();
	test_cli_ollama();
	test_cli_ollama_errors();
	stop_mock_server();
	printf("test_runtime_plan: all tests passed\n");
	return (0);
}
