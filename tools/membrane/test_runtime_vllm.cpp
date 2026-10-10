#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "runtime_vllm.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone L: the read-only vLLM adapter. Fixtures follow vLLM v0.31.0's
 * documented response shapes (see runtime_vllm.h). An in-process mock
 * server stands in for vLLM; every route the adapter must never reach is a
 * trap, and the log is asserted at the end.
 */

static httplib::Server			*g_srv;
static std::thread				*g_thr;
static int						g_port;
static std::mutex				g_mtx;
static std::vector<std::string>	g_log;
static int						g_health_status = 200;
static int						g_version_status = 200;
static std::string				g_version_body = "{\"version\":\"0.31.0\"}";
static std::string				g_models_body =
	"{\"object\":\"list\",\"data\":[{\"id\":\"Qwen/Qwen2.5-1.5B-Instruct\","
	"\"object\":\"model\",\"created\":1,\"owned_by\":\"vllm\","
	"\"root\":\"Qwen/Qwen2.5-1.5B-Instruct\",\"parent\":null,"
	"\"max_model_len\":32768,\"permission\":[]}]}";
static std::string				g_load_body = "{\"server_load\":2}";

static void	start_mock(void)
{
	g_srv = new httplib::Server();
	g_srv->set_logger([](const httplib::Request &req, const httplib::Response &)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_log.push_back(req.method + " " + req.path);
	});
	g_srv->Get("/health", [](const httplib::Request &, httplib::Response &r)
		{ r.status = g_health_status; });
	g_srv->Get("/version", [](const httplib::Request &, httplib::Response &r)
	{
		r.status = g_version_status;
		r.set_content(g_version_body, "application/json");
	});
	g_srv->Get("/v1/models", [](const httplib::Request &, httplib::Response &r)
		{ r.set_content(g_models_body, "application/json"); });
	g_srv->Get("/load", [](const httplib::Request &, httplib::Response &r)
		{ r.set_content(g_load_body, "application/json"); });
	for (const char *p : {"/v1/chat/completions", "/v1/completions",
			"/v1/embeddings", "/sleep", "/wake_up", "/reset_prefix_cache",
			"/collective_rpc"})
		g_srv->Post(p, [](const httplib::Request &, httplib::Response &r)
			{ r.status = 500; });
	g_srv->Get("/metrics", [](const httplib::Request &, httplib::Response &r)
		{ r.status = 500; });
	g_srv->Get("/is_sleeping", [](const httplib::Request &,
			httplib::Response &r) { r.status = 500; });
	g_port = g_srv->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_port > 0, "mock bound");
	g_thr = new std::thread([]() { g_srv->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_srv->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(g_port)).c_str(), 1);
}

static void	test_pure(void)
{
	std::string									v;
	long long									load = -1;
	std::vector<membrane_external_model_t>		models;
	std::string									err;
	membrane_runtime_capabilities_t				c;

	TEST_ASSERT(membrane_vllm_parse_version("{\"version\":\"0.31.0\"}", &v)
		&& v == "0.31.0", "version");
	TEST_ASSERT(!membrane_vllm_parse_version("{\"v\":1}", &v)
		&& !membrane_vllm_parse_version("nope", &v)
		&& !membrane_vllm_parse_version("{\"version\":\"\"}", &v), "bad version");
	TEST_ASSERT(membrane_vllm_parse_load("{\"server_load\":0}", &load)
		&& load == 0 && !membrane_vllm_parse_load("{\"server_load\":-1}", &load)
		&& !membrane_vllm_parse_load("{\"server_load\":\"x\"}", &load), "load");
	TEST_ASSERT(membrane_vllm_parse_models(g_models_body, &models, &err)
		&& models.size() == 1 && models[0].runtime_model_id
			== "Qwen/Qwen2.5-1.5B-Instruct" && models[0].context_length_known
		&& models[0].context_length == 32768 && !models[0].size_known
		&& models[0].runtime_id == "vllm", "models: max_model_len is the "
			"configured max; size is NOT invented");
	TEST_ASSERT(membrane_vllm_parse_models("{\"data\":[]}", &models, &err)
		&& models.empty(), "an empty served list is not an error");
	TEST_ASSERT(!membrane_vllm_parse_models("{\"data\":[{\"id\":\"\"}]}",
		&models, &err) && !membrane_vllm_parse_models("[]", &models, &err),
		"malformed lists rejected");
	membrane_vllm_capabilities(&c);
	TEST_ASSERT(c.model_enumeration == MEMBRANE_CAPABILITY_SUPPORTED
		&& c.chat_completions == MEMBRANE_CAPABILITY_SUPPORTED, "what it has");
	TEST_ASSERT(c.context_control == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.gpu_layer_control == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.kv_precision_control == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.kv_placement_control == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.model_switch == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.model_load_unload == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.dynamic_reconfiguration == MEMBRANE_CAPABILITY_UNSUPPORTED,
		"no control capability is claimed");
	TEST_ASSERT(c.ram_usage == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.vram_usage == MEMBRANE_CAPABILITY_UNSUPPORTED
		&& c.active_context == MEMBRANE_CAPABILITY_PARTIAL, "no memory "
			"telemetry claimed (it lives on /metrics, which is not read)");
}

static void	test_allowlist(void)
{
	for (const char *p : {"/health", "/version", "/v1/models", "/load"})
		TEST_ASSERT(membrane_vllm_request_allowed("GET", p), "allowed read");
	for (const char *p : {"/v1/chat/completions", "/v1/completions",
			"/metrics", "/sleep", "/wake_up", "/reset_prefix_cache",
			"/is_sleeping", "/collective_rpc", "/v1/models/../sleep",
			"/health/", ""})
	{
		TEST_ASSERT(!membrane_vllm_request_allowed("GET", p)
			&& !membrane_vllm_request_allowed("POST", p), "refused");
	}
	TEST_ASSERT(!membrane_vllm_request_allowed("POST", "/health")
		&& !membrane_vllm_request_allowed("POST", "/v1/models")
		&& !membrane_vllm_request_allowed("DELETE", "/load")
		&& !membrane_vllm_request_allowed("PUT", "/version"),
		"there is no mutating method at all");
}

static void	test_describe_states(void)
{
	membrane_runtime_descriptor_t	d;

	membrane_vllm_describe(&d);
	TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
		&& d.health == MEMBRANE_RUNTIME_HEALTH_HEALTHY && d.version_known
		&& std::string(d.version) == "0.31.0" && d.endpoint_known
		&& d.execution_mode == MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL
		&& d.type == MEMBRANE_RUNTIME_TYPE_EXTERNAL, "healthy");
	g_health_status = 503;
	membrane_vllm_describe(&d);
	TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE
		&& std::string(d.unavailable_reason).find("not healthy")
			!= std::string::npos, "engine dead (503) is not 'available'");
	g_health_status = 200;
	g_version_status = 404;
	membrane_vllm_describe(&d);
	TEST_ASSERT(d.health == MEMBRANE_RUNTIME_HEALTH_INCOMPATIBLE,
		"something else on the port => incompatible");
	g_version_status = 200;
	g_version_body = "<html>nope</html>";
	membrane_vllm_describe(&d);
	TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN
		&& d.health == MEMBRANE_RUNTIME_HEALTH_UNKNOWN, "malformed version");
	g_version_body = "{\"version\":\"0.31.0\"}";
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	membrane_vllm_describe(&d);
	TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE
		&& d.health == MEMBRANE_RUNTIME_HEALTH_UNREACHABLE, "unreachable");
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, "https://vllm.example.com", 1);
	membrane_vllm_describe(&d);
	TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE
		&& d.health == MEMBRANE_RUNTIME_HEALTH_NOT_PROBED
		&& std::string(d.unavailable_reason).find("invalid")
			!= std::string::npos, "https / credentials / paths are refused "
				"without any request");
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(g_port)).c_str(), 1);
}

static void	test_models_inspect_load(void)
{
	std::vector<membrane_external_model_t>	models;
	membrane_runtime_error_t				err;
	membrane_external_model_detail_t		det;
	long long								load = -1;

	membrane_external_model_detail_init(&det);
	TEST_ASSERT(membrane_vllm_list_models(&models, &err) && models.size() == 1,
		"list");
	TEST_ASSERT(membrane_vllm_inspect_model("Qwen/Qwen2.5-1.5B-Instruct", &det,
		&err) && det.model.context_length == 32768, "inspect");
	TEST_ASSERT(!membrane_vllm_inspect_model("nope", &det, &err)
		&& err.code == MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND, "not served");
	TEST_ASSERT(membrane_vllm_server_load(&load, &err) && load == 2, "load");
	g_models_body = "garbage";
	TEST_ASSERT(!membrane_vllm_list_models(&models, &err)
		&& err.code == MEMBRANE_RUNTIME_ERR_MALFORMED, "malformed");
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	TEST_ASSERT(!membrane_vllm_list_models(&models, &err)
		&& err.code == MEMBRANE_RUNTIME_ERR_UNAVAILABLE, "unreachable");
}

static void	test_route_trap(void)
{
	std::lock_guard<std::mutex>	l(g_mtx);

	TEST_ASSERT(!g_log.empty(), "the adapter did reach the mock");
	for (const auto &line : g_log)
	{
		size_t		sp = line.find(' ');
		std::string	method = line.substr(0, sp);
		std::string	path = line.substr(sp + 1);

		TEST_ASSERT(membrane_vllm_request_allowed(method, path),
			"every request the server saw is on the allowlist");
		TEST_ASSERT(method == "GET", "GET only: no inference, no admin");
	}
}

int	main(void)
{
	test_pure();
	test_allowlist();
	start_mock();
	test_describe_states();
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(g_port)).c_str(), 1);
	g_models_body =
		"{\"object\":\"list\",\"data\":[{\"id\":\"Qwen/Qwen2.5-1.5B-Instruct\","
		"\"max_model_len\":32768}]}";
	test_models_inspect_load();
	test_route_trap();
	g_srv->stop();
	g_thr->join();
	printf("test_runtime_vllm: all tests passed\n");
	return (0);
}
