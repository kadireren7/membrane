#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "orchestrate_cmd.h"
#include "product_cli.h"
#include "runtime_ollama.h"
#include "runtime_vllm.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone L: `membrane run` and the gathering layer, against in-process
 * mock Ollama and vLLM servers and an EMPTY native registry (so no model is
 * loaded and nothing is installed). The pure ranking is covered by
 * tools/membrane-run/test_orchestrator.c.
 */

static std::string				g_dir;
static httplib::Server			*g_oll;
static httplib::Server			*g_vll;
static std::thread				*g_oll_thr;
static std::thread				*g_vll_thr;
static std::mutex				g_mtx;
static std::vector<std::string>	g_oll_log;
static std::vector<std::string>	g_vll_log;
static bool						g_oll_has_model = true;
static bool						g_oll_resident = true;
static bool						g_vll_has_model = true;

static std::string	capture(const std::vector<std::string> &args, bool want_json,
						int *rc)
{
	std::string	tmpl = g_dir + "/cap-XXXXXX";
	int			fd = mkstemp(&tmpl[0]);

	TEST_ASSERT(fd >= 0, "capture file");
	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_run_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	FILE		*f = fopen(tmpl.c_str(), "rb");
	std::string	out;
	char		buf[4096];
	size_t		n;

	while (f != NULL && (n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	if (f != NULL)
		fclose(f);
	unlink(tmpl.c_str());
	return (out);
}

static int	start(httplib::Server *s, std::thread **t)
{
	int	port = s->bind_to_any_port("127.0.0.1");

	TEST_ASSERT(port > 0, "bound");
	*t = new std::thread([s]() { s->listen_after_bind(); });
	for (int i = 0; i < 100 && !s->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	return (port);
}

static void	start_mocks(void)
{
	g_oll = new httplib::Server();
	g_oll->set_logger([](const httplib::Request &q, const httplib::Response &)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_oll_log.push_back(q.method + " " + q.path);
	});
	g_oll->Get("/api/version", [](const httplib::Request &, httplib::Response &r)
		{ r.set_content("{\"version\":\"0.34.3\"}", "application/json"); });
	g_oll->Get("/api/tags", [](const httplib::Request &, httplib::Response &r)
	{
		json	models = json::array();

		if (g_oll_has_model)
			models.push_back({{"name", "qwen2.5:1.5b"}, {"model", "qwen2.5:1.5b"},
				{"size", 1100000000ull}, {"digest", "d"}, {"details",
					{{"format", "gguf"}, {"family", "qwen2"},
						{"parameter_size", "1.5B"},
						{"quantization_level", "Q4_K_M"}}}});
		r.set_content(json{{"models", models}}.dump(), "application/json");
	});
	g_oll->Get("/api/ps", [](const httplib::Request &, httplib::Response &r)
	{
		json	models = json::array();

		if (g_oll_has_model && g_oll_resident)
			models.push_back({{"name", "qwen2.5:1.5b"}, {"model",
				"qwen2.5:1.5b"}, {"size", 1300000000ull}, {"size_vram", 0},
				{"digest", "d"}, {"context_length", 4096}});
		r.set_content(json{{"models", models}}.dump(), "application/json");
	});
	for (const char *p : {"/api/generate", "/api/chat", "/api/pull",
			"/api/create", "/api/show", "/v1/chat/completions"})
		g_oll->Post(p, [](const httplib::Request &q, httplib::Response &r)
		{
			std::lock_guard<std::mutex>	l(g_mtx);

			g_oll_log.push_back("TRAP " + q.path);
			r.status = 500;
		});
	g_vll = new httplib::Server();
	g_vll->set_logger([](const httplib::Request &q, const httplib::Response &)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_vll_log.push_back(q.method + " " + q.path);
	});
	g_vll->Get("/version", [](const httplib::Request &, httplib::Response &r)
		{ r.set_content("{\"version\":\"0.31.0\"}", "application/json"); });
	g_vll->Get("/health", [](const httplib::Request &, httplib::Response &r)
		{ r.status = 200; });
	g_vll->Get("/v1/models", [](const httplib::Request &, httplib::Response &r)
	{
		json	data = json::array();

		if (g_vll_has_model)
			data.push_back({{"id", "Qwen2.5-1.5B"}, {"max_model_len", 32768}});
		r.set_content(json{{"object", "list"}, {"data", data}}.dump(),
			"application/json");
	});
	g_vll->Post("/v1/chat/completions", [](const httplib::Request &,
			httplib::Response &r)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_vll_log.push_back("TRAP chat");
		r.status = 500;
	});
	int	op = start(g_oll, &g_oll_thr);
	int	vp = start(g_vll, &g_vll_thr);

	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(op)).c_str(), 1);
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, ("http://127.0.0.1:"
		+ std::to_string(vp)).c_str(), 1);
}

static void	test_normalize(void)
{
	TEST_ASSERT(membrane_orch_normalize_model("Qwen2.5-1.5B")
		== membrane_orch_normalize_model("qwen2.5:1.5b"), "separators and "
			"case");
	TEST_ASSERT(membrane_orch_normalize_model("llama3:latest")
		== membrane_orch_normalize_model("llama3"), ":latest is implicit");
	TEST_ASSERT(membrane_orch_normalize_model("qwen2.5-1.5b-instruct")
		!= membrane_orch_normalize_model("qwen2.5:1.5b"), "no silent fuzzy "
			"widening: 'instruct' is a different model id");
}

static void	test_auto_gathers_every_runtime(void)
{
	int			rc;
	json		j;
	std::string	out;

	g_oll_log.clear();
	g_vll_log.clear();
	out = capture({"qwen2.5:1.5b", "--explain"}, true, &rc);
	j = json::parse(out, nullptr, false);
	TEST_ASSERT(!j.is_discarded() && rc == MEMBRANE_EXIT_SUCCESS, "valid JSON");
	TEST_ASSERT(j["candidates"].size() == 3 && j["candidates"][0]["runtime"]
			== "membrane-native" && j["candidates"][1]["runtime"] == "ollama"
		&& j["candidates"][2]["runtime"] == "vllm", "every runtime considered, "
			"in table order");
	TEST_ASSERT(j["candidates"][0]["viable"] == false
		&& j["candidates"][0]["exclusion"]["code"] == "MODEL_NOT_ON_RUNTIME",
		"native: model not installed here");
	TEST_ASSERT(j["selected"] == "ollama" && j["confidence"] == "high"
		&& j["ranking"].size() == 2 && j["ranking"][1] == "vllm",
		"both resident; Ollama exposes more state => selected; vLLM is the "
		"alternative");
	TEST_ASSERT(j["candidates"][1]["evidence"] == "resident"
		&& j["candidates"][1]["configuration"]["context"].is_null()
		&& j["candidates"][1]["configuration"]["gpu_layers"].is_null(),
		"no fabricated Planner-style configuration for an external runtime");
	TEST_ASSERT(j["candidates"][2]["model"] == "Qwen2.5-1.5B"
		&& j["candidates"][2]["memory"].is_null(), "vLLM: matched by "
			"normalized name; no invented footprint");
	TEST_ASSERT(j["hardware"]["ram_total_bytes"].is_number(), "hardware");
	/* route trap + read-only */
	for (const auto &l : g_oll_log)
		TEST_ASSERT(l.compare(0, 3, "GET") == 0, "Ollama: GETs only");
	for (const auto &l : g_vll_log)
		TEST_ASSERT(l.compare(0, 3, "GET") == 0, "vLLM: GETs only");
	/* deterministic: identical facts => identical decision. (The hardware
	 * block is a live measurement and legitimately differs between calls;
	 * everything that is a function of the gathered facts must not.) */
	json	again = json::parse(capture({"qwen2.5:1.5b", "--explain"}, true,
			&rc));
	json	first = j;

	first.erase("hardware");
	again.erase("hardware");
	TEST_ASSERT(first.dump() == again.dump(), "identical facts => "
		"byte-identical decision JSON");
}

static void	test_human_output_explains(void)
{
	int			rc;
	std::string	out;

	out = capture({"qwen2.5:1.5b", "--explain"}, false, &rc);
	TEST_ASSERT(out.find("Runtime choice for \"qwen2.5:1.5b\"")
			!= std::string::npos
		&& out.find("selected") != std::string::npos
		&& out.find("Why:") != std::string::npos
		&& out.find("Confidence: high") != std::string::npos
		&& out.find("nothing was changed") != std::string::npos
		&& out.find("not used -- ") != std::string::npos,
		"choice + why + confidence + exclusions");
	TEST_ASSERT(out.find('\033') == std::string::npos, "no ANSI when "
		"redirected");
	TEST_ASSERT(out.find("What MEMBRANE cannot control on ollama")
		!= std::string::npos && out.find("KV-precision control: unsupported")
		!= std::string::npos, "control limitations are disclosed");
	out = capture({"qwen2.5:1.5b"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("does not launch or "
			"control ollama") != std::string::npos, "external runtime: "
				"honest about not controlling it");
}

static void	test_forced_runtime(void)
{
	int		rc;
	json	j = json::parse(capture({"qwen2.5:1.5b", "--runtime", "vllm",
		"--explain"}, true, &rc));

	TEST_ASSERT(j["candidates"].size() == 1 && j["selected"] == "vllm",
		"--runtime forces one runtime");
	capture({"x", "--runtime", "lmstudio"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown runtime");
}

static void	test_not_resident_is_an_estimate(void)
{
	int		rc;
	json	j;

	g_oll_resident = false;
	g_vll_has_model = false;
	j = json::parse(capture({"qwen2.5:1.5b", "--explain"}, true, &rc));
	TEST_ASSERT(j["selected"] == "ollama" && j["candidates"][1]["resident"]
			== false && j["candidates"][1]["evidence"] == "estimate"
		&& j["confidence"] == "medium", "only a reported-size estimate, so "
			"'medium', never 'planned'");
	TEST_ASSERT(j["candidates"][2]["exclusion"]["code"]
		== "MODEL_NOT_ON_RUNTIME", "vLLM is not serving it");
	g_oll_resident = true;
	g_vll_has_model = true;
}

static void	test_nothing_found(void)
{
	int			rc;
	json		j;
	std::string	out;

	g_oll_has_model = false;
	g_vll_has_model = false;
	out = capture({"no-such-model"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_MODEL_ERROR, "absent everywhere => "
		"model error");
	TEST_ASSERT(out.find("membrane use no-such-model") != std::string::npos
		&& out.find("membrane model list") != std::string::npos,
		"next-action guidance with commands that exist");
	j = json::parse(capture({"no-such-model"}, true, &rc));
	TEST_ASSERT(j["selected"].is_null() && j["reasons"][0]
		== "NO_VIABLE_RUNTIME", "JSON: no selection, reason code");
	g_oll_has_model = true;
	g_vll_has_model = true;
}

static void	test_unavailable_runtimes_are_explained(void)
{
	int			rc;
	json		j;
	std::string	saved_o = getenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
	std::string	saved_v = getenv(MEMBRANE_VLLM_ENDPOINT_ENV);

	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, "http://127.0.0.1:1", 1);
	j = json::parse(capture({"qwen2.5:1.5b", "--explain"}, true, &rc));
	TEST_ASSERT(j["selected"].is_null() && (rc == MEMBRANE_EXIT_MODEL_ERROR
		|| rc == MEMBRANE_EXIT_RUNTIME_ERROR), "nothing can serve it");
	TEST_ASSERT(j["candidates"][1]["exclusion"]["code"]
			== "RUNTIME_UNAVAILABLE"
		&& j["candidates"][1]["exclusion"]["text"].get<std::string>()
			.find("not reachable") != std::string::npos,
		"the runtime's own reason is shown");
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, saved_o.c_str(), 1);
	setenv(MEMBRANE_VLLM_ENDPOINT_ENV, saved_v.c_str(), 1);
}

static void	test_intents_and_errors(void)
{
	int		rc;
	json	j = json::parse(capture({"qwen2.5:1.5b", "--intent",
		"long_context", "--explain"}, true, &rc));

	TEST_ASSERT(j["intent"] == "long_context" && j["selected"] == "vllm",
		"long_context: vLLM's 32768 beats Ollama's unknown max");
	j = json::parse(capture({"qwen2.5:1.5b", "--intent", "memory_saving",
		"--explain"}, true, &rc));
	TEST_ASSERT(j["selected"] == "ollama", "memory_saving: only Ollama has a "
		"known footprint");
	capture({"qwen2.5:1.5b", "--intent", "throughput"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "throughput is not an intent");
	capture({}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "usage");
	capture({"a", "b"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "stray argument");
}

static void	test_no_inference_ever(void)
{
	std::lock_guard<std::mutex>	l(g_mtx);

	for (const auto &x : g_oll_log)
		TEST_ASSERT(x.compare(0, 4, "TRAP") != 0, "no Ollama inference/"
			"mutation route was reached");
	for (const auto &x : g_vll_log)
		TEST_ASSERT(x.compare(0, 4, "TRAP") != 0, "no vLLM inference route "
			"was reached");
}

int	main(void)
{
	char	tmpl[] = "/tmp/membrane-orch-test-XXXXXX";

	TEST_ASSERT(mkdtemp(tmpl) != NULL, "temp dir");
	g_dir = tmpl;
	setenv("MEMBRANE_MODELS_PATH", (g_dir + "/models.json").c_str(), 1);
	setenv("MEMBRANE_SERVER_CONFIG_PATH", (g_dir + "/server.json").c_str(), 1);
	start_mocks();
	test_normalize();
	test_auto_gathers_every_runtime();
	test_human_output_explains();
	test_forced_runtime();
	test_not_resident_is_an_estimate();
	test_nothing_found();
	test_unavailable_runtimes_are_explained();
	test_intents_and_errors();
	test_no_inference_ever();
	g_oll->stop();
	g_vll->stop();
	g_oll_thr->join();
	g_vll_thr->join();
	{
		std::string	cmd = "rm -rf '" + g_dir + "'";

		if (system(cmd.c_str()) != 0)
			fprintf(stderr, "warning: cleanup failed\n");
	}
	printf("test_orchestrate_cmd: all tests passed\n");
	return (0);
}
