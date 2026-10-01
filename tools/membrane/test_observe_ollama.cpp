#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "observe_cmd.h"
#include "observe_ollama.h"
#include "runtime_ollama.h"
#include "product_cli.h"
#include "service_state.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone I2: `membrane observe --runtime ollama`'s own tests.
 *
 * Two layers, mirroring observe_ollama.h's two stages (same split
 * test_observe_cmd.cpp uses for the native provider):
 *   - membrane_observe_ollama_build_snapshot() driven with SYNTHETIC raw
 *     inputs -- deterministic, no network at all.
 *   - the REAL provider end to end against an in-process mock Ollama
 *     server (never a real daemon -- MEMBRANE_OLLAMA_ENDPOINT is always
 *     pinned to the mock or to a closed port) plus a route-trap mock that
 *     proves only GET /api/version and GET /api/ps are ever requested.
 */

/* ---------------------------------------------------------------- */
/* Layer 1: pure snapshot assembly from synthetic inputs             */
/* ---------------------------------------------------------------- */

static membrane_observe_ollama_inputs_t	available_inputs(void)
{
	membrane_observe_ollama_inputs_t	in;

	membrane_observe_ollama_inputs_init(&in);
	in.timestamp_unix_ms = 1790246096789LL;
	in.collection_duration_ms = 12;
	memset(&in.runtime, 0, sizeof(in.runtime));
	in.runtime.availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	in.runtime.endpoint_known = 1;
	strncpy(in.runtime.endpoint, "http://127.0.0.1:11434",
		sizeof(in.runtime.endpoint) - 1);
	in.runtime.version_known = 1;
	strncpy(in.runtime.version, "0.34.3", sizeof(in.runtime.version) - 1);
	in.meminfo_probed = true;
	in.meminfo.ok = true;
	in.meminfo.total_bytes = 8000;
	in.meminfo.available_bytes = 3000;
	in.swap_supported = true;
	in.meminfo_source = "proc_meminfo";
	in.gpu_enumerated = true;
	in.ps_called = true;
	in.ps_ok = true;
	return (in);
}

static membrane_ollama_process_model_t	full_model(const char *name)
{
	membrane_ollama_process_model_t	m;

	m.name = name;
	m.model = name;
	m.digest = "845dbda0ea48";
	m.family = "qwen2";
	m.quant = "Q4_K_M";
	m.size_known = true;
	m.size_bytes = 4683087332ull;
	m.size_vram_known = true;
	m.size_vram_bytes = 4683087332ull;
	m.context_length_known = true;
	m.context_length = 4096;
	m.expires_at = "2026-09-24T14:38:31.83753Z";
	return (m);
}

static void	test_unavailable_snapshot(void)
{
	membrane_observe_ollama_inputs_t	in;

	membrane_observe_ollama_inputs_init(&in);
	memset(&in.runtime, 0, sizeof(in.runtime));
	in.runtime.availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(strcmp(s.runtime_id, MEMBRANE_RUNTIME_ID_OLLAMA) == 0,
		"runtime id is ollama");
	TEST_ASSERT(!s.runtime_observable, "unavailable -> not observable");
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"unavailable status");
	TEST_ASSERT(!s.ram_total_bytes.known,
		"nothing is reported for an unobservable runtime -- /api/ps was "
		"never even attempted");
	TEST_ASSERT(!s.resident_model_count.known, "resident count unknown too");
}

static void	test_zero_loaded_models(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	in.models.clear();

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.runtime_observable, "runtime observable");
	TEST_ASSERT(s.status != MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"a healthy daemon with zero loaded models is NOT unavailable");
	TEST_ASSERT(s.resident_model_count.known && s.resident_model_count.value
		== 0, "zero is a real, known count, not unknown");
	TEST_ASSERT(s.resident_model_count.provenance
		== MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "count is runtime_reported");
	TEST_ASSERT(s.resident_model_len == 0, "no resident model entries");
	TEST_ASSERT(!s.context_active.known
		&& strcmp(s.context_active.source, "no_loaded_model") == 0,
		"no active context when nothing is loaded");
}

static void	test_one_model_full_fields(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	in.models.push_back(full_model("qwen2.5:7b"));

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.resident_model_count.known && s.resident_model_count.value
		== 1, "count 1");
	TEST_ASSERT(s.resident_model_len == 1, "1 resident entry");

	const membrane_obs_resident_model_t	&r = s.resident_models[0];

	TEST_ASSERT(r.name.known && strcmp(r.name.value, "qwen2.5:7b") == 0,
		"name");
	TEST_ASSERT(r.state.known && strcmp(r.state.value, "loaded") == 0,
		"state loaded");
	TEST_ASSERT(r.state.provenance == MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"state is runtime_reported");
	TEST_ASSERT(r.digest.known && strcmp(r.digest.value, "845dbda0ea48")
		== 0, "digest");
	TEST_ASSERT(r.family.known && strcmp(r.family.value, "qwen2") == 0,
		"family");
	TEST_ASSERT(r.quant.known && strcmp(r.quant.value, "Q4_K_M") == 0,
		"quant");
	TEST_ASSERT(r.reported_size_bytes.known
		&& r.reported_size_bytes.value == 4683087332ull
		&& r.reported_size_bytes.provenance
			== MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"reported_size_bytes is runtime_reported, never estimated (Part 7)");
	TEST_ASSERT(r.reported_gpu_bytes.known
		&& r.reported_gpu_bytes.value == 4683087332ull
		&& r.reported_gpu_bytes.provenance
			== MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"reported_gpu_bytes (size_vram) is runtime_reported and "
		"model-specific");
	TEST_ASSERT(r.reported_context.known && r.reported_context.value == 4096,
		"reported_context");
	TEST_ASSERT(r.expires_at.known, "expires_at known");
	/* Native-only fields stay unknown, never guessed. */
	TEST_ASSERT(!r.backend.known && !r.gpu_layers.known
		&& !r.kv_precision.known && !r.estimated_model_bytes.known
		&& !r.estimated_kv_bytes.known,
		"native-only per-model fields are unknown for Ollama");

	/* Exactly one loaded model -> the top-level active context IS filled
	 * (Part 9), unlike the zero/multiple cases. */
	TEST_ASSERT(s.context_active.known && s.context_active.value == 4096
		&& s.context_active.provenance == MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"single loaded model -> active context filled");
}

static void	test_multiple_models_not_discarded(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	in.models.push_back(full_model("qwen2.5:7b"));
	in.models.push_back(full_model("llama3.2:3b"));

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.resident_model_count.value == 2, "count 2");
	TEST_ASSERT(s.resident_model_len == 2, "both entries kept, not just "
		"the first");
	TEST_ASSERT(strcmp(s.resident_models[0].name.value, "qwen2.5:7b") == 0
		&& strcmp(s.resident_models[1].name.value, "llama3.2:3b") == 0,
		"both models present in order");
	/* Part 9: with more than one loaded model, a single "the" active
	 * context is ambiguous -- honestly unknown, never one model picked
	 * arbitrarily. */
	TEST_ASSERT(!s.context_active.known
		&& strcmp(s.context_active.source,
			"ambiguous_multiple_loaded_models") == 0,
		"multiple loaded models -> active context ambiguous, not guessed");
}

static void	test_resident_array_cap_keeps_true_count(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	for (int i = 0; i < MEMBRANE_OBS_MAX_RESIDENT + 3; ++i)
		in.models.push_back(full_model(("m" + std::to_string(i)).c_str()));

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.resident_model_count.value
		== (uint64_t)(MEMBRANE_OBS_MAX_RESIDENT + 3),
		"count reflects the TRUE total even beyond the detail cap");
	TEST_ASSERT(s.resident_model_len == MEMBRANE_OBS_MAX_RESIDENT,
		"the detail array itself is capped");
}

static void	test_missing_optional_fields_partial(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();
	membrane_ollama_process_model_t	m;

	m.name = "bare:latest";
	m.model = "bare:latest";
	m.size_known = false;
	m.size_vram_known = false;
	m.context_length_known = false;
	in.models.push_back(m);

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.resident_model_len == 1, "the entry is still present");
	const membrane_obs_resident_model_t	&r = s.resident_models[0];

	TEST_ASSERT(r.name.known, "name still known");
	TEST_ASSERT(!r.digest.known && !r.reported_size_bytes.known
		&& !r.reported_gpu_bytes.known && !r.reported_context.known
		&& !r.expires_at.known,
		"missing optional fields are unknown, not fabricated");
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_PARTIAL,
		"one broken/missing field set -> partial, not unavailable "
		"(Part 16)");
}

static void	test_ps_failure_degrades_to_partial(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	in.ps_ok = false;
	in.ps_error.code = MEMBRANE_RUNTIME_ERR_MALFORMED;
	in.ps_error.message = "response is not valid JSON";

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(s.runtime_observable, "Ollama itself was still reached");
	TEST_ASSERT(s.status != MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"a broken /api/ps degrades this section only, never the whole "
		"observation (Part 16)");
	TEST_ASSERT(!s.resident_model_count.known
		&& strcmp(s.resident_model_count.source, "ps_malformed_response")
			== 0, "resident count unknown with the ps error reason");
	/* Host/GPU (device-wide, independent of /api/ps) are unaffected. */
	TEST_ASSERT(s.ram_total_bytes.known, "host memory still reported");
}

static void	test_device_wide_vs_model_specific(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();
	membrane_gpu_device_info_t			dev;

	memset(&dev, 0, sizeof(dev));
	strncpy(dev.name, "GTX1650", sizeof(dev.name) - 1);
	strncpy(dev.description, "NVIDIA GTX 1650", sizeof(dev.description) - 1);
	strncpy(dev.backend, "Vulkan", sizeof(dev.backend) - 1);
	dev.type = MEMBRANE_DEV_TYPE_GPU;
	dev.memory_total = 4ull << 30;
	dev.memory_free = 3ull << 30;
	in.devices.push_back(dev);
	in.models.push_back(full_model("qwen2.5:7b"));

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	/* Part 8: device-wide VRAM (MEMBRANE's own probe) and this model's
	 * own GPU allocation (Ollama's size_vram) are DIFFERENT fields with
	 * different provenance -- one is never substituted for the other. */
	TEST_ASSERT(s.vram_total_bytes.known
		&& s.vram_total_bytes.value == (4ull << 30)
		&& s.vram_total_bytes.provenance == MEMBRANE_OBS_PROV_MEASURED,
		"device-wide VRAM total is measured, from MEMBRANE's own probe");
	TEST_ASSERT(s.resident_models[0].reported_gpu_bytes.known
		&& s.resident_models[0].reported_gpu_bytes.value == 4683087332ull
		&& s.resident_models[0].reported_gpu_bytes.provenance
			== MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"this model's own GPU allocation is runtime_reported, a separate "
		"number entirely");
	TEST_ASSERT(s.vram_total_bytes.provenance
		!= s.resident_models[0].reported_gpu_bytes.provenance,
		"device-wide and model-specific GPU bytes carry different "
		"provenance -- they are never conflated into one number");
}

static void	test_not_applicable_fields(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	membrane_observation_snapshot_t	s;

	membrane_observe_ollama_build_snapshot(in, &s);
	TEST_ASSERT(!s.configured_model.known
		&& strcmp(s.configured_model.source,
			"not_applicable_external_runtime") == 0,
		"MEMBRANE's own default-model config does not apply to Ollama");
	TEST_ASSERT(!s.context_planned.known && !s.kv_estimated_bytes.known,
		"Planner v2 does not run against an externally-owned model (I3 "
		"owns recommendation/comparison work, not I2)");
	TEST_ASSERT(!s.service_installed.known
		&& strcmp(s.service_installed.source,
			"not_applicable_external_runtime") == 0,
		"MEMBRANE never probes an OS service manager for Ollama");
	TEST_ASSERT(s.server_endpoint.known
		&& s.server_endpoint.provenance == MEMBRANE_OBS_PROV_CONFIGURED,
		"the Ollama endpoint itself DOES apply");
	TEST_ASSERT(s.server_reachable.known && s.server_reachable.value,
		"reachable, by construction of the available branch");
	TEST_ASSERT(s.server_version.known
		&& strcmp(s.server_version.value, "0.34.3") == 0
		&& s.server_version.provenance == MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"server_version from /api/version, runtime_reported");
}

static void	test_json_deterministic(void)
{
	membrane_observe_ollama_inputs_t	in = available_inputs();

	in.models.push_back(full_model("qwen2.5:7b"));

	membrane_observation_snapshot_t	sa;
	membrane_observation_snapshot_t	sb;

	membrane_observe_ollama_build_snapshot(in, &sa);
	membrane_observe_ollama_build_snapshot(in, &sb);
	json	a = membrane_observe_snapshot_json(sa);
	json	b = membrane_observe_snapshot_json(sb);

	TEST_ASSERT(a.dump() == b.dump(), "same inputs -> byte-identical JSON");
	TEST_ASSERT(a["schema_version"] == MEMBRANE_OBSERVATION_SCHEMA_VERSION
		&& MEMBRANE_OBSERVATION_SCHEMA_VERSION == 2,
		"schema_version 2 (I2 grew resident_models[] fields)");
	TEST_ASSERT(a["model"]["resident_models"][0].contains("reported_gpu_bytes")
		&& a["model"]["resident_models"][0].contains("expires_at"),
		"the new I2 keys are present in resident_models[] JSON");
}

/* ---------------------------------------------------------------- */
/* Pure GET /api/ps parser                                          */
/* ---------------------------------------------------------------- */

static void	test_parse_ps(void)
{
	std::vector<membrane_ollama_process_model_t>	models;
	std::string										err;

	/* Zero loaded models -- valid, not an error (Part 14). */
	TEST_ASSERT(membrane_ollama_parse_ps(R"({"models": []})", &models, &err)
		&& models.empty(), "empty models array parses to an empty vector");

	/* A realistic v0.34.3-shaped entry (docs/openapi.yaml example). */
	const char	*body = R"({
	  "models": [
	    {
	      "name": "gemma4",
	      "model": "gemma4",
	      "size": 6591830464,
	      "digest": "c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb",
	      "details": {
	        "parent_model": "", "format": "gguf", "family": "gemma4",
	        "families": ["gemma4"], "parameter_size": "8.0B",
	        "quantization_level": "Q4_K_M"
	      },
	      "expires_at": "2026-10-17T16:47:07.93355-07:00",
	      "size_vram": 5333539264,
	      "context_length": 4096
	    }
	  ]
	})";

	TEST_ASSERT(membrane_ollama_parse_ps(body, &models, &err)
		&& models.size() == 1, "one model parsed");
	TEST_ASSERT(models[0].name == "gemma4" && models[0].family == "gemma4"
		&& models[0].quant == "Q4_K_M" && models[0].size_known
		&& models[0].size_bytes == 6591830464ull && models[0].size_vram_known
		&& models[0].size_vram_bytes == 5333539264ull
		&& models[0].context_length_known && models[0].context_length == 4096
		&& !models[0].expires_at.empty(), "every documented field mapped");

	/* size_vram entirely absent -- documented Ollama quirk (#4840): a
	 * known 0, not unknown. */
	const char	*no_vram = R"({"models": [{"name": "x", "size": 10}]})";

	TEST_ASSERT(membrane_ollama_parse_ps(no_vram, &models, &err)
		&& models[0].size_vram_known && models[0].size_vram_bytes == 0,
		"absent size_vram is a known 0, per Ollama's own omitempty "
		"behavior");

	/* Malformed. */
	TEST_ASSERT(!membrane_ollama_parse_ps("not json", &models, &err),
		"invalid JSON rejected");
	TEST_ASSERT(!membrane_ollama_parse_ps(R"({"foo": 1})", &models, &err),
		"missing \"models\" array rejected");
	TEST_ASSERT(!membrane_ollama_parse_ps(R"({"models": [1, 2]})", &models,
		&err), "non-object model entry rejected");
	TEST_ASSERT(!membrane_ollama_parse_ps(R"({"models": [{}]})", &models,
		&err), "entry with no name rejected");

	/* Unknown extra fields are ignored, never fail the parse. */
	const char	*extra = R"({"models": [{"name": "x", "size": 1,
		"totally_unknown_future_field": {"nested": true}}]})";

	TEST_ASSERT(membrane_ollama_parse_ps(extra, &models, &err)
		&& models.size() == 1, "unrecognized fields are ignored, not fatal");
}

static void	test_allowlist_includes_ps(void)
{
	TEST_ASSERT(membrane_ollama_request_allowed("GET", "/api/ps"),
		"GET /api/ps is now on the allowlist (I2)");
	TEST_ASSERT(membrane_ollama_request_allowed("GET", "/api/version")
		&& membrane_ollama_request_allowed("GET", "/api/tags")
		&& membrane_ollama_request_allowed("POST", "/api/show"),
		"the 3 H2 calls are still allowed");
	TEST_ASSERT(!membrane_ollama_request_allowed("POST", "/api/ps")
		&& !membrane_ollama_request_allowed("DELETE", "/api/ps"),
		"only GET /api/ps -- exact method+path match");
	for (const char *p : {"/api/generate", "/api/chat", "/api/pull",
			"/api/push", "/api/create", "/api/copy", "/api/delete",
			"/v1/chat/completions"})
		TEST_ASSERT(!membrane_ollama_request_allowed("GET", p)
			&& !membrane_ollama_request_allowed("POST", p),
			"no other route was widened by I2");
}

/* ---------------------------------------------------------------- */
/* Layer 2: real provider against an in-process mock Ollama server   */
/* ---------------------------------------------------------------- */

static httplib::Server				*g_mock;
static std::thread					*g_mock_thread;
static int							g_mock_port;
static std::mutex					g_mock_mutex;
static std::vector<std::string>	g_mock_log;
static int							g_version_status;
static std::string					g_version_body;
static int							g_ps_status;
static std::string					g_ps_body;

static void	reset_mock(void)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);

	g_version_status = 200;
	g_version_body = R"({"version":"0.34.3"})";
	g_ps_status = 200;
	g_ps_body = R"({"models": []})";
	g_mock_log.clear();
}

static void	trap(const httplib::Request &, httplib::Response &res)
{
	res.status = 500;
	res.set_content("{\"error\":\"trap route hit\"}", "application/json");
}

static void	start_mock(void)
{
	reset_mock();
	g_mock = new httplib::Server();
	g_mock->set_logger([](const httplib::Request &req, const httplib::Response &)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_mock_log.push_back(req.method + " " + req.path);
	});
	g_mock->Get("/api/version", [](const httplib::Request &,
			httplib::Response &res)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		res.status = g_version_status;
		res.set_content(g_version_body, "application/json");
	});
	g_mock->Get("/api/ps", [](const httplib::Request &, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		res.status = g_ps_status;
		res.set_content(g_ps_body, "application/json");
	});
	/* Every mutating / inference / cloud / inventory-metadata route this
	 * command must never reach (Part 23). */
	for (const char *p : {"/api/generate", "/api/chat", "/api/embed",
			"/api/embeddings", "/api/pull", "/api/push", "/api/create",
			"/api/copy", "/v1/chat/completions", "/v1/completions",
			"/v1/embeddings"})
		g_mock->Post(p, trap);
	g_mock->Delete("/api/delete", trap);
	g_mock->Get("/api/tags", trap);
	g_mock->Post("/api/show", trap);
	g_mock->Get("/v1/models", trap);
	g_mock_port = g_mock->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_mock_port > 0, "mock bound an ephemeral port");
	g_mock_thread = new std::thread([]() { g_mock->listen_after_bind(); });

	int	attempts = 0;

	while (!g_mock->is_running() && attempts < 100)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		attempts++;
	}
	TEST_ASSERT(g_mock->is_running(), "mock became ready");
}

static void	stop_mock(void)
{
	g_mock->stop();
	g_mock_thread->join();
	delete g_mock_thread;
	delete g_mock;
	g_mock = NULL;
	g_mock_thread = NULL;
}

static std::string	mock_url(void)
{
	return ("http://127.0.0.1:" + std::to_string(g_mock_port));
}

static std::vector<std::string>	mock_log(void)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);

	return (g_mock_log);
}

static std::string	read_capture_file(const std::string &p)
{
	FILE		*f = fopen(p.c_str(), "rb");
	std::string	out;
	char		buf[4096];
	size_t		n;

	if (f == NULL)
		return ("<absent>");
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	fclose(f);
	return (out);
}

static std::string	capture_dispatch(const std::vector<std::string> &args,
				bool want_json, int *rc)
{
	char	tmpl[] = "/tmp/membrane-observe-ollama-test-XXXXXX";
	int		fd = mkstemp(tmpl);

	TEST_ASSERT(fd >= 0, "capture file created");
	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_observe_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	std::string	out = read_capture_file(tmpl);

	unlink(tmpl);
	return (out);
}

/* A loopback port nothing listens on (never listen(), so connecting is
 * refused immediately -- see test_observe_cmd.cpp's own comment on why
 * httplib's bind_to_any_port is not used for this). */
static int	closed_port(void)
{
	int					fd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in	addr;
	socklen_t			len = sizeof(addr);

	TEST_ASSERT(fd >= 0, "socket()");
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = 0;
	TEST_ASSERT(bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0,
		"bind ephemeral port");
	TEST_ASSERT(getsockname(fd, (struct sockaddr *)&addr, &len) == 0,
		"getsockname");
	close(fd);
	return ((int)ntohs(addr.sin_port));
}

static void	use_mock_endpoint(void)
{
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, mock_url().c_str(), 1);
}

static void	use_closed_endpoint(void)
{
	std::string	url = "http://127.0.0.1:" + std::to_string(closed_port());

	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, url.c_str(), 1);
}

static void	test_unavailable_cli_has_no_socket_error(void)
{
	int			rc;
	std::string	out;

	use_closed_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR,
		"unavailable Ollama -> RUNTIME_ERROR exit, not a CLI error");
	TEST_ASSERT(out.find("unavailable") != std::string::npos,
		"status is unavailable");
	TEST_ASSERT(out.find("Connection refused") == std::string::npos
		&& out.find("errno") == std::string::npos
		&& out.find("ECONNREFUSED") == std::string::npos,
		"no raw socket error text is ever printed");

	out = capture_dispatch({"--runtime", "ollama"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR, "same in human mode");
	TEST_ASSERT(out.find("unavailable") != std::string::npos, "human output "
		"also says unavailable");
}

static void	test_end_to_end_healthy_zero_models(void)
{
	int			rc;
	std::string	out;

	use_mock_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "healthy + zero models -> 0");

	json	j = json::parse(out);

	TEST_ASSERT(j["runtime"]["id"] == "ollama", "runtime id");
	TEST_ASSERT(j["runtime"]["observable"] == true, "observable");
	TEST_ASSERT(j["model"]["resident_count"]["value"] == 0
		&& j["model"]["resident_count"]["known"] == true, "known zero");
	TEST_ASSERT(j["model"]["resident_models"].empty(), "empty array");
}

static void	test_end_to_end_one_model(void)
{
	int			rc;
	std::string	out;

	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_ps_body = R"({"models": [{
			"name": "qwen2.5:7b", "model": "qwen2.5:7b",
			"size": 4683087332, "digest": "845dbda0ea48",
			"details": {"family": "qwen2", "quantization_level": "Q4_K_M"},
			"expires_at": "2026-09-24T14:38:31.83753Z",
			"size_vram": 4683087332, "context_length": 4096
		}]})";
	}
	use_mock_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "one model -> 0");

	json	j = json::parse(out);
	json	m = j["model"]["resident_models"][0];

	TEST_ASSERT(m["name"]["value"] == "qwen2.5:7b", "name in JSON");
	TEST_ASSERT(m["reported_size_bytes"]["provenance"] == "runtime_reported",
		"reported_size_bytes provenance in end-to-end JSON");
	TEST_ASSERT(m["reported_gpu_bytes"]["value"] == 4683087332ull,
		"reported_gpu_bytes value in end-to-end JSON");
	TEST_ASSERT(j["context"]["active"]["value"] == 4096, "top-level active "
		"context filled for the single loaded model");

	out = capture_dispatch({"--runtime", "ollama"}, false, &rc);
	TEST_ASSERT(out.find("qwen2.5:7b") != std::string::npos
		&& out.find("runtime_reported") != std::string::npos,
		"human output shows the model and its provenance");
}

static void	test_malformed_ps_non2xx_oversized(void)
{
	int			rc;
	std::string	out;

	use_mock_endpoint();
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);
		g_ps_body = "not json";
	}
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "malformed /api/ps degrades "
		"to partial, not a failure exit -- Ollama itself was reached");
	json	j = json::parse(out);
	TEST_ASSERT(j["status"] == "partial", "partial status");
	TEST_ASSERT(j["model"]["resident_count"]["known"] == false, "resident "
		"count unknown");

	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);
		g_ps_body = R"({"models": []})";
		g_ps_status = 500;
	}
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "non-2xx /api/ps also "
		"degrades to partial");
	j = json::parse(out);
	TEST_ASSERT(j["model"]["resident_count"]["known"] == false,
		"resident count unknown on HTTP error too");

	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);
		g_ps_status = 200;
		g_ps_body = std::string("{\"models\": [") + std::string(20 * 1024 * 1024, 'x')
			+ "]}";
	}
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "oversized /api/ps also "
		"degrades cleanly, never crashes");
	j = json::parse(out);
	TEST_ASSERT(j["model"]["resident_count"]["known"] == false,
		"resident count unknown on an oversized response");
	reset_mock();
}

/* ---------------------------------------------------------------- */
/* Route allowlist / read-only proof (Parts 19, 22-O/P/Q, 23)        */
/* ---------------------------------------------------------------- */

static void	test_only_allowlisted_routes_hit(void)
{
	int			rc;

	reset_mock();
	use_mock_endpoint();
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);
		g_ps_body = R"({"models": [{"name": "qwen2.5:7b", "size": 1,
			"size_vram": 1, "context_length": 1}]})";
	}
	capture_dispatch({"--runtime", "ollama"}, true, &rc);
	capture_dispatch({"--runtime", "ollama"}, false, &rc);

	std::vector<std::string>	log = mock_log();

	TEST_ASSERT(!log.empty(), "the command did reach the mock");
	std::set<std::string>	distinct(log.begin(), log.end());

	TEST_ASSERT(distinct.size() == 2
		&& distinct.count("GET /api/version") == 1
		&& distinct.count("GET /api/ps") == 1,
		"exactly GET /api/version and GET /api/ps, nothing else");
	for (const auto &line : log)
	{
		size_t		sp = line.find(' ');
		std::string	method = line.substr(0, sp);
		std::string	path = line.substr(sp + 1);

		TEST_ASSERT(membrane_ollama_request_allowed(method, path),
			"every request seen is on the read-only allowlist");
		TEST_ASSERT(path != "/api/generate" && path != "/api/chat"
			&& path.compare(0, 4, "/v1/") != 0,
			"no inference request was ever sent");
		TEST_ASSERT(path != "/api/pull" && path != "/api/delete"
			&& path != "/api/create" && path != "/api/copy"
			&& path != "/api/push",
			"no mutating request was ever sent");
		TEST_ASSERT(path != "/api/tags" && path != "/api/show",
			"no /api/tags or /api/show call -- minimal /api/ps-only "
			"observation (Part 5)");
	}
	reset_mock();
}

static void	test_no_state_mutation(void)
{
	char	tmpl[] = "/tmp/membrane-observe-ollama-isolated-XXXXXX";
	char	*dir_c = mkdtemp(tmpl);

	TEST_ASSERT(dir_c != NULL, "isolated dir created");
	std::string	dir(dir_c);

	setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
	setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(), 1);
	setenv("MEMBRANE_SYSTEMD_USER_DIR", (dir + "/systemd").c_str(), 1);

	use_mock_endpoint();
	int	rc;

	capture_dispatch({"--runtime", "ollama"}, true, &rc);
	capture_dispatch({"--runtime", "ollama"}, false, &rc);

	std::string	cmd = "[ -z \"$(ls -A '" + dir + "' 2>/dev/null)\" ]";

	TEST_ASSERT(system(cmd.c_str()) == 0,
		"no registry/config/service file was ever created");
	cmd = "rm -rf '" + dir + "'";
	if (system(cmd.c_str()) != 0)
		fprintf(stderr, "warning: cleanup of %s failed\n", dir.c_str());
	unsetenv("MEMBRANE_MODELS_PATH");
	unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
	unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
}

/* ---------------------------------------------------------------- */
/* Native regression (Part 20)                                       */
/* ---------------------------------------------------------------- */

static void	test_native_unchanged(void)
{
	/* The native path (unlike --runtime ollama) reads MEMBRANE's own
	 * config/registry -- isolate it from the real host, same as
	 * test_observe_cmd.cpp's own s_isolated_env, so this test never
	 * touches real local state. */
	char	tmpl[] = "/tmp/membrane-observe-ollama-native-XXXXXX";
	char	*dir_c = mkdtemp(tmpl);

	TEST_ASSERT(dir_c != NULL, "isolated dir created");
	std::string	dir(dir_c);

	setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
	setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(), 1);
	setenv("MEMBRANE_SYSTEMD_USER_DIR", (dir + "/systemd").c_str(), 1);
	setenv("MEMBRANE_LAUNCHD_USER_DIR", (dir + "/launchd").c_str(), 1);
	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);

	int			rc;
	std::string	out_json = capture_dispatch({}, true, &rc);
	std::string	out_json2 = capture_dispatch({"--runtime",
			MEMBRANE_RUNTIME_ID_NATIVE}, true, &rc);

	json	a = json::parse(out_json);
	json	b = json::parse(out_json2);

	TEST_ASSERT(a["runtime"]["id"] == "membrane-native"
		&& b["runtime"]["id"] == "membrane-native",
		"default and explicit --runtime membrane-native agree");
	TEST_ASSERT(a["schema_version"] == 2,
		"native path also reports the new schema_version (I2 bumped it "
		"for everyone, since resident_models[] grew for BOTH providers)");
	TEST_ASSERT(a["model"].contains("resident_models"),
		"native resident_models[] key still present");

	std::string	unknown_runtime = capture_dispatch({"--runtime", "bogus"},
			true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown runtime -> CLI "
		"error, unchanged");
	TEST_ASSERT(unknown_runtime.find("unknown runtime") != std::string::npos,
		"unchanged error text shape");

	std::string	vllm = capture_dispatch({"--runtime",
			MEMBRANE_RUNTIME_ID_VLLM}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR
		&& vllm.find("not implemented") != std::string::npos,
		"vllm is still not implemented -- I2 only added ollama");

	unsetenv("MEMBRANE_MODELS_PATH");
	unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
	unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
	unsetenv("MEMBRANE_LAUNCHD_USER_DIR");
	unsetenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);
	std::string	cmd = "rm -rf '" + dir + "'";

	if (system(cmd.c_str()) != 0)
		fprintf(stderr, "warning: cleanup of %s failed\n", dir.c_str());
}

int	main(void)
{
	test_unavailable_snapshot();
	test_zero_loaded_models();
	test_one_model_full_fields();
	test_multiple_models_not_discarded();
	test_resident_array_cap_keeps_true_count();
	test_missing_optional_fields_partial();
	test_ps_failure_degrades_to_partial();
	test_device_wide_vs_model_specific();
	test_not_applicable_fields();
	test_json_deterministic();
	test_parse_ps();
	test_allowlist_includes_ps();

	start_mock();
	test_unavailable_cli_has_no_socket_error();
	test_end_to_end_healthy_zero_models();
	test_end_to_end_one_model();
	test_malformed_ps_non2xx_oversized();
	test_only_allowlisted_routes_hit();
	test_no_state_mutation();
	test_native_unchanged();
	stop_mock();

	unsetenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
	printf("test_observe_ollama: all tests passed\n");
	return (0);
}
