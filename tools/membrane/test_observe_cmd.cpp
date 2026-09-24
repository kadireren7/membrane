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
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "observe_cmd.h"
#include "plan_cmd.h"
#include "plan_v2_resolver.h"
#include "registry_core.h"
#include "server_config.h"
#include "product_cli.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone I1: `membrane observe`'s own tests.
 *
 * Two layers, mirroring observe_cmd.h's two stages:
 *   - membrane_observe_build_snapshot() driven with SYNTHETIC raw inputs
 *     (deterministic, host-independent: no-GPU, no-model, platform-
 *     without-swap, resident models, inconsistent data ...).
 *   - the REAL provider end to end in an isolated env: registry/config/
 *     systemd-dir overridden to a temp dir, the service probe overridden
 *     (MEMBRANE_SERVICE_PROBE_OVERRIDE -- no systemctl call), and the
 *     server endpoint pointed at a closed loopback port or an in-process
 *     mock native server that TRAPS every mutating route. Host numbers
 *     are asserted only as invariants, never exact values.
 *
 * Planner-backed cases use the repo's real GGUF fixtures (metadata only,
 * no tensor load, no generation) and SKIP cleanly when models/ is absent
 * (it is gitignored, e.g. in CI).
 */

/* ---------------------------------------------------------------- */
/* Isolated environment                                              */
/* ---------------------------------------------------------------- */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-observe-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	rmdir_recursive(const std::string &dir)
{
	std::string	cmd = "rm -rf '" + dir + "'";

	if (system(cmd.c_str()) != 0)
		fprintf(stderr, "warning: cleanup of %s may have failed\n",
			dir.c_str());
}

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		dir = make_temp_dir();
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_MODELS_INSTALL_DIR", (dir + "/models").c_str(), 1);
		setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(),
			1);
		setenv("MEMBRANE_SYSTEMD_USER_DIR", (dir + "/systemd").c_str(), 1);
		setenv("MEMBRANE_LAUNCHD_USER_DIR", (dir + "/launchd").c_str(), 1);
		setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_MODELS_INSTALL_DIR");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
		unsetenv("MEMBRANE_LAUNCHD_USER_DIR");
		unsetenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);
		rmdir_recursive(dir);
	}
};

/* A loopback port nothing listens on: bind an ephemeral port with a raw
 * socket (never listen()), read it back, close it -- connecting is then
 * refused immediately. (httplib's bind_to_any_port also listens, which
 * would make every probe wait out the client's read timeout.) */
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

static void	write_config(const s_isolated_env &env, int port,
				const std::string &default_model)
{
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	err;

	cfg.listen_address = "127.0.0.1";
	cfg.port = port;
	cfg.default_model = default_model;
	TEST_ASSERT(membrane_server_config_save(env.dir + "/server.json", cfg,
		&err), "test config written");
}

static void	register_entry(const s_isolated_env &env, const std::string &name,
				const std::string &path, const std::string &basename,
				uint64_t size)
{
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_registry_entry_t	e;

	membrane_registry_load(env.dir + "/models.json", &reg, &err);
	e.name = name;
	e.path = path;
	e.basename = basename;
	e.arch_name = "";
	e.model_max_context = 0;
	e.file_size_bytes = size;
	e.file_mtime_ns = 1;
	e.added_at_unix = 1700000000;
	TEST_ASSERT(membrane_registry_add(&reg, e, &err), "test entry added");
	TEST_ASSERT(membrane_registry_save(env.dir + "/models.json", reg, &err),
		"test registry saved");
}

static bool	file_exists(const std::string &p)
{
	struct stat	st;

	return (stat(p.c_str(), &st) == 0);
}

static std::string	smollm2_fixture(void)
{
	return (std::string(MEMBRANE_TEST_SOURCE_DIR)
		+ "/models/SmolLM2-135M-Instruct-f16.gguf");
}

static std::string	read_file(const std::string &p)
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

static std::set<std::string>	list_dir(const std::string &dir)
{
	std::set<std::string>	out;
	DIR						*d = opendir(dir.c_str());

	if (d == NULL)
		return (out);
	for (struct dirent *e = readdir(d); e != NULL; e = readdir(d))
		out.insert(e->d_name);
	closedir(d);
	return (out);
}

static std::string	capture_dispatch(const std::vector<std::string> &args,
				bool want_json, int *rc)
{
	char	tmpl[] = "/tmp/membrane-observe-test-capture-XXXXXX";
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
	std::string	out = read_file(tmpl);

	unlink(tmpl);
	return (out);
}

/* ---------------------------------------------------------------- */
/* Mock native server (GET /v1/status only; mutations are trapped)   */
/* ---------------------------------------------------------------- */

static httplib::Server			*g_mock;
static std::thread				*g_mock_thread;
static int						g_mock_port;
static std::mutex				g_mock_mutex;
static std::vector<std::string>	g_mock_log;
static std::string				g_status_body;

static void	mock_log_request(const httplib::Request &rq)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);

	g_mock_log.push_back(rq.method + " " + rq.path);
}

static void	start_mock(void)
{
	g_mock = new httplib::Server();
	g_mock->Get("/v1/status", [](const httplib::Request &rq,
			httplib::Response &rs)
		{
			mock_log_request(rq);
			std::lock_guard<std::mutex>	lock(g_mock_mutex);

			rs.set_content(g_status_body, "application/json");
		});
	auto	trap = [](const httplib::Request &rq, httplib::Response &rs)
		{
			mock_log_request(rq);
			rs.status = 500;
		};
	for (const char *p : {"/v1/chat/completions", "/v1/completions",
			"/membrane/v1/models/activate", "/membrane/v1/models/pin",
			"/membrane/v1/models/unpin"})
		g_mock->Post(p, trap);
	g_mock->Get("/membrane/v1/capabilities", trap);
	g_mock->Get("/v1/models", trap);
	g_mock_port = g_mock->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_mock_port > 0, "mock bound");
	g_mock_thread = new std::thread([]() { g_mock->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_mock->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	TEST_ASSERT(g_mock->is_running(), "mock running");
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

static std::vector<std::string>	mock_log(void)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);

	return (g_mock_log);
}

/* ---------------------------------------------------------------- */
/* Synthetic inputs                                                   */
/* ---------------------------------------------------------------- */

static membrane_gpu_device_info_t	device(const char *name, int type,
										uint64_t total, uint64_t free_b)
{
	membrane_gpu_device_info_t	d;

	memset(&d, 0, sizeof(d));
	snprintf(d.name, sizeof(d.name), "%s", name);
	snprintf(d.description, sizeof(d.description), "%s desc", name);
	snprintf(d.backend, sizeof(d.backend), "%s",
		type == MEMBRANE_DEV_TYPE_CPU ? "CPU" : "Vulkan");
	d.type = type;
	d.memory_total = total;
	d.memory_free = free_b;
	return (d);
}

/* A fully-populated native input set -- every probe succeeded. */
static membrane_observe_inputs_t	full_inputs(void)
{
	membrane_observe_inputs_t	in;

	membrane_observe_inputs_init(&in);
	in.timestamp_unix_ms = 1790246096789LL;
	in.collection_duration_ms = 12;
	in.runtime_described = membrane_runtime_describe(
		MEMBRANE_RUNTIME_ID_NATIVE, &in.runtime) != 0;
	in.meminfo_probed = true;
	in.meminfo.ok = true;
	in.meminfo.total_bytes = 6ull << 30;
	in.meminfo.available_bytes = 1ull << 30;
	in.meminfo.swap_total_bytes = 2ull << 30;
	in.meminfo.swap_free_bytes = 1ull << 30;
	in.swap_supported = true;
	in.meminfo_source = "proc_meminfo";
	in.gpu_enumerated = true;
	in.devices.push_back(device("CPU", MEMBRANE_DEV_TYPE_CPU, 0, 0));
	in.devices.push_back(device("Vulkan0", MEMBRANE_DEV_TYPE_GPU,
		4ull << 30, 3ull << 30));
	in.service_probed = true;
	in.service.manager_available = true;
	in.service.installed = true;
	in.service.active = true;
	in.service.manager_name = "systemctl";
	in.config_loaded = true;
	in.config.default_model = "qwen";
	in.server_probed = true;
	in.server_reachable = true;
	in.server_status = json::parse(R"({"running":true,"version":"1.0.0",
		"resident_models":[{"model":"qwen","state":"ready","pinned":false,
		"backend":"Vulkan","gpu_layers":24,"kv_precision":"q8_0",
		"estimated_model_bytes":900000000,"estimated_kv_bytes":120000000}],
		"default_model":"qwen","context_policy":"automatic"})");
	in.registry_loaded = true;
	in.entry_found = true;
	in.entry.name = "qwen";
	in.entry.file_size_bytes = 987654321;
	in.gguf_read = true;
	in.gguf.hparams_available = 1;
	snprintf(in.gguf.arch_name, sizeof(in.gguf.arch_name), "qwen2");
	in.gguf.model_max_context = 32768;
	in.gguf.model_max_context_available = 1;
	in.plan_resolved = true;
	in.plan_has_decisions = true;
	in.plan_context = 4096;
	in.plan_kv_precision = MEMBRANE_JOINT_KV_Q8;
	in.plan_kv_placement = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	in.plan_kv_bytes_known = true;
	in.plan_kv_bytes = 620ull << 20;
	in.plan_quant_known = true;
	in.plan_quant = "Q4_K_M";
	return (in);
}

static membrane_observation_snapshot_t	*build(
				const membrane_observe_inputs_t &in)
{
	static membrane_observation_snapshot_t	s;

	membrane_observe_build_snapshot(MEMBRANE_RUNTIME_ID_NATIVE, in, &s);
	return (&s);
}

static const json	&field(const json &doc, const char *sec, const char *key)
{
	TEST_ASSERT(doc.contains(sec) && doc[sec].contains(key),
		"field present in JSON");
	return (doc[sec][key]);
}

/* ---------------------------------------------------------------- */
/* Pure snapshot tests                                                */
/* ---------------------------------------------------------------- */

/* A + D + Part 7: measured and estimated memory side by side, never
 * merged. */
static void	test_measured_vs_estimated(void)
{
	membrane_observation_snapshot_t	*s = build(full_inputs());

	TEST_ASSERT(s->vram_free_bytes.provenance == MEMBRANE_OBS_PROV_MEASURED,
		"VRAM free is measured");
	TEST_ASSERT(s->kv_estimated_bytes.known
		&& s->kv_estimated_bytes.value == (620ull << 20)
		&& s->kv_estimated_bytes.provenance == MEMBRANE_OBS_PROV_ESTIMATED
		&& strcmp(s->kv_estimated_bytes.source, "planner_v2") == 0,
		"planner KV bytes are estimated/planner_v2");
	TEST_ASSERT(!s->kv_measured_bytes.known
		&& s->kv_measured_bytes.provenance == MEMBRANE_OBS_PROV_UNKNOWN,
		"planner math never becomes measured KV usage");
	TEST_ASSERT(s->context_planned.provenance == MEMBRANE_OBS_PROV_ESTIMATED
		&& s->context_planned.value == 4096, "planned context estimated");
	TEST_ASSERT(!s->context_active.known,
		"planned context is never reported as the active context");
	TEST_ASSERT(s->kv_planned_precision.provenance
		== MEMBRANE_OBS_PROV_ESTIMATED
		&& strcmp(s->kv_planned_precision.value, "q8") == 0,
		"planned KV precision is estimated");
	TEST_ASSERT(s->context_model_max.provenance
		== MEMBRANE_OBS_PROV_STATIC_METADATA
		&& s->context_model_max.value == 32768, "GGUF max ctx is static");
	TEST_ASSERT(s->configured_model.provenance == MEMBRANE_OBS_PROV_CONFIGURED,
		"default model is configured intent");
	TEST_ASSERT(s->model_quant.provenance == MEMBRANE_OBS_PROV_STATIC_METADATA,
		"quant is static metadata");

	/* Runtime-reported residency, with the server's byte figures still
	 * labeled as its own estimates. */
	TEST_ASSERT(s->resident_model_count.known
		&& s->resident_model_count.value == 1
		&& s->resident_model_count.provenance
			== MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "resident count reported");
	TEST_ASSERT(s->resident_model_len == 1, "one resident entry");
	TEST_ASSERT(s->resident_models[0].name.provenance
		== MEMBRANE_OBS_PROV_RUNTIME_REPORTED
		&& strcmp(s->resident_models[0].name.value, "qwen") == 0,
		"resident name runtime_reported");
	TEST_ASSERT(s->resident_models[0].estimated_kv_bytes.provenance
		== MEMBRANE_OBS_PROV_ESTIMATED,
		"server-reported KV bytes stay estimated");
	TEST_ASSERT(s->server_version.provenance
		== MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "version runtime_reported");

	/* The JSON carries both, separately, each self-describing. */
	json	doc = membrane_observe_snapshot_json(*s);

	TEST_ASSERT(field(doc, "gpu", "vram_free_bytes")["provenance"]
		== "measured", "JSON: VRAM free measured");
	TEST_ASSERT(field(doc, "kv", "estimated_bytes")["provenance"]
		== "estimated", "JSON: KV estimated");
	TEST_ASSERT(field(doc, "kv", "measured_bytes")["value"].is_null(),
		"JSON: measured KV null");
}

/* C: measured RAM fields + derived used/headroom. */
static void	test_measured_ram(void)
{
	membrane_observation_snapshot_t	*s = build(full_inputs());

	TEST_ASSERT(s->ram_total_bytes.provenance == MEMBRANE_OBS_PROV_MEASURED
		&& strcmp(s->ram_total_bytes.source, "proc_meminfo") == 0,
		"RAM total measured from proc_meminfo");
	TEST_ASSERT(s->ram_used_bytes.value == (5ull << 30), "used derived");
	TEST_ASSERT(s->ram_headroom_bytes.value == (1ull << 30),
		"RAM headroom = available (raw)");
	TEST_ASSERT(s->swap_total_bytes.known, "swap known where probed");
	TEST_ASSERT(!s->process_rss_bytes.known
		&& strcmp(s->process_rss_bytes.source, "not_instrumented") == 0,
		"server RSS is honestly unknown");

	/* Probe failure: unknown, never 0. */
	membrane_observe_inputs_t	in = full_inputs();

	in.meminfo.ok = false;
	s = build(in);
	TEST_ASSERT(!s->ram_total_bytes.known && !s->ram_available_bytes.known
		&& !s->ram_used_bytes.known && !s->ram_headroom_bytes.known,
		"failed RAM probe -> all RAM facts unknown");
	TEST_ASSERT(strcmp(s->ram_total_bytes.source, "probe_failed") == 0,
		"reason recorded");
}

/* O: a platform whose probe cannot read swap reports it unknown (the
 * Windows/macOS branch of membrane_read_host_meminfo leaves swap 0). */
static void	test_platform_without_swap(void)
{
	membrane_observe_inputs_t	in = full_inputs();

	in.swap_supported = false;
	in.meminfo.swap_total_bytes = 0;
	in.meminfo.swap_free_bytes = 0;
	in.meminfo_source = "GlobalMemoryStatusEx";
	membrane_observation_snapshot_t	*s = build(in);

	TEST_ASSERT(!s->swap_total_bytes.known && !s->swap_free_bytes.known,
		"swap not probed -> unknown, not 0 bytes of swap");
	TEST_ASSERT(strcmp(s->swap_total_bytes.source,
		"not_probed_on_platform") == 0, "reason recorded");
	TEST_ASSERT(s->ram_total_bytes.known
		&& strcmp(s->ram_total_bytes.source, "GlobalMemoryStatusEx") == 0,
		"RAM still measured via the platform's own API");
}

/* I: no GPU. */
static void	test_no_gpu(void)
{
	membrane_observe_inputs_t	in = full_inputs();

	in.devices.clear();
	in.devices.push_back(device("CPU", MEMBRANE_DEV_TYPE_CPU, 1, 1));
	membrane_observation_snapshot_t	*s = build(in);

	TEST_ASSERT(s->gpu_device_count.known && s->gpu_device_count.value == 0
		&& s->gpu_device_count.provenance == MEMBRANE_OBS_PROV_MEASURED,
		"zero GPUs is a measured fact");
	TEST_ASSERT(!s->vram_total_bytes.known && !s->vram_free_bytes.known
		&& !s->vram_used_bytes.known && !s->vram_headroom_bytes.known,
		"no GPU -> VRAM unknown, never 0 bytes");
	TEST_ASSERT(strcmp(s->vram_free_bytes.source, "no_gpu_device") == 0,
		"reason recorded");
	TEST_ASSERT(!s->gpu_backend.known, "no GPU backend");
	TEST_ASSERT(s->status == MEMBRANE_OBS_STATUS_PARTIAL, "still partial");

	/* A GPU whose backend cannot report memory (0/0). */
	in = full_inputs();
	in.devices[1].memory_total = 0;
	in.devices[1].memory_free = 0;
	s = build(in);
	TEST_ASSERT(s->gpu_backend.known && !s->vram_total_bytes.known
		&& strcmp(s->vram_total_bytes.source, "not_reported_by_backend") == 0,
		"0/0 device memory is 'not reported', not a 0-byte GPU");

	/* The first GPU/iGPU is chosen, count covers all of them. */
	in = full_inputs();
	in.devices.push_back(device("Vulkan1", MEMBRANE_DEV_TYPE_IGPU, 8, 8));
	s = build(in);
	TEST_ASSERT(s->gpu_device_count.value == 2
		&& strcmp(s->gpu_device_name.value, "Vulkan0") == 0,
		"first GPU chosen, both counted");
}

/* H: no configured model -> model/planner fields unknown with reasons,
 * still a successful partial observation. */
static void	test_no_active_model(void)
{
	membrane_observe_inputs_t	in = full_inputs();

	in.config.default_model = "";
	in.entry_found = false;
	in.gguf_read = false;
	in.plan_resolved = false;
	in.plan_has_decisions = false;
	in.plan_quant_known = false;
	in.server_status["resident_models"] = json::array();
	membrane_observation_snapshot_t	*s = build(in);

	TEST_ASSERT(!s->configured_model.known
		&& strcmp(s->configured_model.source, "none_configured") == 0,
		"no configured model");
	TEST_ASSERT(!s->context_planned.known && !s->kv_estimated_bytes.known,
		"no planner estimates without a model");
	TEST_ASSERT(s->resident_model_count.known
		&& s->resident_model_count.value == 0
		&& s->resident_model_count.provenance
			== MEMBRANE_OBS_PROV_RUNTIME_REPORTED,
		"runtime reports zero resident models");
	TEST_ASSERT(s->status == MEMBRANE_OBS_STATUS_PARTIAL, "partial");

	/* Configured but unregistered. */
	in = full_inputs();
	in.entry_found = false;
	in.gguf_read = false;
	in.plan_resolved = false;
	in.plan_has_decisions = false;
	in.plan_quant_known = false;
	s = build(in);
	TEST_ASSERT(s->configured_model_registered.known
		&& !s->configured_model_registered.value,
		"configured-but-unregistered is a known 'no'");
	TEST_ASSERT(!s->model_file_size_bytes.known
		&& strcmp(s->model_file_size_bytes.source, "not_registered") == 0,
		"size unknown with reason");
}

/* F: service inactive, server unreachable -> runtime still observable,
 * residency unknown (never "nothing loaded"). */
static void	test_service_inactive(void)
{
	membrane_observe_inputs_t	in = full_inputs();

	in.service.active = false;
	in.server_reachable = false;
	in.server_status = json();
	membrane_observation_snapshot_t	*s = build(in);

	TEST_ASSERT(s->runtime_observable, "runtime still observable");
	TEST_ASSERT(s->service_active.known && !s->service_active.value,
		"inactive is a measured fact");
	TEST_ASSERT(s->server_reachable.known && !s->server_reachable.value,
		"unreachable is a measured fact");
	TEST_ASSERT(!s->resident_model_count.known
		&& strcmp(s->resident_model_count.source, "server_unreachable") == 0,
		"residency unknown when the server cannot be asked");
	TEST_ASSERT(s->configured_model.known,
		"configured intent still reported");
	TEST_ASSERT(s->status == MEMBRANE_OBS_STATUS_PARTIAL, "partial");

	/* Service manager missing -> installed/active unknown. */
	in.service.manager_available = false;
	s = build(in);
	TEST_ASSERT(!s->service_installed.known && !s->service_active.known,
		"no service manager -> service state unknown");
}

/* E + status unavailable. */
static void	test_runtime_identity(void)
{
	membrane_observation_snapshot_t	*s = build(full_inputs());

	TEST_ASSERT(strcmp(s->runtime_id, "membrane-native") == 0, "runtime id");
	TEST_ASSERT(s->runtime_observable, "native observable");
	TEST_ASSERT(s->runtime_availability.known
		&& strcmp(s->runtime_availability.value, "available") == 0
		&& s->runtime_availability.provenance
			== MEMBRANE_OBS_PROV_STATIC_METADATA, "availability static");

	membrane_observe_inputs_t	in = full_inputs();

	in.runtime_described = false;
	s = build(in);
	TEST_ASSERT(s->status == MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"undescribable runtime -> unavailable");
	TEST_ASSERT(!s->ram_total_bytes.known,
		"nothing is reported for an unobservable runtime");
}

/* B + J: unknown is null (not 0), every field has the same four keys,
 * the structure is deterministic. */
static void	test_json_structure(void)
{
	membrane_observe_inputs_t	in = full_inputs();

	in.devices.clear();
	membrane_observation_snapshot_t	*s = build(in);
	json	a = membrane_observe_snapshot_json(*s);
	json	b = membrane_observe_snapshot_json(*build(in));

	TEST_ASSERT(a.dump() == b.dump(), "same inputs -> identical JSON");
	for (const char *k : {"schema_version", "timestamp", "runtime",
			"host_memory", "gpu", "model", "context", "kv", "service",
			"headroom", "sources", "status", "fields", "ok", "mode"})
		TEST_ASSERT(a.contains(k), "top-level key present");
	TEST_ASSERT(a["schema_version"] == MEMBRANE_OBSERVATION_SCHEMA_VERSION,
		"schema version");
	TEST_ASSERT(a["timestamp"]["utc"] == "2026-09-24T10:34:56.789Z",
		"timestamp rendered as RFC 3339 UTC");
	TEST_ASSERT(a["timestamp"]["unix_ms"] == 1790246096789LL, "unix ms");

	size_t	n_fields = 0;
	size_t	n_unknown = 0;

	for (const char *sec : {"runtime", "host_memory", "gpu", "model",
			"context", "kv", "service", "headroom"})
		for (auto &el : a[sec].items())
		{
			if (!el.value().is_object())
				continue ;
			const json	&f = el.value();

			TEST_ASSERT(f.size() == 4 && f.contains("value")
				&& f.contains("known") && f.contains("provenance")
				&& f.contains("source"), "field = {value,known,provenance,"
				"source}");
			n_fields++;
			if (!f["known"].get<bool>())
			{
				n_unknown++;
				TEST_ASSERT(f["value"].is_null(),
					"unknown field value is null, never 0");
				TEST_ASSERT(f["provenance"] == "unknown",
					"unknown field provenance is unknown");
			}
			else
				TEST_ASSERT(f["provenance"] != "unknown",
					"known field has a real provenance");
		}
	TEST_ASSERT(n_fields == a["fields"]["total"].get<size_t>(),
		"every table field is rendered");
	TEST_ASSERT(n_unknown == a["fields"]["unknown"].size(),
		"unknown list matches the rendered fields");
	TEST_ASSERT(a["gpu"]["vram_free_bytes"]["value"].is_null(),
		"no-GPU VRAM is null");
	TEST_ASSERT(a["model"]["resident_models"].is_array(),
		"resident models is an array");
	TEST_ASSERT(a["sources"]["planner_v2"].is_array(),
		"sources index by probe");
}

/* ---------------------------------------------------------------- */
/* Real provider, isolated env                                         */
/* ---------------------------------------------------------------- */

/* C (real) + E + F + G + H + K + M: nothing configured, service not
 * installed, server unreachable -- a real, successful partial snapshot;
 * nothing on disk changes, no unit file appears. */
static void	test_real_empty_env_read_only(void)
{
	s_isolated_env	env;
	int				port = closed_port();

	write_config(env, port, "");
	std::string				cfg_before = read_file(env.dir + "/server.json");
	std::set<std::string>	dir_before = list_dir(env.dir);
	int						rc;
	std::string				out = capture_dispatch({}, true, &rc);
	json					doc = json::parse(out);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "partial observation exits 0");
	TEST_ASSERT(doc["status"] == "partial", "status partial");
	TEST_ASSERT(doc["runtime"]["id"] == "membrane-native", "native runtime");
	TEST_ASSERT(doc["runtime"]["availability"]["value"] == "available",
		"native available");

	const json	&total = field(doc, "host_memory", "total_bytes");
	const json	&avail = field(doc, "host_memory", "available_bytes");

	if (total["known"].get<bool>())
	{
		TEST_ASSERT(total["provenance"] == "measured", "RAM measured");
		TEST_ASSERT(avail["known"].get<bool>(), "available read together");
		TEST_ASSERT(avail["value"].get<uint64_t>()
			<= total["value"].get<uint64_t>(), "available <= total");
		TEST_ASSERT(total["value"].get<uint64_t>() > 0, "total > 0");
	}
#ifdef __linux__
	TEST_ASSERT(total["known"].get<bool>(), "Linux /proc/meminfo is read");
#endif
	TEST_ASSERT(field(doc, "service", "installed")["value"] == false
		&& field(doc, "service", "installed")["provenance"] == "measured",
		"service not installed (probe override)");
	TEST_ASSERT(field(doc, "service", "reachable")["value"] == false,
		"closed port unreachable");
	TEST_ASSERT(field(doc, "model", "resident_count")["known"] == false,
		"residency unknown");
	TEST_ASSERT(field(doc, "model", "configured")["known"] == false,
		"no configured model");
	TEST_ASSERT(field(doc, "kv", "estimated_bytes")["known"] == false,
		"no estimate without a model");
	TEST_ASSERT(field(doc, "gpu", "device_count")["provenance"] == "measured",
		"GPU enumeration measured (0 or more devices)");

	/* Read-only: config byte-identical, no registry/unit/install dir
	 * created. */
	TEST_ASSERT(read_file(env.dir + "/server.json") == cfg_before,
		"config untouched");
	TEST_ASSERT(list_dir(env.dir) == dir_before,
		"no file created in the isolated env (registry, unit, models dir)");
	TEST_ASSERT(!file_exists(env.dir + "/systemd"),
		"no service unit directory created");

	/* Human mode also succeeds and prints no recommendation section. */
	out = capture_dispatch({}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "human mode exits 0");
	TEST_ASSERT(out.find("Observation") != std::string::npos
		&& out.find("[measured]") != std::string::npos, "human output");
	TEST_ASSERT(out.find("ecommend") == std::string::npos,
		"no recommendations in I1");
}

/* L + M + K: a reachable mock native server with a resident model --
 * observe issues exactly one GET /v1/status and never a load/activate/
 * inference route; the registry/config stay byte-identical. */
static void	test_real_mock_server_no_mutation(void)
{
	s_isolated_env	env;

	start_mock();
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_mock_log.clear();
		g_status_body = R"({"running":true,"version":"9.9.9",
			"endpoint":"x","resident_model_limit":1,"resident_models":[
			{"model":"m1","state":"ready","pinned":false,"evictable":true,
			"backend":"CPU","gpu_layers":0,"kv_precision":"f16",
			"estimated_model_bytes":1000,"estimated_kv_bytes":2000}],
			"default_model":"m1","context_policy":"automatic"})";
	}
	write_config(env, g_mock_port, "m1");
	register_entry(env, "m1", env.dir + "/does-not-exist.gguf",
		"does-not-exist.gguf", 1234);
	std::string	cfg_before = read_file(env.dir + "/server.json");
	std::string	reg_before = read_file(env.dir + "/models.json");
	int			rc;
	json		doc = json::parse(capture_dispatch({"--runtime",
			"membrane-native"}, true, &rc));
	std::vector<std::string>	log = mock_log();

	stop_mock();
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "exit 0");
	TEST_ASSERT(log.size() == 1 && log[0] == "GET /v1/status",
		"exactly one read-only GET /v1/status -- no activate/pin/chat");
	TEST_ASSERT(field(doc, "service", "reachable")["value"] == true,
		"mock reachable");
	TEST_ASSERT(field(doc, "service", "server_version")["value"] == "9.9.9"
		&& field(doc, "service", "server_version")["provenance"]
			== "runtime_reported", "version runtime_reported");
	TEST_ASSERT(field(doc, "model", "resident_count")["value"] == 1,
		"one resident");
	const json	&r = doc["model"]["resident_models"][0];

	TEST_ASSERT(r["name"]["value"] == "m1"
		&& r["name"]["provenance"] == "runtime_reported", "resident name");
	TEST_ASSERT(r["estimated_kv_bytes"]["provenance"] == "estimated",
		"server KV bytes stay estimated");
	TEST_ASSERT(field(doc, "model", "configured_registered")["value"] == true,
		"configured model registered");
	TEST_ASSERT(field(doc, "model", "arch")["known"] == false
		&& field(doc, "model", "arch")["source"] == "gguf_unreadable",
		"missing GGUF -> arch unknown with reason");
	TEST_ASSERT(field(doc, "kv", "estimated_bytes")["known"] == false,
		"no planner estimate without a readable GGUF");
	TEST_ASSERT(read_file(env.dir + "/server.json") == cfg_before,
		"config untouched");
	TEST_ASSERT(read_file(env.dir + "/models.json") == reg_before,
		"registry untouched");
}

/* D + N (real GGUF fixture): Planner v2 estimates flow through labeled
 * ESTIMATED, and equal what `membrane plan NAME --json` itself reports
 * for the installed variant -- same planner, no second implementation. */
static void	test_real_planner_estimates(void)
{
	if (!file_exists(smollm2_fixture()))
	{
		printf("SKIP test_real_planner_estimates: %s absent\n",
			smollm2_fixture().c_str());
		return ;
	}
	s_isolated_env	env;
	int				port = closed_port();

	write_config(env, port, "smollm2-135m-instruct");
	register_entry(env, "smollm2-135m-instruct", smollm2_fixture(),
		"SmolLM2-135M-Instruct-F16.gguf", 270885952);
	std::string	reg_before = read_file(env.dir + "/models.json");
	int			rc;
	json		doc = json::parse(capture_dispatch({}, true, &rc));

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "exit 0");
	TEST_ASSERT(field(doc, "model", "arch")["value"] == "llama"
		&& field(doc, "model", "arch")["provenance"] == "static_metadata",
		"arch from GGUF");
	TEST_ASSERT(field(doc, "model", "quant")["value"] == "F16",
		"quant via catalog filename match");
	TEST_ASSERT(field(doc, "context", "model_max")["provenance"]
		== "static_metadata", "max ctx static");

	const json	&planned = field(doc, "context", "planned");
	const json	&kv = field(doc, "kv", "estimated_bytes");

	TEST_ASSERT(planned["known"] == true
		&& planned["provenance"] == "estimated"
		&& planned["source"] == "planner_v2", "planned ctx estimated");
	TEST_ASSERT(kv["known"] == true && kv["provenance"] == "estimated"
		&& kv["value"].get<uint64_t>() > 0, "KV estimate present");
	TEST_ASSERT(field(doc, "kv", "measured_bytes")["known"] == false,
		"measured KV unknown");
	TEST_ASSERT(field(doc, "context", "active")["known"] == false,
		"active ctx unknown");

	/* N: identical to the planner's own resolution for this model. */
	membrane_plan_v2_result_t	*res = new membrane_plan_v2_result_t();
	std::string					code;
	std::string					msg;

	TEST_ASSERT(membrane_plan_resolve_installed_v2("smollm2-135m-instruct",
		res, &code, &msg), "planner resolves the installed model");
	bool	found = false;

	for (size_t i = 0; i < res->candidate_count; ++i)
		if (res->candidates[i].identity.installed)
		{
			found = true;
			TEST_ASSERT(res->candidates[i].decisions.context
				== planned["value"].get<uint64_t>(),
				"observe's planned ctx == Planner v2's decision");
		}
	TEST_ASSERT(found, "installed candidate present");
	delete res;

	/* ... and `membrane plan` itself still reports the same decision. */
	fflush(stdout);
	char	tmpl[] = "/tmp/membrane-observe-test-plan-XXXXXX";
	int		fd = mkstemp(tmpl);
	int		saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	int	prc = membrane_plan_cmd_dispatch({"smollm2-135m-instruct",
			"--quant", "F16"}, true);

	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	json	plan = json::parse(read_file(tmpl));

	unlink(tmpl);
	TEST_ASSERT(prc == MEMBRANE_EXIT_SUCCESS, "membrane plan still works");
	TEST_ASSERT(plan["decisions"]["context"].get<uint64_t>()
		== planned["value"].get<uint64_t>(),
		"`membrane plan --quant F16` agrees with observe's planned ctx");
	TEST_ASSERT(plan["decisions"]["kv_precision"]
		== field(doc, "kv", "planned_precision")["value"],
		"`membrane plan` agrees with observe's planned KV precision");
	TEST_ASSERT(read_file(env.dir + "/models.json") == reg_before,
		"registry untouched");
}

static void	test_cli_errors(void)
{
	s_isolated_env	env;
	int				rc;
	json			doc;

	doc = json::parse(capture_dispatch({"--bogus"}, true, &rc));
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR && doc["ok"] == false,
		"unknown option");
	doc = json::parse(capture_dispatch({"--runtime", "ollama"}, true, &rc));
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR
		&& doc["error"]["message"].get<std::string>().find(
			"not implemented") != std::string::npos,
		"ollama observation is refused in I1 (no /api/ps call)");
	doc = json::parse(capture_dispatch({"--runtime", "nope"}, true, &rc));
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR
		&& doc["error"]["message"].get<std::string>().find(
			"unknown runtime") != std::string::npos, "unknown runtime");
}

int	main(void)
{
	test_measured_vs_estimated();
	test_measured_ram();
	test_platform_without_swap();
	test_no_gpu();
	test_no_active_model();
	test_service_inactive();
	test_runtime_identity();
	test_json_structure();
	test_real_empty_env_read_only();
	test_real_mock_server_no_mutation();
	test_real_planner_estimates();
	test_cli_errors();
	printf("test_observe_cmd: all tests passed\n");
	return (0);
}
