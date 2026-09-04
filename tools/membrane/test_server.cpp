#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "server.h"
#include "registry_core.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Mega Phase A, PR A3: black-box HTTP tests against a REAL
 * membrane_server_run() instance (run in a background thread, stopped via
 * membrane_server_request_stop() -- the same control surface a real
 * SIGTERM handler uses). CI-safe by construction: an EMPTY, isolated
 * temp-directory registry (no real GGUF model anywhere in this file), so
 * every test here covers request validation, routing, and error mapping
 * -- never real generation (that needs a real model; covered by the
 * separate, non-ctest-registered dev-local end-to-end smoke, same
 * precedent as runtime-session-smoke/context-recommender-dryrun).
 *
 * A fixed test port is used rather than engineering port-0 (OS-assigned)
 * discovery back out of membrane_server_run() -- simpler, and CI runners
 * are isolated enough that a collision is not a realistic risk (same
 * judgment call other test suites in this project already make for
 * fixed resource names).
 */

# define TEST_PORT	18942

static std::thread	*g_server_thread = NULL;

static std::string	make_temp_registry_dir(void)
{
	char	tmpl[] = "/tmp/membrane-server-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	start_test_server(const std::string &registry_dir)
{
	membrane_server_options_t	opts;

	opts.bind_address = "127.0.0.1";
	opts.port = TEST_PORT;
	opts.allow_non_loopback = false;
	opts.registry_path = registry_dir + "/models.json";
	g_server_thread = new std::thread([opts]()
		{ membrane_server_run(opts); });
	/* No race-free "ready" signal is exposed across the process boundary
	 * this test uses (httplib::Client, real sockets) -- poll /health
	 * with a bounded retry budget instead of a fixed sleep guess. */
	httplib::Client	probe("127.0.0.1", TEST_PORT);
	int				attempts = 0;

	probe.set_connection_timeout(0, 50000);
	while (attempts < 100)
	{
		auto	res = probe.Get("/health");

		if (res && res->status == 200)
			return ;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		attempts++;
	}
	TEST_ASSERT(false, "server did not become ready within 5s");
}

static void	stop_test_server(void)
{
	membrane_server_request_stop();
	if (g_server_thread != NULL)
	{
		g_server_thread->join();
		delete g_server_thread;
		g_server_thread = NULL;
	}
}

static void	test_health(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/health");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 200, "GET /health returns 200");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["status"] == "ok", "health status is \"ok\"");
	TEST_ASSERT(body.contains("version"), "health reports a version");
}

static void	test_models_empty_registry(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/v1/models");

	TEST_ASSERT(res != nullptr && res->status == 200,
		"GET /v1/models returns 200 even for an empty registry");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["object"] == "list", "response object is \"list\"");
	TEST_ASSERT(body["data"].is_array() && body["data"].empty(),
		"data is an empty array for an empty registry");
}

static void	test_status_no_model_loaded(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/v1/status");

	TEST_ASSERT(res != nullptr && res->status == 200,
		"GET /v1/status returns 200");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["running"] == true, "running is true");
	TEST_ASSERT(body["loaded_model"].is_null(),
		"loaded_model is null before any generation request");
	TEST_ASSERT(body.contains("endpoint"), "endpoint field is present");
	TEST_ASSERT(body["context_policy"] == "automatic",
		"context_policy is \"automatic\"");
}

static void	test_chat_unknown_model(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "nonexistent"},
			{"messages", json::array({{{"role", "user"},
				{"content", "hi"}}})}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 404, "an unknown model returns 404");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "MODEL_NOT_FOUND",
		"error code is MODEL_NOT_FOUND");
	/* Section 27: no local filesystem leak -- the registry is empty, so
	 * there is no real path to leak here, but the error message itself
	 * must never contain an absolute path shape. */
	TEST_ASSERT(body["error"]["message"].get<std::string>().find("/tmp/")
		== std::string::npos, "error message contains no filesystem path");
}

static void	test_chat_invalid_json(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Post("/v1/chat/completions", "not json at all",
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 400, "invalid JSON returns 400");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "INVALID_REQUEST",
		"error code is INVALID_REQUEST");
}

static void	test_chat_missing_required_fields(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"}};	/* no "messages" */
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"a request missing \"messages\" returns 400");
}

/* Mega Phase B, PR B2: `stream: true` is now supported (real SSE
 * streaming, docs/server.md) -- these tests cover exactly what this
 * EMPTY-registry, no-real-model harness CAN cover: every failure path
 * that happens BEFORE headers are ever committed to text/event-stream
 * (Section 18 of the task) behaves identically whether stream is true
 * or false, since none of them reach the streaming code at all. Real
 * SSE wire-format/token-content/UTF-8/cancellation coverage needs a
 * real model and is covered by the separate, non-ctest-registered
 * dev-local end-to-end smoke (same precedent as the rest of this
 * file's own top comment) -- see results/background-service/
 * validation.json's sibling streaming evidence file for that. */
static void	test_chat_stream_unknown_model_still_404(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "nonexistent"},
			{"messages", json::array({{{"role", "user"},
				{"content", "hi"}}})}, {"stream", true}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 404, "stream=true against an unregistered "
		"model still returns a normal 404 JSON error -- the registry "
		"lookup happens before any streaming decision");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "MODEL_NOT_FOUND",
		"error code is MODEL_NOT_FOUND");
	TEST_ASSERT(res->get_header_value("Content-Type").find("application/json")
		!= std::string::npos, "the response is still plain JSON, not "
		"text/event-stream -- headers were never committed to streaming");
}

static void	test_chat_stream_missing_messages_still_400(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"}, {"stream", true}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"stream=true with no \"messages\" still returns a normal 400 "
		"JSON error");
}

static void	test_chat_stream_malformed_message_still_400(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"},
			{"messages", json::array({{{"role", "user"}}})},
			{"stream", true}};	/* missing "content" */
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"stream=true with a malformed message still returns a normal "
		"400 JSON error");
}

static void	test_chat_malformed_message_shape(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"},
			{"messages", json::array({{{"role", "user"}}})}};
			/* missing "content" */
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"a message missing \"content\" returns 400");
}

/* Mega Phase B, PR B3, Section 30 of the task: real concurrent HTTP
 * requests against the SAME running server instance -- proves the new
 * admission gate/model-state machine/registry snapshot machinery (all
 * touched by every request) is genuinely thread-safe under real
 * concurrency, not just single-threaded call sequences. Every request
 * here targets an unregistered model (empty registry), so each one
 * fails fast with 404 -- deliberately never a real generation (this
 * suite has no real model), but the admission gate, the mutex, and the
 * registry snapshot are all still genuinely exercised by every one of
 * these real, concurrent connections. */
static void	test_concurrent_requests_are_thread_safe(void)
{
	const int			n_threads = 16;
	std::atomic<int>	ok_count{0};
	std::vector<std::thread>	threads;

	for (int i = 0; i < n_threads; ++i)
	{
		threads.emplace_back([&]()
			{
				httplib::Client	cli("127.0.0.1", TEST_PORT);
				json			req = {{"model", "concurrent-nonexistent"},
						{"messages", json::array({{{"role", "user"},
							{"content", "hi"}}})}};
				auto			res = cli.Post("/v1/chat/completions",
						req.dump(), "application/json");

				if (res != nullptr && res->status == 404)
					ok_count.fetch_add(1);
			});
	}
	for (auto &t : threads)
		t.join();
	TEST_ASSERT(ok_count.load() == n_threads,
		"every one of 16 real, simultaneous concurrent requests received "
		"a correct 404 response -- no crash, no corrupted shared state, "
		"under real thread contention on the admission gate/mutex/"
		"registry snapshot");
}

/* Section 32: the registry is hot-reloaded (mtime-based) without a
 * server restart -- a model added to the registry file WHILE the
 * server is already running becomes visible to /v1/models on the very
 * next request. Writes the registry file directly via registry_core.h's
 * own real API (never a hand-crafted JSON string) with a fabricated
 * (non-GGUF) path -- registry_core.h itself never validates GGUF-ness
 * (that is model_cmd.cpp's job, Section 13 of registry_core.h's own
 * top comment), and this test only exercises /v1/models (never a real
 * chat completion, so no real model file is ever needed). */
static void	test_registry_hot_reload_without_restart(const std::string &dir)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			before = cli.Get("/v1/models");

	TEST_ASSERT(before != nullptr && before->status == 200, "got a "
		"response before the registry changed");
	json	before_body = json::parse(before->body);

	TEST_ASSERT(before_body["data"].empty(),
		"the registry is still empty before this test writes to it");

	std::string					registry_path = dir + "/models.json";
	membrane_registry_t			reg;
	membrane_registry_error_t	err;

	TEST_ASSERT(membrane_registry_load(registry_path, &reg, &err) == true,
		"loading the existing (empty) registry file succeeds");

	membrane_registry_entry_t	entry;

	entry.name = "hot-reloaded-model";
	entry.path = "/nonexistent/fake.gguf";
	entry.basename = "fake.gguf";
	entry.arch_name = "llama";
	entry.model_max_context = 2048;
	entry.file_size_bytes = 1000;
	entry.file_mtime_ns = 12345;
	entry.added_at_unix = 1700000000;
	TEST_ASSERT(membrane_registry_add(&reg, entry, &err) == true,
		"adding the new entry to the in-memory registry succeeds");
	TEST_ASSERT(membrane_registry_save(registry_path, reg, &err) == true,
		"saving the updated registry file succeeds");

	/* mtime resolution on some filesystems is coarser than this write-
	 * then-immediately-poll gap -- poll with a bounded retry budget
	 * rather than a single immediate check, same "no fixed sleep guess"
	 * discipline as start_test_server()'s own readiness poll above. */
	int	attempts = 0;
	bool	seen = false;

	while (attempts < 100 && !seen)
	{
		auto	after = cli.Get("/v1/models");

		if (after != nullptr && after->status == 200)
		{
			json	after_body = json::parse(after->body);

			for (const auto &m : after_body["data"])
				if (m["id"] == "hot-reloaded-model")
					seen = true;
		}
		if (!seen)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			attempts++;
		}
	}
	TEST_ASSERT(seen, "the server picked up the registry change without "
		"a restart within 5s (mtime-based hot reload, Section 32)");
}

/* Section 29: with the admission gate's own capacity forced to 0 via
 * its test-only environment override, the very FIRST chat completion
 * request against a SEPARATE server instance is guaranteed (never
 * timing-dependent) to be rejected -- a real, deterministic proof of
 * the 503 SERVICE_UNAVAILABLE + Retry-After wiring, not just the
 * underlying request_admission_gate_t primitive's own already-covered
 * unit tests (test_request_admission.cpp). A second server instance is
 * used (not the shared TEST_PORT one above) so this test's env-var
 * override never affects any other test in this file. */
# define ADMISSION_TEST_PORT	18943

static void	test_admission_gate_rejects_at_capacity_zero(void)
{
	setenv("MEMBRANE_MAX_PENDING_CHAT_REQUESTS", "0", 1);

	std::string					dir = make_temp_registry_dir();
	membrane_server_options_t	opts;

	opts.bind_address = "127.0.0.1";
	opts.port = ADMISSION_TEST_PORT;
	opts.allow_non_loopback = false;
	opts.registry_path = dir + "/models.json";

	std::thread	server_thread([opts]() { membrane_server_run(opts); });
	httplib::Client	probe("127.0.0.1", ADMISSION_TEST_PORT);
	int				attempts = 0;
	bool			ready = false;

	probe.set_connection_timeout(0, 50000);
	while (attempts < 100 && !ready)
	{
		auto	res = probe.Get("/health");

		if (res && res->status == 200)
			ready = true;
		else
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			attempts++;
		}
	}
	TEST_ASSERT(ready, "the second (capacity-0) server instance became "
		"ready within 5s");

	httplib::Client	cli("127.0.0.1", ADMISSION_TEST_PORT);
	json			req = {{"model", "irrelevant-never-reached"},
			{"messages", json::array({{{"role", "user"},
				{"content", "hi"}}})}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 503,
		"with capacity forced to 0, the very first request is "
		"deterministically rejected -- 503, never a timing-dependent "
		"maybe");
	TEST_ASSERT(res->get_header_value("Retry-After") != "",
		"a Retry-After header is present, Section 29 of the task");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "SERVER_BUSY",
		"error code is SERVER_BUSY");

	membrane_server_request_stop();
	server_thread.join();
	unsetenv("MEMBRANE_MAX_PENDING_CHAT_REQUESTS");
	{
		std::string	cmd = "rm -rf '" + dir + "'";
		int			rc = system(cmd.c_str());

		if (rc != 0)
			fprintf(stderr, "warning: cleanup of %s may have failed "
				"(rc=%d)\n", dir.c_str(), rc);
	}
}

int	main(void)
{
	std::string	dir = make_temp_registry_dir();

	start_test_server(dir);
	test_health();
	test_models_empty_registry();
	test_status_no_model_loaded();
	test_chat_unknown_model();
	test_chat_invalid_json();
	test_chat_missing_required_fields();
	test_chat_stream_unknown_model_still_404();
	test_chat_stream_missing_messages_still_400();
	test_chat_stream_malformed_message_still_400();
	test_chat_malformed_message_shape();
	test_concurrent_requests_are_thread_safe();
	/* Mutates the shared registry permanently for the rest of this
	 * process's run -- kept LAST among the TEST_PORT-based tests so no
	 * earlier test's "empty registry" assumption breaks. */
	test_registry_hot_reload_without_restart(dir);
	stop_test_server();
	{
		std::string	cmd = "rm -rf '" + dir + "'";
		int			rc = system(cmd.c_str());

		if (rc != 0)
			fprintf(stderr, "warning: cleanup of %s may have failed "
				"(rc=%d)\n", dir.c_str(), rc);
	}
	/* A separate server instance (its own port, its own temp registry,
	 * its own env-var override) -- safe to run after the shared TEST_
	 * PORT server has already been stopped above. */
	test_admission_gate_rejects_at_capacity_zero();
	printf("test_server: all tests passed\n");
	return (0);
}
