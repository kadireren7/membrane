#include "server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "llama.h"
#include "runtime_session.h"
#include "registry_core.h"
#include "gpu_policy.h"
#include "product_cli.h"
#include "utf8_stream.h"
#include "stream_queue.h"
#include "request_admission.h"

#include <sys/stat.h>

using json = nlohmann::json;

/*
 * Mega Phase B, PR B3, Section 27 of the task: an explicit model-lifecycle
 * state machine -- replaces the previous implicit "model_loaded bool +
 * loaded_name string" boolean soup with a named, observable state,
 * synchronized the same way every other piece of st's mutable state
 * already is (transitions only ever happen while st->mtx is held).
 * EMPTY: no model has ever been loaded, or the last one was cleanly
 * unloaded -- a healthy, normal state, never an error.
 * LOADING / UNLOADING: a load/unload is in progress (both only ever
 * observed transiently, since they happen while st->mtx is held for the
 * whole request -- exposed anyway for /v1/status's own honesty and for
 * a future phase that might narrow the lock).
 * READY: a model is loaded and idle.
 * GENERATING: a model is loaded and actively generating (streaming or
 * not).
 * ERROR: a model switch failed AND the attempt to restore the
 * previously-loaded model (see ensure_model_loaded()'s own recovery
 * logic) also failed -- distinct from EMPTY specifically so /v1/status
 * can tell "never loaded anything" apart from "something went wrong".
 */
enum e_membrane_model_state
{
	MEMBRANE_MODEL_STATE_EMPTY = 0,
	MEMBRANE_MODEL_STATE_LOADING,
	MEMBRANE_MODEL_STATE_READY,
	MEMBRANE_MODEL_STATE_GENERATING,
	MEMBRANE_MODEL_STATE_UNLOADING,
	MEMBRANE_MODEL_STATE_ERROR,
};

static const char	*membrane_model_state_name(int state)
{
	switch (state)
	{
		case MEMBRANE_MODEL_STATE_EMPTY: return ("empty");
		case MEMBRANE_MODEL_STATE_LOADING: return ("loading");
		case MEMBRANE_MODEL_STATE_READY: return ("ready");
		case MEMBRANE_MODEL_STATE_GENERATING: return ("generating");
		case MEMBRANE_MODEL_STATE_UNLOADING: return ("unloading");
		case MEMBRANE_MODEL_STATE_ERROR: return ("error");
		default: return ("unknown");
	}
}

/* Section 29: a bounded pending-request admission gate for POST /v1/chat/
 * completions -- see request_admission.h's own top comment. 8
 * concurrently-admitted chat requests is generous headroom above the
 * "1 active generation, the rest waiting on st->mtx" reality this
 * server already has (no continuous batching), while still bounding how
 * many requests can pile up before a caller gets an honest, immediate
 * 503 instead of an ever-growing wait. This is deliberately NOT exposed
 * as a documented/supported server_config.h setting (Section 8's own
 * "keep minimal" convention) -- MEMBRANE_MAX_PENDING_CHAT_REQUESTS only
 * exists so test_server.cpp can deterministically force admission
 * rejection (capacity 0 -- the very first request always gets a real
 * 503) without needing a real model or a timing-dependent flood of
 * concurrent requests. */
# define MEMBRANE_DEFAULT_MAX_PENDING_CHAT_REQUESTS	8

static int	membrane_max_pending_chat_requests(void)
{
	const char	*env = getenv("MEMBRANE_MAX_PENDING_CHAT_REQUESTS");

	if (env != NULL && env[0] != '\0')
	{
		int	parsed = atoi(env);

		if (parsed >= 0)
			return (parsed);
	}
	return (MEMBRANE_DEFAULT_MAX_PENDING_CHAT_REQUESTS);
}

/*
 * See server.h's own top comment for the full contract. This file never
 * calls any of the CLI's own print_*()/argv-parsing code (Section 1/5 of
 * the Mega Phase A task's own "which CLI parsing code must NOT leak into
 * server core" question) -- it builds its own membrane_run_opts_t/
 * membrane_generation_request_t values directly and talks JSON, never
 * text, to its callers.
 */

static std::atomic<bool>	g_stop_requested{false};

void	membrane_server_request_stop(void)
{
	g_stop_requested.store(true);
}

static void	handle_signal(int sig)
{
	(void)sig;
	g_stop_requested.store(true);
}

/* Section 27: HTTP-facing error text is built HERE, deliberately never
 * forwarding membrane_run_error_t::human/message verbatim -- those are
 * written for a terminal user who already owns the machine and may
 * legitimately include a real absolute filesystem path (Section 27: "No
 * local filesystem leak" over HTTP). */
static void	send_json_error(httplib::Response &res, int status,
				const std::string &code, const std::string &message)
{
	json	j;

	j["error"] = {{"code", code}, {"message", message}};
	res.status = status;
	res.set_content(j.dump(), "application/json");
}

struct s_membrane_server_model_state
{
	std::mutex					mtx;	/* Section 24: serializes every
										 * generation request -- no
										 * continuous batching, correctness
										 * first */
	membrane_runtime_t			rt = {};
	bool						model_loaded = false;
	std::string					loaded_name;
	membrane_model_session_t	session;
	std::string					cached_chat_template;	/* valid iff
										 * model_loaded is true and refers
										 * to `loaded_name` -- avoids a
										 * redundant vocab-only model load
										 * (real, measured cost: 4 GGUF
										 * metadata loads for 2 requests
										 * to the same already-loaded
										 * model before this cache existed)
										 * on every request past the first
										 * to a given model */
	std::string					default_model;	/* Section 9 -- "" = none
										 * configured; a chat request
										 * omitting "model" falls back to
										 * this, never proactively loaded */

	/* Mega Phase B, PR B3, Section 27: only ever mutated while mtx is
	 * held, but readable atomically (e.g. by /v1/status, itself also
	 * under mtx today -- see handle_status()) without a data race
	 * either way. */
	std::atomic<int>			model_state{MEMBRANE_MODEL_STATE_EMPTY};

	/* Section 29: bounded admission for POST /v1/chat/completions --
	 * deliberately its OWN synchronization primitive, checked BEFORE
	 * mtx is ever touched, so a caller past capacity gets an immediate
	 * 503 instead of joining an ever-growing queue of threads blocked
	 * on mtx. */
	request_admission_gate_t	admission_gate{membrane_max_pending_chat_requests()};

	/* Mega Phase B, PR B3, Section 32: the model registry now lives
	 * HERE (moved out of membrane_server_run()'s own local variable)
	 * so it can be hot-refreshed from an mtime check without any
	 * caller needing to hold st->mtx for the whole operation --
	 * registry_mtx is a separate, short-lived lock (registry lookups
	 * are cheap and must never wait behind a long-running generation
	 * just to list/resolve model names). */
	std::mutex					registry_mtx;
	membrane_registry_t			registry;
	std::string					registry_path;
	int64_t						registry_mtime_ns = 0;
};

static bool	load_chat_template(const std::string &model_path,
				std::string *out_template, std::string *err_message)
{
	llama_model_params	mp = llama_model_default_params();

	mp.vocab_only = true;
	llama_model	*m = llama_model_load_from_file(model_path.c_str(), mp);

	if (m == NULL)
	{
		*err_message = "could not read model metadata";
		return (false);
	}
	const char	*tmpl = llama_model_chat_template(m, NULL);

	if (tmpl == NULL || tmpl[0] == '\0')
	{
		llama_model_free(m);
		*err_message = "this model has no usable chat template -- the "
			"OpenAI-compatible chat endpoint requires one (Section 20 of "
			"the Mega Phase A task: never a naive manual prompt join)";
		return (false);
	}
	*out_template = tmpl;
	llama_model_free(m);
	return (true);
}

static bool	apply_chat_template(const std::string &tmpl,
				const std::vector<llama_chat_message> &chat,
				std::string *out_prompt)
{
	std::vector<char>	buf(4096);
	int32_t				needed = llama_chat_apply_template(tmpl.c_str(),
			chat.data(), chat.size(), true, buf.data(), (int32_t)buf.size());

	if (needed < 0)
		return (false);
	if ((size_t)needed > buf.size())
	{
		buf.resize((size_t)needed);
		needed = llama_chat_apply_template(tmpl.c_str(), chat.data(),
				chat.size(), true, buf.data(), (int32_t)buf.size());
		if (needed < 0)
			return (false);
	}
	*out_prompt = std::string(buf.data(), (size_t)needed);
	return (true);
}

/* Section 21/22: the SAME context-recommendation + host-memory-guard +
 * joint-planner pipeline --ctx auto uses (via membrane_resolve_ctx_auto(),
 * runtime_session.h, PR A1) -- never a second/simplified planner. Builds
 * the fully-automatic membrane_run_opts_t a bare `membrane-run --auto`
 * (no other flags) would produce, so the resolved plan is provably the
 * exact same one the CLI's own --auto would pick for this model/prompt.
 */
static void	fill_auto_opts(membrane_run_opts_t *o, const char *model_path,
				int gen_tokens)
{
	*o = membrane_run_opts_t();
	o->model_path = model_path;
	o->ctx_mode = MEMBRANE_RUN_CTX_AUTO;
	o->kv_mode = MEMBRANE_KV_STORE_ADAPTIVE;
	o->gpu_layers = MEMBRANE_GPU_LAYERS_AUTO;
	o->kv_placement = MEMBRANE_KV_PLACEMENT_AUTO;
	o->auto_mode = 1;
	o->gen_tokens = gen_tokens;
	o->quiet = 1;
}

static int	membrane_kv_precision_name_to_json(int mode,
				const char **out_name)
{
	if (mode == MEMBRANE_KV_STORE_Q8)
		*out_name = "q8";
	else if (mode == MEMBRANE_KV_STORE_Q5)
		*out_name = "q5";
	else
		*out_name = "native";
	return (1);
}

/* Attempts to load exactly ONE candidate (`entry`) as the server's active
 * model -- no unload/recovery logic of its own, just "try this one, real
 * host-memory snapshot taken fresh right now" (Section 33 of the task:
 * memory revalidation happens on EVERY call here, never a cached
 * startup-time snapshot -- membrane_read_host_meminfo() is called fresh
 * each time, so a switch attempted after the host's free memory changed
 * -- e.g. another process using more RAM, or this same server having
 * just freed the previous model's memory -- always sees that current
 * reality, not stale data). Factored out of ensure_model_loaded() so the
 * same real load path can be reused both for the actual requested switch
 * and for automatically restoring a previous model after a failed one
 * (see ensure_model_loaded()'s own recovery logic below). */
static bool	try_load_one(s_membrane_server_model_state *st,
				const membrane_registry_entry_t &entry,
				const std::string &first_prompt, int gen_tokens,
				std::string *err_code, std::string *err_message,
				int *http_status)
{
	membrane_run_opts_t	o;

	fill_auto_opts(&o, entry.path.c_str(), gen_tokens);

	membrane_host_meminfo_t		host;
	membrane_ctxauto_outcome_t	ctxauto;

	membrane_read_host_meminfo(&host);
	if (!membrane_resolve_ctx_auto(o, first_prompt, host, &ctxauto)
		|| !ctxauto.rec.ok)
	{
		*err_code = "NO_FEASIBLE_CONTEXT";
		*err_message = "no context/hardware plan could be resolved for "
			"this model on this host";
		*http_status = 503;
		return (false);
	}
	o.ctx = (uint32_t)ctxauto.rec.recommended_context;

	membrane_model_session_t	session;
	membrane_run_error_t		open_err;

	if (!membrane_model_open(&st->rt, entry.path.c_str(), o, o.ctx, &session,
			&open_err))
	{
		*err_code = "MODEL_LOAD_FAILED";
		*err_message = "the model could not be loaded";
		*http_status = 500;
		return (false);
	}
	st->session = session;
	st->model_loaded = true;
	st->loaded_name = entry.name;
	return (true);
}

/* Loads `entry` as the server's one active model, unloading whatever was
 * loaded before (Section 23: one active model). On the FIRST load of
 * this model, runs the real context-recommendation pipeline against
 * `first_prompt` to decide gpu_layers/KV precision/ctx (Section 21) --
 * this is a REAL, disclosed limitation (documented in docs/server.md):
 * that initial plan is sized for whichever request happens to trigger
 * the load, not a hypothetical future largest prompt; gpu_layers/KV
 * precision cannot change again without a reload.
 *
 * Mega Phase B, PR B3, Section 31: model-switch FAILURE recovery -- a
 * naive "unload A, then try to load B" leaves the server with NO model
 * loaded at all if B fails, even though A was working fine a moment ago
 * (a real regression this project would otherwise be silently
 * introducing relative to "don't leave corrupted state"). This function
 * instead remembers A's own name, and if B's load fails, automatically
 * attempts to RELOAD A (a fresh try_load_one() call, re-validating
 * memory again) before giving up -- the caller's own request for B is
 * still correctly reported as a failure (err_code/err_message/
 * http_status describe B's own failure, never silently swapped for a
 * misleading "success"), but the SERVER itself ends up back in a known-
 * good state (READY on A) rather than EMPTY, whenever that recovery
 * itself succeeds. Only if BOTH B's load and A's own restore attempt
 * fail does the server end up in the explicit ERROR state -- distinct
 * from EMPTY specifically so /v1/status can tell "never loaded anything"
 * apart from "a switch attempt left this host unable to serve anything
 * right now". */
static bool	ensure_model_loaded(s_membrane_server_model_state *st,
				const membrane_registry_t &registry_snapshot,
				const membrane_registry_entry_t &entry,
				const std::string &first_prompt, int gen_tokens,
				std::string *err_code, std::string *err_message,
				int *http_status)
{
	if (st->model_loaded && st->loaded_name == entry.name)
		return (true);

	std::string	previous_name = st->loaded_name;
	bool		had_previous = st->model_loaded;

	if (st->model_loaded)
	{
		st->model_state.store(MEMBRANE_MODEL_STATE_UNLOADING,
			std::memory_order_relaxed);
		membrane_model_close(&st->session);
		st->model_loaded = false;
		st->loaded_name.clear();
		st->cached_chat_template.clear();
	}
	st->model_state.store(MEMBRANE_MODEL_STATE_LOADING,
		std::memory_order_relaxed);
	if (try_load_one(st, entry, first_prompt, gen_tokens, err_code,
			err_message, http_status))
	{
		st->model_state.store(MEMBRANE_MODEL_STATE_READY,
			std::memory_order_relaxed);
		return (true);
	}
	if (had_previous)
	{
		const membrane_registry_entry_t	*prev_entry
				= membrane_registry_find(registry_snapshot, previous_name);

		if (prev_entry != NULL)
		{
			std::string	restore_err_code;
			std::string	restore_err_message;
			int			restore_http_status = 500;

			st->model_state.store(MEMBRANE_MODEL_STATE_LOADING,
				std::memory_order_relaxed);
			if (try_load_one(st, *prev_entry, first_prompt, gen_tokens,
					&restore_err_code, &restore_err_message,
					&restore_http_status))
			{
				st->model_state.store(MEMBRANE_MODEL_STATE_READY,
					std::memory_order_relaxed);
				/* The ORIGINAL request (for `entry`) is still a failure
				 * -- err_code/err_message/http_status already describe
				 * it and are left untouched -- only the server's own
				 * resting state improved. */
				return (false);
			}
		}
	}
	st->model_state.store(MEMBRANE_MODEL_STATE_ERROR,
		std::memory_order_relaxed);
	return (false);
}

static void	handle_health(const httplib::Request &, httplib::Response &res)
{
	json	j;

	j["status"] = "ok";
	j["version"] = MEMBRANE_VERSION;
	res.set_content(j.dump(), "application/json");
}

/* Mega Phase B, PR B3, Section 32 of the task: "the server should see
 * registry updates without restart if cheap" -- an mtime check (a single
 * stat(), not re-reading/re-parsing the file every request) followed by
 * a real reload only when the file actually changed. Deliberately no
 * file-watcher/inotify complexity (the task's own "no file-watcher
 * complexity required" allowance) -- a plain poll-on-use is simpler and
 * just as correct here, since registry changes are rare (an operator
 * running `membrane model add`), not a hot path. A reload that fails to
 * parse (e.g. caught mid-write) is silently ignored -- the OLD, still-
 * good registry_snapshot is kept rather than corrupting server state
 * over a transient race with another process's own atomic rename() (the
 * same atomicity registry_core.h's own save() already provides, but a
 * reader can still observe the pre-rename mtime one poll too early in
 * principle). registry_mtx is a separate, short-lived lock -- a registry
 * lookup never has to wait behind a long-running generation. */
static membrane_registry_t	refresh_and_snapshot_registry(
				s_membrane_server_model_state *st)
{
	struct stat	stat_buf;

	if (stat(st->registry_path.c_str(), &stat_buf) == 0)
	{
		int64_t	mtime_ns = (int64_t)stat_buf.st_mtim.tv_sec * 1000000000LL
				+ (int64_t)stat_buf.st_mtim.tv_nsec;

		std::lock_guard<std::mutex>	lock(st->registry_mtx);

		if (mtime_ns != st->registry_mtime_ns)
		{
			membrane_registry_t		reloaded;
			membrane_registry_error_t	err;

			if (membrane_registry_load(st->registry_path, &reloaded, &err))
			{
				st->registry = reloaded;
				st->registry_mtime_ns = mtime_ns;
			}
		}
		return (st->registry);
	}
	std::lock_guard<std::mutex>	lock(st->registry_mtx);

	return (st->registry);	/* stat() failure (e.g. file momentarily
							 * missing) -- keep whatever is already
							 * cached, matching registry_core.h's own
							 * "missing is not an error" convention */
}

static void	handle_models(s_membrane_server_model_state *st,
				const httplib::Request &, httplib::Response &res)
{
	membrane_registry_t	reg = refresh_and_snapshot_registry(st);
	json	arr = json::array();

	for (const auto &e : reg.entries)
	{
		json	m;

		m["id"] = e.name;
		m["object"] = "model";
		m["owned_by"] = "membrane";
		arr.push_back(m);
	}
	json	root;

	root["object"] = "list";
	root["data"] = arr;
	res.set_content(root.dump(), "application/json");
}

/* Section 37 of the Mega Phase A task: a membrane-specific (not OpenAI)
 * status surface for the new `membrane status` CLI command below --
 * never claims a daemon/process-management capability this project does
 * not have (`serve` stays foreground-only); this is a thin, honest
 * "what does the currently-loaded session look like" read. */
static void	handle_status(s_membrane_server_model_state *st,
				const std::string &bind, int port, const httplib::Request &,
				httplib::Response &res)
{
	json	j;

	j["running"] = true;
	j["version"] = MEMBRANE_VERSION;
	j["endpoint"] = "http://" + bind + ":" + std::to_string(port);
	std::lock_guard<std::mutex>	lock(st->mtx);

	j["model_state"] = membrane_model_state_name(
		st->model_state.load(std::memory_order_relaxed));
	if (st->model_loaded)
	{
		const char	*kv_name;

		membrane_kv_precision_name_to_json(
			st->session.gs.adaptive_used ? st->session.gs.adaptive_selected_mode
				: MEMBRANE_KV_STORE_NATIVE, &kv_name);
		j["loaded_model"] = st->loaded_name;
		j["backend"] = st->session.gs.requested
			? st->session.gs.backend_selected : "CPU";
		j["gpu_layers"] = st->session.gs.gpu_layers_selected;
		j["kv_precision"] = kv_name;
	}
	else
		j["loaded_model"] = nullptr;
	j["context_policy"] = "automatic";
	res.set_content(j.dump(), "application/json");
}

/*
 * Mega Phase B, PR B2: `stream: true` support.
 *
 * Architecture (Section 16-18 of the task): membrane_session_generate()'s
 * token callback is push-based (called synchronously, per-token, from
 * inside the decode loop); cpp-httplib's chunked content provider is
 * pull-based (httplib calls back to ask for the next chunk). Bridged
 * here with a dedicated generation WORKER thread (runs membrane_session_
 * generate() with a token_cb that pushes onto a bounded queue) plus this
 * queue itself, which the HTTP connection thread's own content-provider
 * callback pops from. Never the whole completion accumulated before
 * streaming starts -- the first queued token can reach the client before
 * generation finishes.
 *
 * Backpressure (Section 24): the queue is BOUNDED (MEMBRANE_STREAM_QUEUE_
 * CAPACITY). A slow client blocks the worker thread's own push, never
 * grows memory unboundedly -- generation pauses, it does not buffer
 * ahead of what the client has actually consumed.
 *
 * Cancellation (Section 20): the content-provider callback is the ONLY
 * place able to detect a dead client (via DataSink::is_writable(), which
 * cpp-httplib itself backs with a real socket liveness check) -- on
 * detecting one, it sets a shared std::atomic<bool> cancel_flag, which
 * (a) wakes the worker thread's own blocked queue push immediately
 * (never waits for the next token) and (b) is the SAME flag threaded all
 * the way down into run_generation()'s own per-step check
 * (decode_loop.h) -- the runtime core itself only ever sees "the caller
 * asked to stop," never any HTTP/socket-specific concept.
 *
 * st->mtx (Section 24's existing full-request-serialization mutex) is
 * held for the ENTIRE request, streaming tail included -- not just the
 * synchronous prefix -- via a std::unique_lock moved into the stream
 * state and released only in the resource_releaser once the whole
 * chunked response (success or failure) is complete. This preserves the
 * exact same "one generation at a time" policy streaming already had to
 * honor for non-streaming requests; a second concurrent request genuinely
 * waits for a still-streaming one to finish, exactly as it already waits
 * for a non-streaming one today.
 */

struct s_stream_worker_ctx
{
	stream_queue_t				*queue;
	const std::atomic<bool>	*cancel_flag;
	std::string					utf8_pending;	/* worker-thread-only, no
											 * synchronization needed */
};

/* The one bridge from the push-based decode loop (this runs on the
 * WORKER thread, synchronously inside run_generation()'s own loop) into
 * the pull-based HTTP queue above. */
static void	stream_token_cb(const char *piece, size_t piece_len, int step,
				void *user_data)
{
	(void)step;
	s_stream_worker_ctx	*wctx = (s_stream_worker_ctx *)user_data;

	wctx->utf8_pending.append(piece, piece_len);
	size_t	incomplete = membrane_utf8_incomplete_suffix_len(wctx->utf8_pending);
	size_t	emit_len = wctx->utf8_pending.size() - incomplete;

	if (emit_len == 0)
		return ;
	s_stream_event	ev;

	ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
	ev.text = wctx->utf8_pending.substr(0, emit_len);
	wctx->utf8_pending.erase(0, emit_len);
	wctx->queue->push_blocking(std::move(ev));
}

struct s_stream_request_state
{
	stream_queue_t				queue;
	std::atomic<bool>			cancel_flag{false};
	std::thread					worker;
	std::unique_lock<std::mutex>	server_lock;	/* holds st->mtx for the
											 * whole request -- see this
											 * section's own top comment */

	/* Owned here so it outlives handle_chat_completions()'s own return
	 * (the worker thread and the content-provider/resource-releaser
	 * callbacks below all run AFTER that function has already
	 * returned). model_path is a real std::string, NOT just req_o's own
	 * raw model_path pointer copied verbatim -- Mega Phase B, PR B3
	 * turned the registry entry handle_chat_completions resolves
	 * `entry` from into a per-request SNAPSHOT COPY (registry hot-
	 * reload, Section 32), so `entry->path.c_str()` itself is only
	 * valid for handle_chat_completions()'s own stack frame; req_o.
	 * model_path must point into storage that outlives it instead (set
	 * from THIS string's own .c_str(), after this string is populated --
	 * see the call site in handle_chat_completions()). */
	membrane_run_opts_t			req_o;
	std::string					model_path;
	std::string					prompt_text;
	std::string					model_name;
	int							max_tokens = 0;
	bool						include_usage = false;
	s_membrane_server_model_state	*st = NULL;
	std::string					id;
	request_admission_ticket_t	admission_ticket;

	s_stream_request_state() : queue(&cancel_flag) {}
};

/* Runs entirely on the dedicated worker thread this request's own
 * s_stream_request_state::worker owns -- never touches httplib/DataSink
 * directly (Section 20: "runtime core only understands 'caller requested
 * cancellation'... no server-specific socket logic"), only the queue and
 * the shared cancel_flag. */
static void	stream_worker_fn(std::shared_ptr<s_stream_request_state> state)
{
	s_stream_worker_ctx	wctx;

	wctx.queue = &state->queue;
	wctx.cancel_flag = &state->cancel_flag;

	membrane_generation_request_t	gen_req = {};
	membrane_generation_result_t	gen_res;

	gen_req.o = &state->req_o;
	gen_req.prompt_text = state->prompt_text;
	gen_req.ctx_size = 0;
	gen_req.token_cb = stream_token_cb;
	gen_req.token_cb_ud = &wctx;
	gen_req.cancel_flag = &state->cancel_flag;
	membrane_session_generate(&state->st->session, gen_req, &gen_res);
	/* Flush whatever partial UTF-8 tail never resolved -- better to emit
	 * it than to silently drop real generated bytes (Section 21: only
	 * reachable if generation stopped (limit/EOG/cancellation) exactly
	 * mid-character, a rare edge case, never treated as an excuse to
	 * lose output). */
	if (!wctx.utf8_pending.empty())
	{
		s_stream_event	flush_ev;

		flush_ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
		flush_ev.text = wctx.utf8_pending;
		state->queue.push_blocking(std::move(flush_ev));
	}
	s_stream_event	terminal;

	if (gen_res.cancelled)
		terminal.type = MEMBRANE_STREAM_EVENT_CANCELLED;
	else if (!gen_res.ok)
	{
		terminal.type = MEMBRANE_STREAM_EVENT_ERROR;
		terminal.error_code = (gen_res.err.set
				&& gen_res.err.reason_code[0] != '\0')
			? gen_res.err.reason_code : "GENERATION_FAILED";
		terminal.error_message = "generation failed for this request";
	}
	else
	{
		terminal.type = MEMBRANE_STREAM_EVENT_DONE;
		terminal.finish_reason = ((size_t)gen_res.gen_result.tokens.size()
				>= (size_t)state->max_tokens) ? "length" : "stop";
		terminal.prompt_tokens = gen_res.prompt_tokens.size();
		terminal.completion_tokens = gen_res.gen_result.tokens.size();
		terminal.include_usage = state->include_usage;
	}
	/* Best-effort: if the queue is full AND cancel_flag is already true
	 * (the only way push_blocking() returns false), there is no reader
	 * left to care about this terminal event anyway -- dropping it here
	 * is correct, not a leak (the content-provider side never blocks on
	 * "a terminal event must eventually arrive" once it has itself
	 * already detected the disconnect and returned). */
	state->queue.push_blocking(std::move(terminal));
}

static std::string	sse_frame(const json &payload)
{
	return ("data: " + payload.dump() + "\n\n");
}

static json	stream_chunk_json(const std::string &id,
				const std::string &model_name, const json &delta,
				const char *finish_reason)
{
	json	chunk;

	chunk["id"] = id;
	chunk["object"] = "chat.completion.chunk";
	chunk["created"] = (int64_t)time(NULL);
	chunk["model"] = model_name;
	json	choice;

	choice["index"] = 0;
	choice["delta"] = delta;
	choice["finish_reason"] = finish_reason == NULL ? json(nullptr)
			: json(finish_reason);
	chunk["choices"] = json::array({choice});
	return (chunk);
}

/* The pull side: called repeatedly by cpp-httplib's own chunked-write
 * loop (write_content_chunked(), httplib.cpp) until it returns false or
 * calls sink.done(). Never blocks forever: pop_wait()'s own bounded
 * timeout is the only way this function regains control to check
 * sink.is_writable() (Section 20 -- there is no other hook cpp-httplib
 * exposes for "has the peer gone away" while no data is ready to write). */
static bool	stream_provide(std::shared_ptr<s_stream_request_state> state,
				size_t offset, httplib::DataSink &sink)
{
	(void)offset;
	s_stream_event	ev;

	if (!state->queue.pop_wait(std::chrono::milliseconds(200), &ev))
	{
		if (!sink.is_writable())
		{
			state->cancel_flag.store(true, std::memory_order_relaxed);
			return (false);
		}
		return (true);	/* nothing ready yet, peer still alive -- httplib
						 * calls this function again immediately */
	}
	switch (ev.type)
	{
		case MEMBRANE_STREAM_EVENT_TOKEN:
		{
			json	delta;

			delta["content"] = ev.text;
			std::string	frame = sse_frame(stream_chunk_json(state->id,
					state->model_name, delta, NULL));
			return (sink.write(frame.data(), frame.size()));
		}
		case MEMBRANE_STREAM_EVENT_DONE:
		{
			json	empty_delta = json::object();
			std::string	frame = sse_frame(stream_chunk_json(state->id,
					state->model_name, empty_delta,
					ev.finish_reason.c_str()));

			if (!sink.write(frame.data(), frame.size()))
				return (false);
			if (ev.include_usage)
			{
				json	usage_chunk;

				usage_chunk["id"] = state->id;
				usage_chunk["object"] = "chat.completion.chunk";
				usage_chunk["created"] = (int64_t)time(NULL);
				usage_chunk["model"] = state->model_name;
				usage_chunk["choices"] = json::array();
				usage_chunk["usage"] = {
					{"prompt_tokens", ev.prompt_tokens},
					{"completion_tokens", ev.completion_tokens},
					{"total_tokens", ev.prompt_tokens + ev.completion_tokens},
				};
				std::string	usage_frame = sse_frame(usage_chunk);

				if (!sink.write(usage_frame.data(), usage_frame.size()))
					return (false);
			}
			static const char	done_marker[] = "data: [DONE]\n\n";

			if (!sink.write(done_marker, sizeof(done_marker) - 1))
				return (false);
			sink.done();
			return (true);
		}
		case MEMBRANE_STREAM_EVENT_ERROR:
		{
			/* Section 18: after streaming has begun, a failure is a
			 * terminal SSE event, never a status-code change (headers
			 * are already committed to text/event-stream) and never a
			 * crash. Same {"error": {"code", "message"}} shape as a
			 * pre-stream JSON error, so a client parsing every `data:`
			 * payload as JSON can distinguish it from a normal chunk by
			 * the presence of the "error" key alone. */
			json	err_payload;

			err_payload["error"] = {{"code", ev.error_code},
				{"message", ev.error_message}};
			std::string	frame = sse_frame(err_payload);

			if (!sink.write(frame.data(), frame.size()))
				return (false);
			static const char	done_marker[] = "data: [DONE]\n\n";

			if (!sink.write(done_marker, sizeof(done_marker) - 1))
				return (false);
			sink.done();
			return (true);
		}
		case MEMBRANE_STREAM_EVENT_CANCELLED:
		default:
			/* No reader is meant to observe this in practice (our only
			 * cancellation trigger IS this same function detecting a
			 * dead peer, at which point nothing further is written) --
			 * closing cleanly rather than writing anything is still the
			 * correct behavior if it is ever reached some other way. */
			sink.done();
			return (true);
	}
}

static void	stream_release(std::shared_ptr<s_stream_request_state> state,
				bool success)
{
	(void)success;
	/* Regardless of how the stream ended (client finished reading
	 * normally, a write failed, or the server itself is shutting down --
	 * cpp-httplib's own is_shutting_down() path can exit write_content_
	 * chunked()'s loop WITHOUT ever calling stream_provide() again, so
	 * this cancel_flag store is the only guaranteed signal in that case),
	 * a still-running generation serves no purpose once the Response
	 * object is being destroyed -- always request cancellation before
	 * joining, so this never blocks waiting on a runaway generation. */
	state->cancel_flag.store(true, std::memory_order_relaxed);
	if (state->worker.joinable())
		state->worker.join();
	/* Section 27: back to READY (never left at GENERATING) -- still
	 * under state->server_lock (still held here, unlocked right below),
	 * so this transition is exactly as synchronized as every other one. */
	state->st->model_state.store(MEMBRANE_MODEL_STATE_READY,
		std::memory_order_relaxed);
	state->server_lock.unlock();
}

static void	handle_chat_completions(s_membrane_server_model_state *st,
				const httplib::Request &req, httplib::Response &res)
{
	/* Section 29 of the task: admission is checked FIRST, before any
	 * other work (even JSON parsing) -- a server already at capacity
	 * gives an immediate, cheap 503 rather than spending any more work
	 * on a request it cannot serve promptly anyway. The ticket's own
	 * RAII destructor releases the slot on every return path below
	 * (including every early "return ;") for the non-streaming path;
	 * for the streaming path, ownership is explicitly moved into the
	 * async s_stream_request_state so the slot stays held for the
	 * whole streamed response, not just this function's own synchronous
	 * prefix. */
	request_admission_ticket_t	admission_ticket
			= membrane_try_admit(&st->admission_gate);

	if (!admission_ticket.admitted())
	{
		res.set_header("Retry-After", "1");
		send_json_error(res, 503, "SERVER_BUSY", "too many concurrent chat "
			"completion requests are already being handled -- retry "
			"shortly (this server has no continuous batching and fully "
			"serializes generation, Section 24/29 of the task)");
		return ;
	}
	membrane_registry_t	reg = refresh_and_snapshot_registry(st);
	json				body;

	try
	{
		body = json::parse(req.body);
	}
	catch (const json::parse_error &)
	{
		send_json_error(res, 400, "INVALID_REQUEST", "request body is not "
			"valid JSON");
		return ;
	}
	/* Section 9 of the Mega Phase B task: a request may omit "model"
	 * entirely (or send it as an empty string) iff a default_model is
	 * configured -- substituted below, once, right after this shape
	 * check. A request with no "model" field AND no configured default
	 * is still a 400, exactly as before this phase. */
	bool	has_model_field = body.is_object() && body.contains("model")
			&& body["model"].is_string() && !body["model"].get<std::string>()
				.empty();

	if (!body.is_object() || (!has_model_field && st->default_model.empty())
		|| !body.contains("messages") || !body["messages"].is_array()
		|| body["messages"].empty())
	{
		send_json_error(res, 400, "INVALID_REQUEST", "request must be a "
			"JSON object with a string \"model\" (or a configured default "
			"model) and a non-empty \"messages\" array");
		return ;
	}
	bool	want_stream = body.contains("stream") && body["stream"].is_boolean()
			&& body["stream"].get<bool>() == true;
	/* Real OpenAI convention (Section 19 of the Mega Phase B task,
	 * "usage handling"): a streaming response never includes usage
	 * unless the request explicitly opts in via stream_options.
	 * include_usage -- ignored entirely (never an error) when stream is
	 * false, matching every other unsupported-but-harmless request field
	 * this endpoint already tolerates (e.g. temperature/top_p). */
	bool	want_usage_in_stream = want_stream && body.contains("stream_options")
			&& body["stream_options"].is_object()
			&& body["stream_options"].value("include_usage", false);
	/* Request-shape validation (this loop) runs BEFORE the registry
	 * lookup below -- a malformed request is a 400 regardless of
	 * whether the named model happens to exist, never a 404 that
	 * implies "fix the model name and this would have worked" (real
	 * bug found and fixed during testing: the original code checked
	 * model existence first, so a malformed message against an
	 * unregistered model name incorrectly reported 404 instead of
	 * 400). */
	std::vector<llama_chat_message>	chat;
	std::vector<std::string>			role_storage;
	std::vector<std::string>			content_storage;

	for (const auto &msg : body["messages"])
	{
		if (!msg.is_object() || !msg.contains("role")
			|| !msg.contains("content") || !msg["role"].is_string()
			|| !msg["content"].is_string())
		{
			send_json_error(res, 400, "INVALID_REQUEST", "every message "
				"must have a string \"role\" and a string \"content\"");
			return ;
		}
		role_storage.push_back(msg["role"]);
		content_storage.push_back(msg["content"]);
	}
	std::string	model_name = has_model_field ? body["model"].get<std::string>()
			: st->default_model;
	const membrane_registry_entry_t	*entry = membrane_registry_find(reg,
			model_name);

	if (entry == NULL)
	{
		send_json_error(res, 404, "MODEL_NOT_FOUND", "no model named '"
			+ model_name + "' is registered (see `membrane model list`)");
		return ;
	}
	for (size_t i = 0; i < role_storage.size(); ++i)
	{
		llama_chat_message	m;

		m.role = role_storage[i].c_str();
		m.content = content_storage[i].c_str();
		chat.push_back(m);
	}
	int	max_tokens = 512;

	if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
		max_tokens = body["max_tokens"];
	else if (body.contains("max_completion_tokens")
		&& body["max_completion_tokens"].is_number_integer())
		max_tokens = body["max_completion_tokens"];
	if (max_tokens < 1)
		max_tokens = 1;

	std::unique_lock<std::mutex>	lock(st->mtx);
	std::string					tmpl;
	std::string					tmpl_err;

	if (st->model_loaded && st->loaded_name == entry->name
		&& !st->cached_chat_template.empty())
		tmpl = st->cached_chat_template;
	else if (!load_chat_template(entry->path, &tmpl, &tmpl_err))
	{
		send_json_error(res, 500, "CHAT_TEMPLATE_UNAVAILABLE", tmpl_err);
		return ;
	}
	std::string	prompt_text;

	if (!apply_chat_template(tmpl, chat, &prompt_text))
	{
		send_json_error(res, 500, "CHAT_TEMPLATE_FAILED", "the model's "
			"chat template could not be applied to these messages");
		return ;
	}
	std::string	err_code;
	std::string	err_message;
	int			err_status = 500;

	if (!ensure_model_loaded(st, reg, *entry, prompt_text, max_tokens,
			&err_code, &err_message, &err_status))
	{
		send_json_error(res, err_status, err_code, err_message);
		return ;
	}
	st->cached_chat_template = tmpl;

	char	id_buf[64];

	snprintf(id_buf, sizeof(id_buf), "chatcmpl-%llx",
		(unsigned long long)time(NULL) ^ (unsigned long long)(size_t)&res);
	if (want_stream)
	{
		/* Section 18: everything that can fail with a normal JSON
		 * status-code error has already happened above (parse, shape,
		 * model lookup, chat template, model load) -- from this point on,
		 * headers are about to commit to text/event-stream, so any LATER
		 * failure (only reachable from generation itself, on the worker
		 * thread) becomes a terminal SSE event instead (stream_provide()'s
		 * MEMBRANE_STREAM_EVENT_ERROR case), never a status-code change. */
		auto	state = std::make_shared<s_stream_request_state>();

		state->st = st;
		state->model_path = entry->path;	/* a real copy -- `entry` points
									 * into `reg`, a snapshot local to
									 * THIS function, which is about to be
									 * destroyed; req_o.model_path must
									 * point into storage that outlives
									 * it (this string, owned by `state`)
									 * instead (Section 32's registry hot-
									 * reload made `reg` a per-request
									 * copy, not the long-lived registry
									 * membrane_server_run() used to pass
									 * by reference). */
		fill_auto_opts(&state->req_o, state->model_path.c_str(), max_tokens);
		state->prompt_text = prompt_text;
		state->model_name = model_name;
		state->max_tokens = max_tokens;
		state->include_usage = want_usage_in_stream;
		state->id = id_buf;
		state->server_lock = std::move(lock);
		state->admission_ticket = std::move(admission_ticket);	/* Section
									 * 29: the slot stays held for the
									 * whole async stream, released by
									 * stream_release() -- not by this
									 * function's own (already-passed)
									 * return. */
		st->model_state.store(MEMBRANE_MODEL_STATE_GENERATING,
			std::memory_order_relaxed);
		state->worker = std::thread(stream_worker_fn, state);
		res.set_header("Cache-Control", "no-cache");
		res.set_chunked_content_provider("text/event-stream",
			[state](size_t offset, httplib::DataSink &sink)
			{ return (stream_provide(state, offset, sink)); },
			[state](bool success) { stream_release(state, success); });
		return ;
	}

	membrane_run_opts_t	req_o;

	fill_auto_opts(&req_o, entry->path.c_str(), max_tokens);

	membrane_generation_request_t	gen_req = {};
	membrane_generation_result_t	gen_res;

	gen_req.o = &req_o;
	gen_req.prompt_text = prompt_text;
	gen_req.ctx_size = 0;	/* auto-size to THIS request's own prompt,
							 * bounded by whatever gpu_layers/KV precision
							 * were already fixed at load time -- Section
							 * 5 of the task: "persistent model, new
							 * context per request." */
	gen_req.token_cb = NULL;
	st->model_state.store(MEMBRANE_MODEL_STATE_GENERATING,
		std::memory_order_relaxed);
	membrane_session_generate(&st->session, gen_req, &gen_res);
	st->model_state.store(MEMBRANE_MODEL_STATE_READY,
		std::memory_order_relaxed);
	if (!gen_res.ok)
	{
		if (gen_res.err.set)
			send_json_error(res, 500, gen_res.err.reason_code[0] != '\0'
				? gen_res.err.reason_code : "GENERATION_FAILED",
				"generation failed for this request");
		else
			send_json_error(res, 500, "GENERATION_FAILED", "generation "
				"failed for this request");
		return ;
	}

	json	response;

	response["id"] = id_buf;
	response["object"] = "chat.completion";
	response["created"] = (int64_t)time(NULL);
	response["model"] = model_name;
	json	choice;

	choice["index"] = 0;
	choice["message"] = {{"role", "assistant"}, {"content", gen_res.text}};
	choice["finish_reason"] = ((size_t)gen_res.gen_result.tokens.size()
			>= (size_t)max_tokens) ? "length" : "stop";
	response["choices"] = json::array({choice});
	size_t	prompt_tokens = gen_res.prompt_tokens.size();
	size_t	completion_tokens = gen_res.gen_result.tokens.size();

	response["usage"] = {
		{"prompt_tokens", prompt_tokens},
		{"completion_tokens", completion_tokens},
		{"total_tokens", prompt_tokens + completion_tokens},
	};
	const char	*kv_name;

	membrane_kv_precision_name_to_json(gen_res.effective_kv_mode, &kv_name);
	response["membrane"] = {
		{"context", gen_res.ctx_size},
		{"gpu_layers", st->session.gs.gpu_layers_selected},
		{"kv_precision", kv_name},
		{"kv_placement", st->session.gs.kv_placement_resolved
			? "placed" : "default"},
		{"sampling", "greedy (temperature/top_p not yet supported -- "
			"any request value is accepted and ignored)"},
	};
	res.set_content(response.dump(), "application/json");
}

int	membrane_server_run(const membrane_server_options_t &opts)
{
	/* Real bug found and fixed during PR B3 development: g_stop_requested
	 * is a file-scope global (the only race-free way membrane_server_
	 * request_stop()/a signal handler can reach into a running instance
	 * from outside), but it was never reset here -- every OTHER piece of
	 * per-run state (`state`, `svr`, ...) is a fresh local, so a SECOND
	 * membrane_server_run() call in the same process (e.g. a test
	 * harness running two differently-configured server instances
	 * sequentially) would see a stale `true` left over from the
	 * PREVIOUS instance's own shutdown and exit its poll loop
	 * immediately, never actually serving a single request. Every real
	 * product call site (the `membrane serve` CLI, one process per
	 * invocation) happened to never hit this, since a fresh process
	 * always starts with g_stop_requested's own static initializer
	 * (false) -- confirmed as a real, previously-undetected gap, not a
	 * hypothetical one, by test_server.cpp's own new admission-gate test
	 * needing a second, differently-configured instance. */
	g_stop_requested.store(false);

	std::string	registry_path = opts.registry_path.empty()
			? membrane_registry_resolve_path() : opts.registry_path;

	if (registry_path.empty())
	{
		fprintf(stderr, "membrane serve: neither XDG_DATA_HOME nor HOME "
			"is set -- cannot locate the model registry\n");
		return (1);
	}

	s_membrane_server_model_state	state;
	membrane_registry_error_t		reg_err;

	/* Mega Phase B, PR B3, Section 32: the registry now lives on `state`
	 * itself (registry_path/registry_mtime_ns alongside it) so it can be
	 * hot-refreshed per-request via refresh_and_snapshot_registry() --
	 * this initial load is identical to before, just writing into
	 * state.registry instead of a local variable the handlers used to
	 * capture by reference. registry_mtime_ns is left at its default (0)
	 * -- the first refresh_and_snapshot_registry() call will always see
	 * a real mtime != 0 and reload from scratch anyway, so there is no
	 * need to stat() the file twice here just to seed it "correctly". */
	if (!membrane_registry_load(registry_path, &state.registry, &reg_err))
	{
		fprintf(stderr, "membrane serve: could not load the model "
			"registry: %s\n", reg_err.message.c_str());
		return (1);
	}
	state.registry_path = registry_path;

	std::string	bind = opts.bind_address.empty() ? "127.0.0.1"
			: opts.bind_address;

	if (bind != "127.0.0.1" && bind != "localhost" && !opts.allow_non_loopback)
	{
		fprintf(stderr, "membrane serve: refusing to bind '%s' -- only "
			"127.0.0.1/localhost is allowed without an explicit opt-in "
			"(Section 28 of the Mega Phase A task: no silent LAN "
			"exposure)\n", bind.c_str());
		return (1);
	}
	if (bind != "127.0.0.1" && bind != "localhost")
		fprintf(stderr, "membrane serve: WARNING -- binding to '%s', not "
			"loopback-only. This server has NO authentication. Anyone who "
			"can reach this address can run inference as you.\n",
			bind.c_str());

	state.default_model = opts.default_model;
	membrane_runtime_init(&state.rt);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	httplib::Server	svr;
	int				port = opts.port;

	svr.Get("/health", handle_health);
	svr.Get("/v1/models", [&](const httplib::Request &rq,
			httplib::Response &rs) { handle_models(&state, rq, rs); });
	svr.Get("/v1/status", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_status(&state, bind, port, rq, rs); });
	svr.Post("/v1/chat/completions", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_chat_completions(&state, rq, rs); });

	if (!svr.bind_to_port(bind, port))
	{
		fprintf(stderr, "membrane serve: could not bind %s:%d (port "
			"already in use?)\n", bind.c_str(), port);
		if (state.model_loaded)
			membrane_model_close(&state.session);
		membrane_runtime_shutdown(&state.rt);
		return (1);
	}
	printf("MEMBRANE server listening on http://%s:%d\n", bind.c_str(),
		port);
	fflush(stdout);

	std::thread	listener([&svr]() { svr.listen_after_bind(); });

	/* Real bug found and fixed during local testing: svr.is_running()
	 * starts false and only becomes true once the listener thread's own
	 * listen_internal() actually begins running -- std::thread's
	 * constructor returning is NOT a guarantee the new thread has
	 * started executing yet. Without this wait, the poll loop below
	 * could see is_running()==false on its very first check (the
	 * listener thread simply hadn't started yet) and exit immediately;
	 * svr.stop() would then be a silent no-op (its own is_running_
	 * check also still false at that moment, so it never closes the
	 * listening socket), and listener.join() would then block forever
	 * on a listener that had, by then, actually started serving --
	 * confirmed directly: SIGTERM was delivered and handled (proven via
	 * an instrumented build) but the process never exited. httplib's own
	 * wait_until_ready() is the correct, race-free primitive for this. */
	svr.wait_until_ready();
	while (!g_stop_requested.load() && svr.is_running())
	{
		struct timespec	ts = {0, 100000000L};	/* 100ms poll -- Section
												 * 38: graceful shutdown,
												 * never a busy loop */

		nanosleep(&ts, NULL);
	}
	svr.stop();
	if (listener.joinable())
		listener.join();
	if (state.model_loaded)
		membrane_model_close(&state.session);
	membrane_runtime_shutdown(&state.rt);
	printf("MEMBRANE server stopped\n");
	return (0);
}
