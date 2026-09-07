#include "server.h"
#include "membrane/posix_compat.h"

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
#include "decode_concurrency.h"
#include "request_state.h"
#include "fs_util.h"

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

/* Mega Phase E, PR E1: see decode_concurrency.h's own top comment for the
 * full rationale. Floored at 1 (never 0 -- unlike the admission gate,
 * whose whole purpose includes a test-only "reject the very first
 * request" capacity-0 mode, a decode gate that can never admit anything
 * would deadlock every real request forever, not reject them). */
static int	membrane_max_concurrent_decode(void)
{
	const char	*env = getenv("MEMBRANE_MAX_CONCURRENT_DECODE");

	if (env != NULL && env[0] != '\0')
	{
		int	parsed = atoi(env);

		if (parsed >= 1)
			return (parsed);
	}
	return (MEMBRANE_DEFAULT_MAX_CONCURRENT_DECODE);
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
	/* Mega Phase E, PR E1: mtx's SCOPE narrowed from "held for the whole
	 * generation" (Section 24 of the Mega Phase A task, the original
	 * comment this replaces) to "held only while reading/mutating the
	 * model-lifecycle fields below (model_loaded/loaded_name/session/
	 * cached_chat_template) and while incrementing decode_inflight" --
	 * see decode_slot_enter()/decode_slot_exit()/ensure_model_loaded()'s
	 * own comments. Actual decode (membrane_session_generate()) now runs
	 * WITHOUT mtx held, on a per-request COPY of `session` (never the
	 * shared struct itself -- see handle_chat_completions()'s own PR E1
	 * comment for why a shared-struct copy, not a shared reference, is
	 * required for correctness). */
	std::mutex					mtx;
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

	/* Mega Phase E, PR E1, Section 5/9/10 of the task: the real concurrent-
	 * decode bound -- see decode_concurrency.h's own top comment. Distinct
	 * from admission_gate above: admission_gate bounds "how many requests
	 * may be admitted at all" (unchanged from PR B3, still 8 by default);
	 * decode_gate bounds "how many of those admitted requests may be
	 * inside llama_decode() at the same physical moment" (2 by default) --
	 * an admitted request past decode_gate's own capacity waits (in the
	 * QUEUED request_state_t) rather than being rejected. */
	decode_concurrency_gate_t	decode_gate{membrane_max_concurrent_decode()};

	/* Mega Phase E, PR E1, Section 12/29 of the task: how many requests
	 * are currently inside membrane_session_generate() for the CURRENTLY
	 * loaded model, right now -- the real "is it safe to close this
	 * model's session out from under someone" signal. Every mutation
	 * (increment in decode_slot_enter(), decrement in decode_slot_exit())
	 * happens either while mtx is already held (enter) or by taking mtx
	 * itself around the mutation (exit) specifically so drain_cv's own
	 * wait_for() predicate check can never race a concurrent decrement
	 * into a lost wakeup (the standard condition_variable pitfall of
	 * mutating predicate state without the SAME mutex the waiter checks
	 * it under) -- see decode_slot_exit()'s own comment. */
	std::atomic<int>			decode_inflight{0};
	std::condition_variable	drain_cv;	/* paired with mtx; see
										 * ensure_model_loaded()'s own PR E1
										 * drain-wait comment */

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

/* Mega Phase E, PR E1: call ONLY while st->mtx is already held (both real
 * call sites -- handle_chat_completions() non-stream and stream branches
 * -- call this right before releasing/moving the lock they already took
 * to run ensure_model_loaded()). Bumps decode_inflight and, on the
 * 0->1 transition, reflects that in model_state -- GENERATING now
 * honestly means "at least one request is decoding", not "exactly one",
 * now that E1 allows real concurrency. */
static void	decode_slot_enter(s_membrane_server_model_state *st)
{
	if (st->decode_inflight.fetch_add(1, std::memory_order_acq_rel) == 0)
		st->model_state.store(MEMBRANE_MODEL_STATE_GENERATING,
			std::memory_order_relaxed);
}

/* Mega Phase E, PR E1: the counterpart to decode_slot_enter() -- takes
 * st->mtx itself (never assume the caller already holds it; unlike enter,
 * this runs from contexts that do NOT hold mtx, e.g. right after
 * membrane_session_generate() returns with the lock already released, or
 * from stream_release() on the streaming worker-join path). Mutating
 * decode_inflight and checking "did it reach zero" under the SAME mutex
 * ensure_model_loaded()'s drain_cv.wait_for() predicate also uses is what
 * makes that wait race-free (see the mtx member's own top comment). The
 * notify_all() itself deliberately happens AFTER releasing the lock
 * (the standard "don't wake a thread that will immediately block on a
 * lock you're still holding" optimization) -- this is safe specifically
 * because the state mutation that made the predicate true already
 * happened while the lock was held, so no wakeup can be lost. */
static void	decode_slot_exit(s_membrane_server_model_state *st)
{
	{
		std::lock_guard<std::mutex>	lock(st->mtx);

		if (st->decode_inflight.fetch_sub(1, std::memory_order_acq_rel) == 1
			&& st->model_loaded)
			st->model_state.store(MEMBRANE_MODEL_STATE_READY,
				std::memory_order_relaxed);
	}
	st->drain_cv.notify_all();
}

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

	/* Mega Phase D, PR D8, Section 15 of the task: a real, disclosed PR
	 * D7 finding was that an extremely oversized prompt made the server
	 * unresponsive for minutes on a memory-constrained host, with no
	 * fast, structured rejection. Root-caused this phase (real source
	 * read, runtime_session.cpp's own membrane_resolve_ctx_auto()):
	 * that function tokenizes the ENTIRE prompt (a real vocab-only
	 * model load + real BPE tokenization, itself scaling with prompt
	 * size) BEFORE ever comparing the result against entry.model_max_
	 * context -- the existing, already-correct fast rejection
	 * (context_recommender.c's own real "minimum_required_context >
	 * model_max_context" check) only runs AFTER that cost is already
	 * paid. This cheap, PRE-tokenization guard skips that cost entirely
	 * for a prompt that could not possibly fit regardless of tokenizer
	 * behavior: every GGUF tokenizer this project supports (byte-level
	 * BPE and SentencePiece alike) merges bytes INTO tokens, never the
	 * reverse, so real token count can never exceed real raw byte
	 * count -- a x6 margin (real English/code text averages ~4 bytes/
	 * token; even deliberately dense text stays well above 1.5-2
	 * bytes/token) leaves zero realistic chance of a false rejection
	 * while still catching a prompt that is unambiguously, wildly too
	 * large. entry.model_max_context is already real, cheap, cached
	 * registry data (captured once at `membrane model add`/`install`
	 * time) -- no new I/O, no new dependency. A prompt this check lets
	 * through still goes through the exact same real, unmodified
	 * tokenize-then-compare pipeline below; this only ever short-
	 * circuits the unambiguous, extreme case. */
	if (entry.model_max_context > 0
		&& first_prompt.size() > (size_t)entry.model_max_context * 6)
	{
		*err_code = "CTX_TOO_SMALL_FOR_PROMPT";
		*err_message = "this prompt is far larger than the model's real "
			"maximum context (" + std::to_string(entry.model_max_context)
			+ " tokens) -- rejected before an expensive real tokenization "
			"attempt; try a shorter prompt";
		*http_status = 400;
		return (false);
	}

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
				std::unique_lock<std::mutex> &lock,
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
		/* Mega Phase E, PR E1, Section 6/16 of the task: a model switch
		 * must never free (membrane_model_close(), below) a session a
		 * concurrently-running request is still decoding against -- that
		 * request holds its own COPY of st->session (see handle_chat_
		 * completions()'s own PR E1 comment), but the copy's `model`
		 * pointer is only valid as long as the ORIGINAL llama_model
		 * membrane_model_close() would free is still alive. Bounded wait
		 * (5s, matching the pre-E1 handle_activate_model() retry budget
		 * this replaces) rather than an indefinite one -- a switch that
		 * can't get a safe window in 5s reports MODEL_SWITCH_BUSY (503)
		 * and leaves the CURRENT model fully intact and still serving,
		 * never a partial/corrupted unload. */
		bool	drained = st->drain_cv.wait_for(lock,
				std::chrono::seconds(5),
				[st] { return (st->decode_inflight.load(
						std::memory_order_acquire) == 0); });

		if (!drained)
		{
			*err_code = "MODEL_SWITCH_BUSY";
			*err_message = "a generation against the currently-loaded "
				"model is still in progress -- retry shortly";
			*http_status = 503;
			return (false);
		}
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
		int64_t	mtime_ns = membrane_stat_mtime_ns(stat_buf);

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

/* Mega Phase D, PR D6, Section 12 of the task: a loopback-only,
 * MEMBRANE-specific (never /v1/..., never part of the OpenAI-compatible
 * surface) admin endpoint letting `membrane use` trigger a live model
 * switch without requiring `membrane serve`/the service to be restarted.
 * This is a THIN wrapper around ensure_model_loaded() -- the exact same
 * function handle_chat_completions() itself already calls -- never a
 * second switch/lifecycle policy: Section 17's idempotence ("already
 * active" -> no reload) and Section 16's failure-recovery honesty (a
 * failed switch that successfully restores the previous model is still
 * reported as a FAILURE for the requested model, with the real recovered
 * model named in active_model) both come from ensure_model_loaded()
 * itself, for free.
 *
 * The context-recommendation pipeline ensure_model_loaded() drives
 * needs SOME real prompt text to size context against (see try_load_
 * one()'s own call to membrane_resolve_ctx_auto()) -- there is no real
 * user message yet for an admin-triggered warm switch, so a short, fixed
 * placeholder chat turn is applied through the model's own real chat
 * template (load_chat_template()/apply_chat_template(), the SAME
 * functions handle_chat_completions() itself uses -- never a second
 * templating path). Real, disclosed design finding (docs/model-
 * lifecycle.md has the detail): context_recommender.c's own algorithm
 * always maximizes recommended_context to fit real host memory (the
 * prompt's own token count is only ever a FLOOR), so this placeholder
 * sizes context almost identically to what a real first chat message
 * would in the common case -- not a corner this endpoint is quietly
 * cutting. */
static void	handle_activate_model(s_membrane_server_model_state *st,
				const httplib::Request &req, httplib::Response &res)
{
	json	body;

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
	if (!body.is_object() || !body.contains("model")
		|| !body["model"].is_string() || body["model"].get<std::string>()
			.empty())
	{
		send_json_error(res, 400, "INVALID_REQUEST", "request must be a "
			"JSON object with a non-empty string \"model\"");
		return ;
	}
	std::string			model_name = body["model"];
	membrane_registry_t	reg = refresh_and_snapshot_registry(st);
	const membrane_registry_entry_t	*entry = membrane_registry_find(reg,
			model_name);

	if (entry == NULL)
	{
		send_json_error(res, 404, "MODEL_NOT_FOUND", "no model named '"
			+ model_name + "' is registered (see `membrane model list`)");
		return ;
	}
	/* Mega Phase E, PR E1: mtx now only guards the model-lifecycle fields
	 * (not the whole generation, see mtx's own top comment) -- a plain
	 * blocking lock is correct and fast here (real generation never holds
	 * mtx anymore). "Never unload a model beneath an active generation"
	 * (Section 15 of the original Mega Phase A task) is now enforced by
	 * ensure_model_loaded()'s own bounded drain-wait against decode_
	 * inflight (see its PR E1 comment) -- this replaces the old try_lock
	 * retry loop's SERVER_BUSY with that function's own MODEL_SWITCH_BUSY,
	 * same "bounded wait, never indefinite, never corrupts anything"
	 * contract. */
	std::unique_lock<std::mutex>	lock(st->mtx);

	if (st->model_loaded && st->loaded_name == entry->name)
	{
		const char	*kv_name;

		membrane_kv_precision_name_to_json(st->session.gs.adaptive_used
			? st->session.gs.adaptive_selected_mode : MEMBRANE_KV_STORE_NATIVE,
			&kv_name);
		json	j;

		j["ok"] = true;
		j["already_active"] = true;
		j["active_model"] = st->loaded_name;
		j["backend"] = st->session.gs.requested
			? st->session.gs.backend_selected : "CPU";
		res.set_content(j.dump(), "application/json");
		return ;
	}
	std::string	tmpl;
	std::string	tmpl_err;

	if (!load_chat_template(entry->path, &tmpl, &tmpl_err))
	{
		send_json_error(res, 500, "CHAT_TEMPLATE_UNAVAILABLE", tmpl_err);
		return ;
	}
	std::vector<llama_chat_message>	chat;
	llama_chat_message					warmup;

	warmup.role = "user";
	warmup.content = "Hello";
	chat.push_back(warmup);
	std::string	prompt_text;

	if (!apply_chat_template(tmpl, chat, &prompt_text))
	{
		send_json_error(res, 500, "CHAT_TEMPLATE_FAILED", "the model's chat "
			"template could not be applied to a warm-up message");
		return ;
	}
	std::string	err_code;
	std::string	err_message;
	int			err_status = 500;
	bool		ok = ensure_model_loaded(st, lock, reg, *entry, prompt_text,
			512, &err_code, &err_message, &err_status);

	if (ok)
		st->cached_chat_template = tmpl;
	json	j;

	j["ok"] = ok;
	j["already_active"] = false;
	j["active_model"] = st->model_loaded ? json(st->loaded_name)
			: json(nullptr);
	if (st->model_loaded)
	{
		j["backend"] = st->session.gs.requested
			? st->session.gs.backend_selected : "CPU";
	}
	if (!ok)
	{
		j["error"] = {{"code", err_code}, {"message", err_message}};
		res.status = err_status;
	}
	res.set_content(j.dump(), "application/json");
}

static void	handle_models(s_membrane_server_model_state *st,
				const httplib::Request &, httplib::Response &res)
{
	membrane_registry_t	reg = refresh_and_snapshot_registry(st);
	json	arr = json::array();

	for (const auto &e : reg.entries)
	{
		json	m;

		/* Section 17/18 of the Mega Phase D, PR D7 task: `e.name` is the
		 * SAME registry name `membrane use NAME`/`membrane model use
		 * NAME` already select by -- one canonical, client-safe id, never
		 * a filesystem path and never a second alias a client would need
		 * to learn. */
		m["id"] = e.name;
		m["object"] = "model";
		/* Real, first-attempt Python openai SDK compatibility finding
		 * (PR D7): openai.types.model.Model declares `created: int` as a
		 * REQUIRED field (no default) -- client.models.list() genuinely
		 * failed pydantic validation with this field missing. e.added_
		 * at_unix is real data already captured at `add`/`install` time
		 * (registry_core.h), never fabricated. */
		m["created"] = e.added_at_unix;
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
	/* Mega Phase D, PR D6, Section 20/22: exposes the SAME default_model
	 * `membrane serve`/the service read at startup (server_config.h,
	 * unchanged by a live switch -- see server.h's own top comment on
	 * default_model) so `membrane status`/`membrane doctor` can honestly
	 * compare "what the config prefers" against "what is actually
	 * loaded right now" without a second, independently-drifting config
	 * read of their own. */
	j["default_model"] = st->default_model.empty() ? json(nullptr)
			: json(st->default_model);
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
 * Mega Phase E, PR E1 UPDATE: st->mtx is now held ONLY for the brief
 * model-lifecycle setup above (ensure_model_loaded() + snapshotting a
 * per-request COPY of st->session into state->local_session) -- released
 * before the worker thread is even spawned. The worker thread instead
 * acquires a bounded decode_concurrency_gate_t ticket (decode_concurrency.h)
 * around its own membrane_session_generate() call, so multiple streaming
 * (and/or non-streaming) requests genuinely decode at the same time, up
 * to that gate's own capacity -- not serialized on st->mtx anymore. What
 * IS still serialized/protected: (a) decode_inflight (incremented while
 * st->mtx is held, at setup time; decremented, under st->mtx again, in
 * stream_release() once the worker thread has fully joined) guarantees a
 * model switch can never free a session a request is still using, and
 * (b) each request operates on its OWN local_session copy, never the
 * shared st->session, so concurrent requests cannot race on session->gs's
 * own per-call-mutated planner fields (see s_stream_request_state's own
 * local_session comment).
 */

/* Mega Phase D, PR D7, Section 12 of the task: real "stop" string
 * support, implemented ENTIRELY at this file's own product layer --
 * decode_loop.h/run_generation() needed no change at all. Reuses the
 * SAME cancel_flag mechanism PR B2 already built for client-disconnect
 * cancellation (a plain std::atomic<bool>* polled once per free-running
 * step) for a second purpose: a token_cb that accumulates the response
 * text and, on a real match, truncates the about-to-be-emitted text at
 * the match and sets cancel_flag -- generation genuinely halts (real
 * compute savings, not a post-hoc truncation of output that was already
 * fully generated). Earliest match across every requested stop string
 * wins, matching real OpenAI behavior. */
static bool	find_stop_match(const std::string &acc,
				const std::vector<std::string> &stops, size_t *out_pos)
{
	bool	found = false;
	size_t	best = std::string::npos;

	for (const auto &s : stops)
	{
		if (s.empty())
			continue ;
		size_t	pos = acc.find(s);

		if (pos != std::string::npos && (!found || pos < best))
		{
			found = true;
			best = pos;
		}
	}
	if (found)
		*out_pos = best;
	return (found);
}

/* The non-streaming analogue of stream_token_cb()'s own stop-matching
 * logic above -- no queue/SSE framing involved, just a plain
 * accumulator and a local cancel_flag this function's own caller (the
 * non-streaming branch of handle_chat_completions()) owns. */
struct s_nonstream_stop_ctx
{
	std::string						acc;
	const std::vector<std::string>	*stops;
	std::atomic<bool>				*cancel_flag;
	bool							matched = false;
	size_t							match_pos = 0;
};

static void	nonstream_stop_token_cb(const char *piece, size_t piece_len,
				int step, void *user_data)
{
	(void)step;
	s_nonstream_stop_ctx	*ctx = (s_nonstream_stop_ctx *)user_data;

	ctx->acc.append(piece, piece_len);
	size_t	match_pos;

	if (find_stop_match(ctx->acc, *ctx->stops, &match_pos))
	{
		ctx->matched = true;
		ctx->match_pos = match_pos;
		ctx->cancel_flag->store(true, std::memory_order_relaxed);
	}
}

struct s_stream_worker_ctx
{
	stream_queue_t				*queue;
	std::atomic<bool>			*cancel_flag;	/* PR D7: writable now --
											 * stream_token_cb() itself
											 * sets this on a real stop-
											 * sequence match (see its own
											 * top comment) */
	std::string					utf8_pending;	/* worker-thread-only, no
											 * synchronization needed */
	/* PR D7: NULL/empty stop_sequences (the default, and every request
	 * that does not send "stop") leaves this codepath completely
	 * untouched -- byte-identical to before this phase (Section 4's own
	 * "explicit beats implicit" convention). full_acc/stop_matched are
	 * worker-thread-only (same single-writer/single-reader reasoning as
	 * utf8_pending above -- stream_token_cb and stream_worker_fn's own
	 * post-generate read of stop_matched never race, since the callback
	 * only ever runs synchronously inside membrane_session_generate(),
	 * which has already returned by the time stop_matched is read). */
	const std::vector<std::string>	*stop_sequences = NULL;
	std::string					full_acc;
	bool						stop_matched = false;
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
	std::string	emit_text = wctx->utf8_pending.substr(0, emit_len);

	wctx->utf8_pending.erase(0, emit_len);
	if (wctx->stop_sequences != NULL && !wctx->stop_sequences->empty())
	{
		size_t	already_sent = wctx->full_acc.size();

		wctx->full_acc += emit_text;
		size_t	match_pos;

		if (find_stop_match(wctx->full_acc, *wctx->stop_sequences,
				&match_pos))
		{
			if (match_pos > already_sent)
			{
				std::string	truncated = emit_text.substr(0,
						match_pos - already_sent);
				s_stream_event	ev;

				ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
				ev.text = truncated;
				wctx->queue->push_blocking(std::move(ev));
			}
			wctx->stop_matched = true;
			wctx->cancel_flag->store(true, std::memory_order_relaxed);
			return ;
		}
	}
	s_stream_event	ev;

	ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
	ev.text = emit_text;
	wctx->queue->push_blocking(std::move(ev));
}

struct s_stream_request_state
{
	stream_queue_t				queue;
	std::atomic<bool>			cancel_flag{false};	/* the QUEUE's own
											 * disconnect-shutdown flag
											 * ONLY (stream_queue_t's own
											 * top comment/push_blocking()'s
											 * own contract) -- true here
											 * makes push_blocking() give
											 * up and drop ANY further
											 * event, including a real
											 * terminal one, matching its
											 * pre-D7 "a dead connection's
											 * queue can never wedge the
											 * worker thread forever"
											 * design. PR D7's own real,
											 * first-attempt bug: a stop-
											 * sequence match must NEVER
											 * set THIS flag (it would
											 * silently drop the real
											 * DONE frame a still-connected
											 * client is waiting for) --
											 * gen_cancel_flag below is the
											 * separate, correct flag for
											 * that. */
	std::atomic<bool>			gen_cancel_flag{false};	/* PR D7: what
											 * actually gets passed to
											 * run_generation() as its own
											 * cancel_flag parameter --
											 * set true by EITHER a real
											 * client disconnect (stream_
											 * provide()/stream_release(),
											 * alongside `cancel_flag`
											 * above) OR a stop-sequence
											 * match (stream_token_cb(),
											 * alongside `stop_matched`,
											 * deliberately NEVER
											 * alongside `cancel_flag`
											 * above) */
	std::thread					worker;
	membrane_model_session_t	local_session;	/* Mega Phase E, PR E1: a
										 * real VALUE COPY of st->session,
										 * snapshotted in handle_chat_
										 * completions() while st->mtx is
										 * still held, and used for this
										 * request's own membrane_session_
										 * generate() call INSTEAD of the
										 * shared st->session -- required
										 * for correctness now that decode
										 * runs without st->mtx held and
										 * potentially concurrently with
										 * another request's own decode
										 * (see handle_chat_completions()'s
										 * own PR E1 top comment for the
										 * full race this avoids: gs's
										 * fields -- adaptive_used, backend_
										 * selected, gpu_layers_selected,
										 * kv_placement_resolved -- are
										 * genuinely re-written per generate()
										 * call, so two concurrent requests
										 * sharing one session would race on
										 * them and could report EACH
										 * OTHER's planner results). `model`
										 * itself (the llama_model* the copy
										 * shares with st->session) stays
										 * valid for as long as decode_
										 * inflight is nonzero -- see
										 * ensure_model_loaded()'s own
										 * drain-wait. */

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
	std::vector<std::string>	stop_sequences;	/* PR D7 -- empty (the
											 * default) leaves the stop-
											 * matching codepath in
											 * stream_token_cb() entirely
											 * unreached */

	/* Mega Phase E, PR E1, Section 4 of the task: explicit request
	 * lifecycle -- see request_state.h's own top comment for the full
	 * state list/meaning. Starts QUEUED (admitted, not yet decoding);
	 * every later transition below is a plain, traceable store, never
	 * inferred after the fact from other bookkeeping. Atomic only
	 * because it is set from the worker thread and could in principle be
	 * read from another (no reader exists yet -- this is the same
	 * "observability primitive, not a policy of its own" scope request_
	 * state.h's own top comment describes). */
	std::atomic<membrane_request_state_t>	req_state{
			membrane_request_state_t::QUEUED};

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
	/* PR D7, real bug found and fixed via a real Python-SDK-driven hang:
	 * this must be gen_cancel_flag, NEVER state->cancel_flag -- stream_
	 * queue_t::push_blocking() itself refuses to deliver ANY further
	 * event (including the real terminal DONE frame a still-connected
	 * client is waiting for) once state->cancel_flag is true (its own
	 * documented "a dead connection's queue can never wedge the worker
	 * thread forever" contract). A stop-sequence match is emphatically
	 * NOT a dead connection -- using state->cancel_flag here silently
	 * dropped the DONE frame every time, hanging the client forever
	 * (confirmed with real stderr instrumentation: cancel_flag really
	 * was set, run_generation() really did stop, but push_blocking()
	 * silently discarded the terminal event because that SAME flag was
	 * already true). gen_cancel_flag is a separate flag with no such
	 * side effect on the queue. */
	wctx.cancel_flag = &state->gen_cancel_flag;

	membrane_generation_request_t	gen_req = {};
	membrane_generation_result_t	gen_res;

	if (!state->stop_sequences.empty())
		wctx.stop_sequences = &state->stop_sequences;
	gen_req.o = &state->req_o;
	gen_req.prompt_text = state->prompt_text;
	gen_req.ctx_size = 0;
	gen_req.token_cb = stream_token_cb;
	gen_req.token_cb_ud = &wctx;
	gen_req.cancel_flag = &state->gen_cancel_flag;
	/* Mega Phase E, PR E1, Section 5/9: blocks here (never rejects -- this
	 * request was already admitted) until fewer than decode_gate's own
	 * capacity requests are actually decoding -- the real bound on
	 * concurrent llama_decode() load. Released automatically (RAII) the
	 * moment this scope exits, right after generate() returns below. */
	decode_concurrency_ticket_t	decode_ticket(&state->st->decode_gate);

	state->req_state.store(membrane_request_state_t::ACTIVE,
		std::memory_order_relaxed);
	membrane_session_generate(&state->local_session, gen_req, &gen_res);
	/* Flush whatever partial UTF-8 tail never resolved -- better to emit
	 * it than to silently drop real generated bytes (Section 21: only
	 * reachable if generation stopped (limit/EOG/cancellation) exactly
	 * mid-character, a rare edge case, never treated as an excuse to
	 * lose output). PR D7: skipped when a stop sequence matched -- those
	 * trailing bytes are generated content PAST the point the client
	 * asked to cut the response at, never emitted. */
	if (!wctx.utf8_pending.empty() && !wctx.stop_matched)
	{
		s_stream_event	flush_ev;

		flush_ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
		flush_ev.text = wctx.utf8_pending;
		state->queue.push_blocking(std::move(flush_ev));
	}
	s_stream_event	terminal;

	/* PR D7: a stop-sequence match also sets cancel_flag (reusing the
	 * same real mechanism client-disconnect cancellation already uses),
	 * so gen_res.cancelled is true here too -- wctx.stop_matched is the
	 * one thing that tells these two real, different causes apart. A
	 * stop match is a NORMAL, successful completion from the client's
	 * point of view (a real "data: [DONE]" DONE frame with finish_
	 * reason "stop"), never the silent-close CANCELLED path a dead
	 * peer gets. */
	if (gen_res.cancelled && !wctx.stop_matched)
	{
		terminal.type = MEMBRANE_STREAM_EVENT_CANCELLED;
		state->req_state.store(membrane_request_state_t::CANCELLED,
			std::memory_order_relaxed);
	}
	else if (!gen_res.ok)
	{
		terminal.type = MEMBRANE_STREAM_EVENT_ERROR;
		terminal.error_code = (gen_res.err.set
				&& gen_res.err.reason_code[0] != '\0')
			? gen_res.err.reason_code : "GENERATION_FAILED";
		terminal.error_message = "generation failed for this request";
		state->req_state.store(membrane_request_state_t::ERROR,
			std::memory_order_relaxed);
	}
	else
	{
		terminal.type = MEMBRANE_STREAM_EVENT_DONE;
		state->req_state.store(membrane_request_state_t::DONE,
			std::memory_order_relaxed);
		terminal.finish_reason = wctx.stop_matched ? "stop"
			: (((size_t)gen_res.gen_result.tokens.size()
					>= (size_t)state->max_tokens) ? "length" : "stop");
		terminal.prompt_tokens = gen_res.prompt_tokens.size();
		/* PR D7: gen_res.gen_result.tokens.size() counts every token the
		 * runtime actually decoded, including the one whose OWN piece
		 * text contained (or completed) the stop match and was itself
		 * only partially emitted -- usage.completion_tokens can be at
		 * most one token higher than what the client actually received
		 * as text in that real, disclosed edge case (docs/api-
		 * contract.md), never fabricated lower to look tidy. */
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
			/* A real disconnect: both flags -- state->cancel_flag (so
			 * the queue itself gives up on any further push, correctly,
			 * nobody is left to read them) AND gen_cancel_flag (so
			 * run_generation() actually stops decoding; see stream_
			 * worker_fn()'s own top comment on why these are two
			 * separate flags, not one). */
			state->cancel_flag.store(true, std::memory_order_relaxed);
			state->gen_cancel_flag.store(true, std::memory_order_relaxed);
			state->req_state.store(membrane_request_state_t::CANCELLING,
				std::memory_order_relaxed);
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
	 * joining, so this never blocks waiting on a runaway generation.
	 * Both flags -- see stream_worker_fn()'s own top comment on why. */
	state->cancel_flag.store(true, std::memory_order_relaxed);
	state->gen_cancel_flag.store(true, std::memory_order_relaxed);
	if (state->worker.joinable())
		state->worker.join();
	/* Mega Phase E, PR E1: this request is now FULLY done with local_
	 * session's own copy of the model pointer -- decode_slot_exit()
	 * decrements decode_inflight (only now, after join(), guaranteeing
	 * membrane_session_generate() has truly returned) and, on the ->0
	 * transition, sets model_state back to READY and wakes any
	 * ensure_model_loaded() drain-wait blocked on a model switch. Section
	 * 27's original "back to READY" guarantee, preserved -- just no
	 * longer via server_lock (removed; decode never held st->mtx). */
	decode_slot_exit(state->st);
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
	/* Mega Phase D, PR D7, Section 15 of the task: "tools"/"tool_choice"
	 * are checked BEFORE model resolution, same "malformed/unsupported
	 * request is rejected regardless of which model was named" precedent
	 * as the messages-shape loop above. Explicit rejection, never a
	 * silent drop -- MEMBRANE does not execute tools, and silently
	 * ignoring a real tool schema would let a client believe a tool call
	 * might come back when one never will (Section 10's own "a field
	 * that changes intended behavior must be rejected, not ignored"). */
	if (body.contains("tools") || body.contains("tool_choice"))
	{
		send_json_error(res, 400, "UNSUPPORTED_TOOL_CALLING", "tool "
			"calling is not supported -- remove \"tools\"/\"tool_choice\" "
			"from the request (see docs/api-contract.md)");
		return ;
	}
	/* Section 14: response_format is only ever a real no-op for its own
	 * default ("text", or the field simply absent) -- any other value
	 * (e.g. "json_object"/"json_schema") is rejected explicitly rather
	 * than silently ignored, since this server has no JSON-mode grammar/
	 * constrained-decoding path that could actually honor it (Section
	 * 14: "do not pretend JSON mode exists merely by prompt injection"). */
	if (body.contains("response_format") && body["response_format"].is_object()
		&& body["response_format"].value("type", std::string("text"))
			!= "text")
	{
		send_json_error(res, 400, "UNSUPPORTED_RESPONSE_FORMAT",
			"response_format type '"
			+ body["response_format"].value("type", std::string(""))
			+ "' is not supported -- only the default (\"text\", or the "
			"field omitted entirely) is supported");
		return ;
	}
	/* Section 12: real stop-string support -- see find_stop_match()/
	 * stream_token_cb()'s own top comments for the full design. Accepts
	 * the same two real shapes OpenAI's own API does: a single string,
	 * or an array of up to 4 strings (OpenAI's own real limit) -- more
	 * than 4 is a real, explicit 400, never silently truncated to 4. */
	std::vector<std::string>	stop_sequences;

	if (body.contains("stop") && !body["stop"].is_null())
	{
		const json	&stop_field = body["stop"];

		if (stop_field.is_string())
			stop_sequences.push_back(stop_field.get<std::string>());
		else if (stop_field.is_array())
		{
			if (stop_field.size() > 4)
			{
				send_json_error(res, 400, "INVALID_REQUEST", "\"stop\" "
					"accepts at most 4 strings");
				return ;
			}
			for (const auto &item : stop_field)
			{
				if (!item.is_string())
				{
					send_json_error(res, 400, "INVALID_REQUEST", "every "
						"\"stop\" entry must be a string");
					return ;
				}
				stop_sequences.push_back(item.get<std::string>());
			}
		}
		else
		{
			send_json_error(res, 400, "INVALID_REQUEST", "\"stop\" must "
				"be a string or an array of up to 4 strings");
			return ;
		}
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

	if (!ensure_model_loaded(st, lock, reg, *entry, prompt_text, max_tokens,
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
		state->stop_sequences = stop_sequences;
		state->id = id_buf;
		state->local_session = st->session;	/* PR E1: real value copy,
									 * taken while `lock` is still held --
									 * see s_stream_request_state's own
									 * local_session comment. */
		state->admission_ticket = std::move(admission_ticket);	/* Section
									 * 29: the slot stays held for the
									 * whole async stream, released by
									 * stream_release() -- not by this
									 * function's own (already-passed)
									 * return. */
		decode_slot_enter(st);	/* PR E1: still holding `lock` here --
									 * marks this request as using the
									 * current model before the lock (and
									 * therefore the guarantee nothing else
									 * can close it out from under us) is
									 * released, right below. */
		lock.unlock();	/* PR E1: decode itself never holds st->mtx
									 * anymore -- released before the
									 * worker thread is even spawned, so a
									 * second request's own ensure_model_
									 * loaded() call can proceed
									 * immediately if it names the same
									 * (already-loaded) model. */
		state->worker = std::thread(stream_worker_fn, state);
		res.set_header("Cache-Control", "no-cache");
		res.set_chunked_content_provider("text/event-stream",
			[state](size_t offset, httplib::DataSink &sink)
			{ return (stream_provide(state, offset, sink)); },
			[state](bool success) { stream_release(state, success); });
		return ;
	}

	/* Mega Phase E, PR E1: same real value-copy-then-release pattern as
	 * the streaming branch above -- see s_stream_request_state's own
	 * local_session comment for why a copy (never &st->session directly)
	 * is required once decode can run concurrently with another
	 * request's own decode. */
	membrane_model_session_t	local_session = st->session;

	decode_slot_enter(st);
	lock.unlock();

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
	/* PR D7: only wired up when the request actually asked for one (the
	 * default -- empty stop_sequences -- keeps this branch's own
	 * gen_req fields at their pre-D7 values, byte-identical behavior). */
	std::atomic<bool>		nonstream_stop_flag{false};
	s_nonstream_stop_ctx	nonstream_stop_ctx;

	if (!stop_sequences.empty())
	{
		nonstream_stop_ctx.stops = &stop_sequences;
		nonstream_stop_ctx.cancel_flag = &nonstream_stop_flag;
		gen_req.token_cb = nonstream_stop_token_cb;
		gen_req.token_cb_ud = &nonstream_stop_ctx;
		gen_req.cancel_flag = &nonstream_stop_flag;
	}
	/* Mega Phase E, PR E1, Section 4 of the task: explicit request
	 * lifecycle for the non-streaming path too -- a plain local (this
	 * function has no concurrent reader of its own request's state,
	 * unlike the streaming path's cross-thread s_stream_request_state::
	 * req_state), same real transition points as stream_worker_fn's own
	 * QUEUED->ACTIVE->{DONE,CANCELLED,ERROR}. */
	membrane_request_state_t	req_state = membrane_request_state_t::QUEUED;

	{
		/* Mega Phase E, PR E1, Section 5/9: bounded, blocking -- see the
		 * streaming branch's own decode_concurrency_ticket_t comment.
		 * Scoped so the slot is released the moment generate() returns,
		 * before decode_slot_exit()'s own (separate) st->mtx acquisition
		 * below -- no functional requirement either order would break,
		 * just avoids holding it any longer than needed. */
		decode_concurrency_ticket_t	decode_ticket(&st->decode_gate);

		req_state = membrane_request_state_t::ACTIVE;
		membrane_session_generate(&local_session, gen_req, &gen_res);
	}
	decode_slot_exit(st);
	/* A stop-sequence match sets gen_req.cancel_flag, so gen_res.cancelled
	 * is true here too (ok is unaffected -- see runtime_session.h's own
	 * doc comment) -- never treated as a real failure: a stop match is a
	 * normal, successful completion (DONE), never CANCELLED, same
	 * distinction stream_worker_fn's own terminal classification makes. */
	if (gen_res.cancelled && !nonstream_stop_ctx.matched)
		req_state = membrane_request_state_t::CANCELLED;
	else if (!gen_res.ok)
		req_state = membrane_request_state_t::ERROR;
	else
		req_state = membrane_request_state_t::DONE;
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

	std::string	final_text = (nonstream_stop_ctx.matched)
			? nonstream_stop_ctx.acc.substr(0, nonstream_stop_ctx.match_pos)
			: gen_res.text;
	json		response;

	response["id"] = id_buf;
	response["object"] = "chat.completion";
	response["created"] = (int64_t)time(NULL);
	response["model"] = model_name;
	json	choice;

	choice["index"] = 0;
	choice["message"] = {{"role", "assistant"}, {"content", final_text}};
	choice["finish_reason"] = nonstream_stop_ctx.matched ? "stop"
		: (((size_t)gen_res.gen_result.tokens.size()
				>= (size_t)max_tokens) ? "length" : "stop");
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
		{"gpu_layers", local_session.gs.gpu_layers_selected},
		{"kv_precision", kv_name},
		{"kv_placement", local_session.gs.kv_placement_resolved
			? "placed" : "default"},
		{"sampling", "greedy (temperature/top_p not yet supported -- "
			"any request value is accepted and ignored)"},
	};
	res.set_content(response.dump(), "application/json");
	/* req_state's own real value (DONE at this point, always -- the
	 * earlier CANCELLED/ERROR branches both return before reaching here)
	 * has no external reader yet on the non-streaming path (unlike
	 * s_stream_request_state::req_state, which a future admin/status
	 * surface could read cross-thread) -- Section 4 of the task only
	 * requires the transitions themselves to be explicit, traceable
	 * stores, not that every request's lifecycle already be exposed
	 * externally. Silences -Wunused-but-set-variable without faking a
	 * reader. */
	(void)req_state;
}

/* Mega Phase E, PR E1: a real, previously-undiscovered hang-class bug
 * found via this phase's own real concurrent-load testing (scripts/
 * verify-continuous-batching.py), root-caused directly rather than
 * dismissed as environmental. Every chat request opens a brand-new
 * llama_context (this project's own "persistent model, new context per
 * request" architecture, unchanged) -- llama.cpp's OWN internal logger
 * (llama_log_set()'s default, unset here before this fix) writes real,
 * substantial per-context-creation diagnostic output (graph_reserve/
 * sched_reserve/resolve_fused_ops lines -- measured directly: ~13 KB per
 * request on this project's own real SmolLM2-135M fixture) to stderr on
 * EVERY request, completely outside MEMBRANE's own quiet/verbose design
 * (tools/membrane-run/main.cpp's own quiet_log_callback already
 * suppresses this for the CLI, via the exact same llama_log_set() API --
 * this call site was simply missing on the SERVER path). Under any real
 * deployment/test harness whose stdout/stderr sink has bounded buffering
 * and no continuous reader (a plain OS pipe -- e.g. Python's own
 * subprocess.PIPE without a draining thread, confirmed as this bug's
 * own real, reproduced trigger; NOT systemd/journald, which reads
 * continuously and never blocks this way), enough real sequential
 * requests (~5 on this project's own real measurement) fill that buffer
 * and the request-handling thread's own fprintf() call BLOCKS FOREVER --
 * which in turn starves every other request waiting on that thread's
 * held decode_gate/admission slot, exactly the "one request corrupts/
 * starves others" failure Section 6 of the task forbids. Suppressing
 * llama.cpp's own default verbose logging here (server output was never
 * meant to include raw internal graph-reservation debug lines on every
 * request in the first place -- a real hygiene fix independent of the
 * hang) removes the root cause entirely, rather than papering over it by
 * only fixing this project's OWN test scripts (also done, see scripts/
 * soak-test-server.py and scripts/concurrency-soak-server.py's own PR E1
 * updates) -- a real user's own process supervisor could have the exact
 * same bounded-buffer characteristic this project's test harness
 * happened to surface first. */
static void	membrane_server_log_callback(enum ggml_log_level level,
				const char *text, void *user_data)
{
	(void)user_data;
	if (level == GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

int	membrane_server_run(const membrane_server_options_t &opts)
{
	llama_log_set(membrane_server_log_callback, NULL);

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

	/* Mega Phase B, PR B4, Section 47 of the task: real bug found and
	 * fixed while testing "port already in use" startup robustness --
	 * cpp-httplib's own default_socket_options() sets SO_REUSEPORT (not
	 * just SO_REUSEADDR) whenever the platform has it (Linux does),
	 * which lets a SECOND, completely independent membrane serve
	 * process bind_to_port() the exact SAME address:port successfully
	 * WHILE THE FIRST IS STILL ACTIVELY LISTENING -- the kernel then
	 * load-balances incoming connections across both processes'
	 * completely separate model state, a silent, confusing multi-
	 * instance situation this project never wants (Section 23: "one
	 * active model" is a promise about ONE process's own state, not
	 * something a caller can accidentally defeat by starting a second
	 * instance). Confirmed directly: a second `membrane serve` on an
	 * already-bound port used to print "MEMBRANE server listening"
	 * successfully instead of failing. Overriding the socket options to
	 * SO_REUSEADDR only (still lets a clean restart quickly rebind a
	 * port stuck in TIME_WAIT, e.g. systemd's own RestartSec=2) fixes
	 * this: a second instance now correctly fails with EADDRINUSE. */
	svr.set_socket_options([](socket_t sock)
		{ httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });

	svr.Get("/health", handle_health);
	svr.Get("/v1/models", [&](const httplib::Request &rq,
			httplib::Response &rs) { handle_models(&state, rq, rs); });
	svr.Get("/v1/status", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_status(&state, bind, port, rq, rs); });
	svr.Post("/v1/chat/completions", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_chat_completions(&state, rq, rs); });
	svr.Post("/membrane/v1/models/activate", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_activate_model(&state, rq, rs); });

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
