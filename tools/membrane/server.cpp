#include "server.h"
#include "membrane/posix_compat.h"

#include <algorithm>
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
#include "gpu_device.h"
#include "registry_core.h"
#include "gpu_policy.h"
#include "product_cli.h"
#include "utf8_stream.h"
#include "stream_queue.h"
#include "request_admission.h"
#include "decode_concurrency.h"
#include "request_state.h"
#include "residency_planner.h"
#include "fs_util.h"

#include <sys/stat.h>

using json = nlohmann::json;

/*
 * Mega Phase B, PR B3, Section 27 of the task: an explicit model-lifecycle
 * state machine -- replaces the previous implicit "model_loaded bool +
 * loaded_name string" boolean soup with a named, observable state,
 * synchronized the same way every other piece of a slot's mutable state
 * already is (transitions only ever happen while that slot's own mtx --
 * s_model_slot::mtx, Mega Phase E, PR E2 -- is held).
 * EMPTY: no model has ever been loaded into this slot, or the last one
 * was cleanly unloaded/evicted -- a healthy, normal state, never an
 * error.
 * LOADING / UNLOADING: a load/unload is in progress (both only ever
 * observed transiently, since they happen while the slot's own mtx is
 * held for the whole request -- exposed anyway for /v1/status's own
 * honesty and for a future phase that might narrow the lock).
 * READY: a model is loaded and idle.
 * GENERATING: a model is loaded and actively generating (streaming or
 * not).
 * ERROR: a model switch/eviction failed AND the attempt to restore the
 * previously-loaded model (see acquire_model_slot()'s own recovery
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
 * "1 active generation per slot, the rest waiting on that slot's own
 * mtx" reality this server already has (no intra-model batching), while
 * still bounding how
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

/* Mega Phase E, PR E2, Section 10 of the task: "start conservatively --
 * max 2 resident models, do NOT make this unlimited". Deliberately NOT
 * exposed as a documented server_config.h setting for the same reason
 * MEMBRANE_MAX_CONCURRENT_DECODE isn't (Section 8's "keep minimal"
 * convention) -- the env override exists only so test_server.cpp can
 * force a small, deterministic slot count (e.g. 1) without needing
 * MEMBRANE_DEFAULT_MAX_RESIDENT_MODELS-many real models to exhaust
 * residency. Floored at 1 (a server with zero slots could never load
 * anything at all) and capped at a generous but bounded 8 -- a
 * misconfigured huge value would otherwise let a test/operator silently
 * defeat the whole "bounded, conservative residency" policy Section 10
 * asks for. */
# define MEMBRANE_DEFAULT_MAX_RESIDENT_MODELS	2
# define MEMBRANE_MAX_RESIDENT_MODELS_HARD_CAP	8

static int	membrane_max_resident_models(void)
{
	const char	*env = getenv("MEMBRANE_MAX_RESIDENT_MODELS");

	if (env != NULL && env[0] != '\0')
	{
		int	parsed = atoi(env);

		if (parsed >= 1 && parsed <= MEMBRANE_MAX_RESIDENT_MODELS_HARD_CAP)
			return (parsed);
	}
	return (MEMBRANE_DEFAULT_MAX_RESIDENT_MODELS);
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

/*
 * Mega Phase E, PR E2, Section 10-12 of the task: ONE resident-model
 * slot. Everything that used to be the single, global "the loaded
 * model" fields on s_membrane_server_model_state (see PR E1's own
 * comment on that struct, which this replaces) now lives here instead,
 * multiplied by up to membrane_max_resident_models() -- s_membrane_
 * server_model_state below holds a std::vector of these rather than one
 * inline copy. mtx's scope is UNCHANGED from PR E1's own contract, just
 * narrowed further to "one slot's own lifecycle fields" instead of "the
 * server's one and only model": held only while reading/mutating this
 * slot's model_loaded/loaded_name/session/cached_chat_template and while
 * incrementing decode_inflight; real decode still never holds it (a
 * per-request COPY of `session` is used instead -- see handle_chat_
 * completions()'s own PR E1 comment, unchanged by this phase). */
struct s_model_slot
{
	std::mutex					mtx;
	bool						model_loaded = false;
	std::string					loaded_name;
	membrane_model_session_t	session;
	std::string					cached_chat_template;	/* valid iff
										 * model_loaded is true and refers
										 * to `loaded_name` -- see PR E1's
										 * own comment on the field this
										 * replaces for the real, measured
										 * cost this cache avoids */

	/* Mega Phase B, PR B3, Section 27 (unchanged by PR E2, just now
	 * per-slot): only ever mutated while mtx is held, but readable
	 * atomically (e.g. by /v1/status) without a data race either way. */
	std::atomic<int>			model_state{MEMBRANE_MODEL_STATE_EMPTY};

	/* Mega Phase E, PR E1, Section 12/29 of the task (unchanged by PR
	 * E2, just now per-slot): how many requests are currently inside
	 * membrane_session_generate() against THIS slot's model, right now --
	 * the real "is it safe to close this out from under someone" signal.
	 * See decode_slot_exit()'s own comment for the condition_variable
	 * race this, paired with mtx, avoids. */
	std::atomic<int>			decode_inflight{0};
	std::condition_variable	drain_cv;	/* paired with mtx */

	/* Mega Phase E, PR E2, Section 12/13 of the task: eviction/pinning
	 * state -- see acquire_model_slot()'s own top comment for how these
	 * are actually used. Both are guarded by mtx (real settled state),
	 * EXCEPT last_used_seq's own read from acquire_model_slot()'s
	 * residency_mtx-held selection snapshot, which is a deliberate,
	 * harmless benign race (see that function's own comment on why a
	 * slightly-stale LRU timestamp used only to CHOOSE an eviction
	 * candidate, never to decide correctness, needs no stronger
	 * synchronization). */
	bool						pinned = false;
	int64_t						last_used_seq = 0;

	/* Mega Phase E, PR E2: guarded by s_membrane_server_model_state::
	 * residency_mtx ONLY (never mtx above) -- the real name this slot
	 * currently represents to residency SELECTION, kept in sync with
	 * loaded_name at every settled moment but the one thing a second
	 * request can observe WITHOUT waiting on this slot's own (possibly
	 * slow -- disk I/O, a bounded drain-wait) mtx. See residency_
	 * planner.h's own top comment on membrane_residency_slot_view_t::name
	 * for the full race this claim-before-you-load marker avoids: two
	 * concurrent requests for two DIFFERENT new model names must never
	 * both pick this same free/evictable slot. */
	std::string					claimed_name;
};

/*
 * Mega Phase E, PR E2, Section 10 of the task: the top-level server state.
 * PR E1's own single inline model (mtx/model_loaded/loaded_name/session/
 * cached_chat_template/model_state/decode_inflight/drain_cv) is now
 * `slots` -- a fixed-size (membrane_max_resident_models(), sized once at
 * construction, never resized afterward) vector of s_model_slot. Two
 * gates stay deliberately GLOBAL, not per-slot, both for the same real-
 * memory reason PR E1's own decode_gate comment gives: admission_gate
 * bounds how many requests may be admitted at all (unchanged since PR
 * B3); decode_gate bounds how many requests across EVERY resident model
 * combined may be inside llama_decode() at the same physical moment --
 * making it per-slot instead would let N slots multiply the real
 * concurrent-KV-cache memory pressure PR E1's own gate exists to bound,
 * exactly the "start conservatively" this task's Section 9/26 both ask
 * for.
 */
struct s_membrane_server_model_state
{
	membrane_runtime_t			rt = {};
	std::string					default_model;	/* Section 9 -- "" = none
										 * configured; a chat request
										 * omitting "model" falls back to
										 * this, never proactively loaded */

	request_admission_gate_t	admission_gate{membrane_max_pending_chat_requests()};
	decode_concurrency_gate_t	decode_gate{membrane_max_concurrent_decode()};

	/* Mega Phase E, PR E2, Section 10-12: residency_mtx guards ONLY the
	 * brief slot-SELECTION step (which slot serves a given model name --
	 * see acquire_model_slot()'s own top comment) and each slot's own
	 * claimed_name/last_used_seq bookkeeping; it is NEVER held during the
	 * actual (potentially slow) load/evict work, which uses that slot's
	 * own mtx instead -- a request for an already-resident model in slot
	 * B never blocks behind a concurrent cold load into slot A. */
	std::mutex					residency_mtx;
	std::vector<s_model_slot>	slots;

	s_membrane_server_model_state() : slots(membrane_max_resident_models())
	{
	}

	/* Mega Phase B, PR B3, Section 32 (unchanged by PR E2): the model
	 * registry lives HERE so it can be hot-refreshed from an mtime check
	 * without any caller needing to hold a slot's own mtx for the whole
	 * operation -- registry_mtx is a separate, short-lived lock. */
	std::mutex					registry_mtx;
	membrane_registry_t			registry;
	std::string					registry_path;
	int64_t						registry_mtime_ns = 0;
};

/* Mega Phase E, PR E2: a single process-wide monotonic counter standing
 * in for "time" for LRU purposes -- see s_model_slot::last_used_seq's own
 * comment. A plain incrementing counter (never wall-clock time) keeps
 * eviction ordering exactly reproducible in tests (residency_planner.h's
 * own test suite already covers the pure decision logic against
 * synthetic sequence numbers; this is the one real source of them). */
static std::atomic<int64_t>	g_residency_seq{0};

/* Mega Phase E, PR E1 (unchanged in spirit, now per-slot): call ONLY
 * while slot->mtx is already held (both real call sites -- handle_chat_
 * completions() non-stream and stream branches -- call this right before
 * releasing/moving the lock they already took to run acquire_model_
 * slot()). Bumps decode_inflight and, on the 0->1 transition, reflects
 * that in model_state -- GENERATING now honestly means "at least one
 * request is decoding", not "exactly one", now that E1 allows real
 * concurrency. */
static void	decode_slot_enter(s_model_slot *slot)
{
	if (slot->decode_inflight.fetch_add(1, std::memory_order_acq_rel) == 0)
		slot->model_state.store(MEMBRANE_MODEL_STATE_GENERATING,
			std::memory_order_relaxed);
}

/* Mega Phase E, PR E1 (unchanged in spirit, now per-slot): the
 * counterpart to decode_slot_enter() -- takes slot->mtx itself (never
 * assume the caller already holds it; unlike enter, this runs from
 * contexts that do NOT hold it, e.g. right after membrane_session_
 * generate() returns with the lock already released, or from stream_
 * release() on the streaming worker-join path). Mutating decode_inflight
 * and checking "did it reach zero" under the SAME mutex acquire_model_
 * slot()'s drain_cv.wait_for() predicate also uses is what makes that
 * wait race-free. The notify_all() itself deliberately happens AFTER
 * releasing the lock (the standard "don't wake a thread that will
 * immediately block on a lock you're still holding" optimization) --
 * this is safe specifically because the state mutation that made the
 * predicate true already happened while the lock was held, so no wakeup
 * can be lost. */
static void	decode_slot_exit(s_model_slot *slot)
{
	{
		std::lock_guard<std::mutex>	lock(slot->mtx);

		if (slot->decode_inflight.fetch_sub(1, std::memory_order_acq_rel) == 1
			&& slot->model_loaded)
			slot->model_state.store(MEMBRANE_MODEL_STATE_READY,
				std::memory_order_relaxed);
	}
	slot->drain_cv.notify_all();
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
 * reality, not stale data). Factored out of acquire_model_slot() so the
 * same real load path can be reused both for the actual requested switch
 * and for automatically restoring a previous model after a failed one
 * (see acquire_model_slot()'s own recovery logic below). */
static bool	try_load_one(membrane_runtime_t *rt, s_model_slot *slot,
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

	if (!membrane_model_open(rt, entry.path.c_str(), o, o.ctx, &session,
			&open_err))
	{
		*err_code = "MODEL_LOAD_FAILED";
		*err_message = "the model could not be loaded";
		*http_status = 500;
		return (false);
	}
	slot->session = session;
	slot->model_loaded = true;
	slot->loaded_name = entry.name;
	return (true);
}

/*
 * Mega Phase E, PR E2, Section 10-14 of the task: resolves which
 * resident SLOT serves `entry`, loading/evicting/restoring as needed --
 * the direct multi-slot generalization of PR E1's own ensure_model_
 * loaded() (which this replaces), now choosing among up to membrane_
 * max_resident_models() slots instead of always operating on the one
 * single model.
 *
 * Two-phase, matching s_membrane_server_model_state::residency_mtx's own
 * top comment: phase 1 (residency_mtx, always brief -- a snapshot copy
 * plus one pure membrane_plan_residency() call, residency_planner.h)
 * picks/claims a slot; phase 2 (that ONE slot's own mtx, potentially
 * slow -- disk I/O, the real context/GPU planner, a bounded drain-wait)
 * does the real load/evict/restore. A concurrent request naming a
 * DIFFERENT, already-resident model never blocks on this at all (Section
 * 14: hot switch never blocks on an unrelated slot's own load) -- it
 * only ever contends on the brief phase-1 lock, never phase 2's.
 *
 * Contract: returns NULL only when membrane_plan_residency() itself
 * reports EXHAUSTED (Section 12: "if all residents are non-evictable,
 * fail safely") -- no slot was ever claimed, *err_code/message/status
 * describe RESIDENCY_EXHAUSTED, and no caller state changed. Every OTHER
 * outcome returns the real slot that was engaged (with *out_lock holding
 * ITS mtx, still locked, exactly like PR E1's own `lock` parameter did)
 * -- callers must check `err_code->empty()` to tell success from
 * failure, since Section 16's own failure-recovery honesty (a failed
 * switch that successfully restores the previous model still reports
 * the ORIGINAL request as a failure, naming the real recovered model in
 * the slot the caller can still read) needs the slot pointer even when
 * this returns a failure.
 */
static s_model_slot	*acquire_model_slot(s_membrane_server_model_state *st,
				std::unique_lock<std::mutex> *out_lock,
				const membrane_registry_t &registry_snapshot,
				const membrane_registry_entry_t &entry,
				const std::string &first_prompt, int gen_tokens,
				std::string *err_code, std::string *err_message,
				int *http_status)
{
	int	slot_index;

	{
		std::lock_guard<std::mutex>						rlock(
				st->residency_mtx);
		std::vector<membrane_residency_slot_view_t>		views;

		for (const auto &s : st->slots)
		{
			membrane_residency_slot_view_t	v;

			v.name = s.claimed_name;
			v.model_loaded = s.model_loaded;
			v.pinned = s.pinned;
			v.generating = s.decode_inflight.load(
					std::memory_order_relaxed) > 0;
			v.last_used_seq = s.last_used_seq;
			views.push_back(v);
		}
		membrane_residency_plan_t	plan = membrane_plan_residency(views,
				entry.name);

		if (plan.decision == membrane_residency_decision_t::EXHAUSTED)
		{
			*err_code = "RESIDENCY_EXHAUSTED";
			*err_message = "no resident-model slot is available right "
				"now -- every resident model is either pinned or "
				"actively generating (see `membrane status`)";
			*http_status = 503;
			return (NULL);
		}
		slot_index = plan.slot_index;
		st->slots[slot_index].claimed_name = entry.name;
	}

	s_model_slot	*slot = &st->slots[slot_index];

	*out_lock = std::unique_lock<std::mutex>(slot->mtx);
	err_code->clear();
	err_message->clear();
	if (slot->model_loaded && slot->loaded_name == entry.name)
	{
		std::lock_guard<std::mutex>	rlock(st->residency_mtx);

		slot->last_used_seq = g_residency_seq.fetch_add(1);
		return (slot);
	}

	std::string	previous_name = slot->loaded_name;
	bool		had_previous = slot->model_loaded;

	if (slot->model_loaded)
	{
		/* Mega Phase E, PR E1, Section 6/16 of the task (unchanged in
		 * spirit, now per-slot): a model switch/eviction must never free
		 * (membrane_model_close(), below) a session a concurrently-
		 * running request is still decoding against. Bounded wait (5s)
		 * rather than an indefinite one -- see PR E1's own comment on
		 * this exact drain-wait for the full rationale. */
		bool	drained = slot->drain_cv.wait_for(*out_lock,
				std::chrono::seconds(5),
				[slot] { return (slot->decode_inflight.load(
						std::memory_order_acquire) == 0); });

		if (!drained)
		{
			std::lock_guard<std::mutex>	rlock(st->residency_mtx);

			/* Abort the swap entirely -- the old model is still
			 * genuinely resident in this slot, so residency selection
			 * must see it under its OLD name again, not the target's. */
			slot->claimed_name = previous_name;
			*err_code = "MODEL_SWITCH_BUSY";
			*err_message = "a generation against a resident model in "
				"this slot is still in progress -- retry shortly";
			*http_status = 503;
			return (slot);
		}
		slot->model_state.store(MEMBRANE_MODEL_STATE_UNLOADING,
			std::memory_order_relaxed);
		membrane_model_close(&slot->session);
		slot->model_loaded = false;
		slot->loaded_name.clear();
		slot->cached_chat_template.clear();
	}
	slot->model_state.store(MEMBRANE_MODEL_STATE_LOADING,
		std::memory_order_relaxed);
	if (try_load_one(&st->rt, slot, entry, first_prompt, gen_tokens,
			err_code, err_message, http_status))
	{
		slot->model_state.store(MEMBRANE_MODEL_STATE_READY,
			std::memory_order_relaxed);
		std::lock_guard<std::mutex>	rlock(st->residency_mtx);

		slot->last_used_seq = g_residency_seq.fetch_add(1);
		return (slot);
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

			slot->model_state.store(MEMBRANE_MODEL_STATE_LOADING,
				std::memory_order_relaxed);
			if (try_load_one(&st->rt, slot, *prev_entry, first_prompt,
					gen_tokens, &restore_err_code, &restore_err_message,
					&restore_http_status))
			{
				slot->model_state.store(MEMBRANE_MODEL_STATE_READY,
					std::memory_order_relaxed);
				std::lock_guard<std::mutex>	rlock(st->residency_mtx);

				/* Section 16: the ORIGINAL request (for `entry`) is
				 * still a failure -- err_code/err_message/http_status
				 * already describe it and are left untouched -- only
				 * the SLOT's own resting state improved (back to the
				 * previous, still-real model). */
				slot->claimed_name = previous_name;
				slot->last_used_seq = g_residency_seq.fetch_add(1);
				return (slot);
			}
		}
	}
	slot->model_state.store(MEMBRANE_MODEL_STATE_ERROR,
		std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex>	rlock(st->residency_mtx);

		slot->claimed_name.clear();
	}
	return (slot);
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
 * This is a THIN wrapper around acquire_model_slot() -- the exact same
 * function handle_chat_completions() itself already calls -- never a
 * second switch/lifecycle policy: Section 17's idempotence ("already
 * active" -> no reload) and Section 16's failure-recovery honesty (a
 * failed switch that successfully restores the previous model is still
 * reported as a FAILURE for the requested model, with the real recovered
 * model named in active_model) both come from acquire_model_slot()
 * itself, for free.
 *
 * The context-recommendation pipeline acquire_model_slot() drives
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
	/* Mega Phase E, PR E2: a cheap, residency_mtx-only peek for the
	 * common "already active, nothing to do" case -- skips the chat-
	 * template warm-up dance entirely when it can, same real-work-
	 * avoidance PR E1's own single-slot shortcut (which this replaces)
	 * already provided. Never itself a source of truth for correctness:
	 * a miss here (model_name resident but this peek raced and missed
	 * it) just falls through to the full acquire_model_slot() path
	 * below, which is always correct regardless. */
	{
		s_model_slot	*peek = NULL;

		{
			std::lock_guard<std::mutex>	rlock(st->residency_mtx);

			for (auto &s : st->slots)
			{
				if (s.claimed_name == model_name)
				{
					peek = &s;
					break ;
				}
			}
		}
		if (peek != NULL)
		{
			std::lock_guard<std::mutex>	slock(peek->mtx);

			if (peek->model_loaded && peek->loaded_name == model_name)
			{
				const char	*kv_name;

				membrane_kv_precision_name_to_json(
					peek->session.gs.adaptive_used
						? peek->session.gs.adaptive_selected_mode
						: MEMBRANE_KV_STORE_NATIVE, &kv_name);
				json	j;

				j["ok"] = true;
				j["already_active"] = true;
				j["active_model"] = peek->loaded_name;
				j["backend"] = peek->session.gs.requested
					? peek->session.gs.backend_selected : "CPU";
				res.set_content(j.dump(), "application/json");
				return ;
			}
		}
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
	std::string						err_code;
	std::string						err_message;
	int								err_status = 500;
	std::unique_lock<std::mutex>	lock;
	s_model_slot					*slot = acquire_model_slot(st, &lock, reg,
			*entry, prompt_text, 512, &err_code, &err_message, &err_status);
	bool							ok = slot != NULL && err_code.empty();

	if (ok)
		slot->cached_chat_template = tmpl;
	json	j;

	j["ok"] = ok;
	j["already_active"] = false;
	j["active_model"] = (slot != NULL && slot->model_loaded)
			? json(slot->loaded_name) : json(nullptr);
	if (slot != NULL && slot->model_loaded)
	{
		j["backend"] = slot->session.gs.requested
			? slot->session.gs.backend_selected : "CPU";
	}
	if (!ok)
	{
		j["error"] = {{"code", err_code}, {"message", err_message}};
		res.status = err_status;
	}
	res.set_content(j.dump(), "application/json");
}

/*
 * Mega Phase E, PR E2, Section 13 of the task: `membrane model pin/unpin`'s
 * server-side counterpart -- loopback-only, MEMBRANE-specific (never
 * /v1/...), same admin namespace as handle_activate_model() above.
 * "Pinned means: prefer not to evict. It does NOT mean violate memory
 * safety" (Section 13) -- pinning is a property of a CURRENTLY RESIDENT
 * slot, not a standing preference remembered for a not-yet-loaded model
 * (this project's own registry/server_config already own that kind of
 * durable preference, via default_model); pinning a name that is not
 * resident right now is a real, explicit 409, never silently
 * remembered for later. */
static void	handle_pin_model(s_membrane_server_model_state *st, bool pin,
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
	std::string		model_name = body["model"];
	s_model_slot	*target = NULL;

	{
		std::lock_guard<std::mutex>	rlock(st->residency_mtx);

		for (auto &s : st->slots)
		{
			if (s.claimed_name == model_name)
			{
				target = &s;
				break ;
			}
		}
	}
	if (target == NULL)
	{
		send_json_error(res, 409, "MODEL_NOT_RESIDENT", "'" + model_name
			+ "' is not currently resident -- pin/unpin only applies to "
			"a model that is already loaded (see `membrane status`)");
		return ;
	}
	std::lock_guard<std::mutex>	slock(target->mtx);

	if (!target->model_loaded || target->loaded_name != model_name)
	{
		send_json_error(res, 409, "MODEL_NOT_RESIDENT", "'" + model_name
			+ "' is not currently resident -- pin/unpin only applies to "
			"a model that is already loaded (see `membrane status`)");
		return ;
	}
	target->pinned = pin;
	json	j;

	j["ok"] = true;
	j["model"] = model_name;
	j["pinned"] = pin;
	res.set_content(j.dump(), "application/json");
}

/*
 * Mega Phase E, PR E3, Section 20 of the task: a MEMBRANE-specific
 * (never `/v1/...`) capabilities surface -- lets a real client (or a
 * human) discover what this build/host actually supports without
 * probing endpoints or reading source. Every field is a REAL fact
 * about this build/host, never a static aspiration: `backends` comes
 * from a real, live `membrane_gpu_list_devices()` enumeration (the
 * exact same one `membrane doctor`/the real GPU-selection pipeline
 * already use -- never a second, independently-drifting hardware
 * probe), and `resident_model_limit` is the real, current membrane_
 * max_resident_models() value (env-overridable, see that function's
 * own top comment) rather than a hardcoded "2". `tool_calling`/
 * `embeddings` are honestly `false` -- see docs/api-v1-stability.md's
 * own "Not implemented" section for why neither is faked via prompt
 * injection or a stub. */
static void	handle_capabilities(s_membrane_server_model_state *st,
				const httplib::Request &, httplib::Response &res)
{
	json	j;

	j["streaming"] = true;
	j["stop"] = true;
	j["tool_calling"] = false;
	j["embeddings"] = false;

	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n_devices = membrane_gpu_list_devices(devices,
			MEMBRANE_GPU_MAX_DEVICES);
	std::vector<std::string>	backend_names;

	for (size_t i = 0; i < n_devices; ++i)
	{
		std::string	name = devices[i].backend;

		if (std::find(backend_names.begin(), backend_names.end(), name)
			== backend_names.end())
			backend_names.push_back(name);
	}
	j["backends"] = backend_names;
	j["resident_model_limit"] = (int)st->slots.size();
#if defined(_WIN32)
	j["platform"] = "windows";
#elif defined(__APPLE__)
	j["platform"] = "macos";
#else
	j["platform"] = "linux";
#endif
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
/*
 * Mega Phase E, PR E2, Section 15 of the task: "expose clearly in
 * `membrane status`: active model, resident models, pinned, evictable,
 * backend, memory estimate". `loaded_model`/single-model `backend`/
 * `gpu_layers`/`kv_precision` (PR E1's own singular fields) are replaced
 * by `resident_models` -- an array, one entry per slot that actually
 * holds a model right now (an EMPTY/never-loaded slot is omitted
 * entirely, not reported as a null placeholder) -- since more than one
 * can now be true simultaneously. This MEMBRANE-specific endpoint (see
 * this function's own pre-existing top comment) is not yet under the
 * API v1 stability freeze (Section 18, a later phase -- E3), so
 * reshaping it here, before that freeze, is the intended time to do it,
 * not a breaking change against any already-frozen contract. */
static void	handle_status(s_membrane_server_model_state *st,
				const std::string &bind, int port, const httplib::Request &,
				httplib::Response &res)
{
	json	j;

	j["running"] = true;
	j["version"] = MEMBRANE_VERSION;
	j["endpoint"] = "http://" + bind + ":" + std::to_string(port);
	j["resident_model_limit"] = (int)st->slots.size();
	json	resident = json::array();

	for (auto &slot : st->slots)
	{
		std::lock_guard<std::mutex>	lock(slot.mtx);

		if (!slot.model_loaded)
			continue ;
		const char	*kv_name;

		membrane_kv_precision_name_to_json(
			slot.session.gs.adaptive_used
				? slot.session.gs.adaptive_selected_mode
				: MEMBRANE_KV_STORE_NATIVE, &kv_name);
		json	m;

		m["model"] = slot.loaded_name;
		m["state"] = membrane_model_state_name(
			slot.model_state.load(std::memory_order_relaxed));
		m["pinned"] = slot.pinned;
		m["evictable"] = !slot.pinned
			&& slot.decode_inflight.load(std::memory_order_relaxed) == 0;
		m["backend"] = slot.session.gs.requested
			? slot.session.gs.backend_selected : "CPU";
		m["gpu_layers"] = slot.session.gs.gpu_layers_selected;
		m["kv_precision"] = kv_name;
		m["estimated_model_bytes"] = slot.session.gs.estimated_model_bytes;
		m["estimated_kv_bytes"] = slot.session.gs.estimated_kv_bytes;
		resident.push_back(m);
	}
	j["resident_models"] = resident;
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
 * Mega Phase E, PR E1 UPDATE (and PR E2, Section 10: multi-model
 * residency): the resolved slot's own mtx (s_model_slot::mtx) is now
 * held ONLY for the brief model-lifecycle setup above (acquire_model_
 * slot() + snapshotting a per-request COPY of slot->session into state->
 * local_session) -- released before the worker thread is even spawned.
 * The worker thread instead acquires a bounded decode_concurrency_gate_t
 * ticket (decode_concurrency.h, still GLOBAL across every resident slot
 * -- see s_membrane_server_model_state's own top comment on why) around
 * its own membrane_session_generate() call, so multiple streaming (and/
 * or non-streaming) requests -- against the SAME resident model or, as
 * of PR E2, DIFFERENT simultaneously-resident ones -- genuinely decode
 * at the same time, up to that gate's own capacity, never serialized on
 * any one slot's mtx. What IS still serialized/protected: (a) decode_
 * inflight (incremented while the slot's mtx is held, at setup time;
 * decremented, under that same mtx again, in stream_release() once the
 * worker thread has fully joined) guarantees a model switch/eviction can
 * never free a session a request is still using, and (b) each request
 * operates on its OWN local_session copy, never the shared slot->session,
 * so concurrent requests cannot race on session->gs's own per-call-
 * mutated planner fields (see s_stream_request_state's own local_session
 * comment).
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
	membrane_model_session_t	local_session;	/* Mega Phase E, PR E1
										 * (unchanged in spirit under PR
										 * E2): a real VALUE COPY of
										 * slot->session, snapshotted in
										 * handle_chat_completions() while
										 * that slot's own mtx is still
										 * held, and used for this
										 * request's own membrane_session_
										 * generate() call INSTEAD of the
										 * shared slot->session -- required
										 * for correctness now that decode
										 * runs without the slot's mtx held
										 * and potentially concurrently with
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
										 * shares with slot->session) stays
										 * valid for as long as decode_
										 * inflight is nonzero -- see
										 * acquire_model_slot()'s own
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
	/* Mega Phase E, PR E2: `st` (for its now-global decode_gate ONLY --
	 * see s_membrane_server_model_state's own top comment on why that
	 * gate stayed global, not per-slot) and `slot` (this request's own
	 * resolved resident slot, for decode_inflight/model_state/drain_cv)
	 * are now two separate pointers, where PR E1 only ever needed one. */
	s_membrane_server_model_state	*st = NULL;
	s_model_slot					*slot = NULL;
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
		state->req_state.store(membrane_request_state_t::FAILED,
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
	 * acquire_model_slot() drain-wait blocked on a model switch. Section
	 * 27's original "back to READY" guarantee, preserved -- just no
	 * longer via a global server lock (removed; decode never held the
	 * slot's own mtx either). */
	decode_slot_exit(state->slot);
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

	/* Mega Phase E, PR E2: the chat-template cache check now peeks
	 * residency_mtx + (briefly) the candidate slot's own mtx, mirroring
	 * PR E1's own single-slot check -- see acquire_model_slot()'s own
	 * top comment for why this peek is always SAFE to be wrong (a miss
	 * just costs one extra real load_chat_template() call, never a
	 * correctness issue; the real, authoritative load happens inside
	 * acquire_model_slot() below regardless). */
	std::string	tmpl;
	std::string	tmpl_err;

	{
		s_model_slot	*peek = NULL;

		{
			std::lock_guard<std::mutex>	rlock(st->residency_mtx);

			for (auto &s : st->slots)
			{
				if (s.claimed_name == entry->name)
				{
					peek = &s;
					break ;
				}
			}
		}
		if (peek != NULL)
		{
			std::lock_guard<std::mutex>	slock(peek->mtx);

			if (peek->model_loaded && peek->loaded_name == entry->name
				&& !peek->cached_chat_template.empty())
				tmpl = peek->cached_chat_template;
		}
	}
	if (tmpl.empty() && !load_chat_template(entry->path, &tmpl, &tmpl_err))
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
	std::string						err_code;
	std::string						err_message;
	int								err_status = 500;
	std::unique_lock<std::mutex>	lock;
	s_model_slot					*slot = acquire_model_slot(st, &lock, reg,
			*entry, prompt_text, max_tokens, &err_code, &err_message,
			&err_status);

	if (slot == NULL || !err_code.empty())
	{
		send_json_error(res, err_status, err_code, err_message);
		return ;
	}
	slot->cached_chat_template = tmpl;

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
		state->slot = slot;
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
		state->local_session = slot->session;	/* PR E1: real value copy,
									 * taken while `lock` is still held --
									 * see s_stream_request_state's own
									 * local_session comment. */
		state->admission_ticket = std::move(admission_ticket);	/* Section
									 * 29: the slot stays held for the
									 * whole async stream, released by
									 * stream_release() -- not by this
									 * function's own (already-passed)
									 * return. */
		decode_slot_enter(slot);	/* PR E1: still holding `lock` here --
									 * marks this request as using this
									 * slot's model before the lock (and
									 * therefore the guarantee nothing else
									 * can close it out from under us) is
									 * released, right below. */
		lock.unlock();	/* PR E1: decode itself never holds the slot's
									 * own mtx anymore -- released before
									 * the worker thread is even spawned,
									 * so a second request's own acquire_
									 * model_slot() call can proceed
									 * immediately if it names the same
									 * (already-loaded) model, or a
									 * DIFFERENT already-resident one in
									 * another slot entirely (PR E2). */
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
	 * local_session comment for why a copy (never &slot->session
	 * directly) is required once decode can run concurrently with
	 * another request's own decode. */
	membrane_model_session_t	local_session = slot->session;

	decode_slot_enter(slot);
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
		 * Scoped so the decode ticket is released the moment generate()
		 * returns, before decode_slot_exit()'s own (separate) resident-
		 * slot mtx acquisition below -- no functional requirement either
		 * order would break, just avoids holding it any longer than
		 * needed. */
		decode_concurrency_ticket_t	decode_ticket(&st->decode_gate);

		req_state = membrane_request_state_t::ACTIVE;
		membrane_session_generate(&local_session, gen_req, &gen_res);
	}
	decode_slot_exit(slot);
	/* A stop-sequence match sets gen_req.cancel_flag, so gen_res.cancelled
	 * is true here too (ok is unaffected -- see runtime_session.h's own
	 * doc comment) -- never treated as a real failure: a stop match is a
	 * normal, successful completion (DONE), never CANCELLED, same
	 * distinction stream_worker_fn's own terminal classification makes. */
	if (gen_res.cancelled && !nonstream_stop_ctx.matched)
		req_state = membrane_request_state_t::CANCELLED;
	else if (!gen_res.ok)
		req_state = membrane_request_state_t::FAILED;
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
	 * instance situation this project never wants (Section 10/23: bounded
	 * residency -- at most membrane_max_resident_models() slots -- is a
	 * promise about ONE process's own state, not something a caller can
	 * accidentally defeat by starting a second instance). Confirmed
	 * directly: a second `membrane serve` on an
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
	svr.Post("/membrane/v1/models/pin", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_pin_model(&state, true, rq, rs); });
	svr.Post("/membrane/v1/models/unpin", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_pin_model(&state, false, rq, rs); });
	svr.Get("/membrane/v1/capabilities", [&](const httplib::Request &rq,
			httplib::Response &rs) { handle_capabilities(&state, rq, rs); });

	if (!svr.bind_to_port(bind, port))
	{
		fprintf(stderr, "membrane serve: could not bind %s:%d (port "
			"already in use?)\n", bind.c_str(), port);
		for (auto &slot : state.slots)
			if (slot.model_loaded)
				membrane_model_close(&slot.session);
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
		listener.join();	/* Mega Phase E, PR E4, Section 6/29 of the
									 * task ("server shutdown while active"):
									 * this join is what makes the model-
									 * close loop below safe against a
									 * request handler still actively
									 * generating. Verified by reading
									 * vendored cpp-httplib's own Server::
									 * listen_internal() (third_party/
									 * llama.cpp/vendor/cpp-httplib/
									 * httplib.cpp): the thread `listener`
									 * runs does not itself return until it
									 * has called task_queue->shutdown(),
									 * which JOINS every worker-pool thread
									 * first -- including one currently
									 * inside handle_chat_completions()'s
									 * non-streaming branch, or one running
									 * the streaming path's own stream_
									 * release() (which itself joins that
									 * request's dedicated generation-
									 * worker thread before returning). So
									 * by the time this call returns, no
									 * handler can still hold/use any
									 * slot's session -- confirmed
									 * empirically too (a real SIGTERM sent
									 * mid-generation; the process stayed
									 * parked here, in futex_do_wait, for
									 * exactly as long as that real
									 * generation took, then exited
									 * cleanly). Real, disclosed asymmetry:
									 * the non-streaming path has no
									 * cancellation signal wired to
									 * shutdown at all (unlike streaming's
									 * own stream_release()), so this wait
									 * is bounded by that generation's own
									 * max_tokens, never by anything this
									 * function does -- see docs/soak-and-
									 * concurrency-testing.md's own PR E4
									 * section. */
	for (auto &slot : state.slots)
		if (slot.model_loaded)
			membrane_model_close(&slot.session);
	membrane_runtime_shutdown(&state.rt);
	printf("MEMBRANE server stopped\n");
	return (0);
}
