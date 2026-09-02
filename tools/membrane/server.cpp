#include "server.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
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

using json = nlohmann::json;

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

/* Loads `name` (registry entry) as the server's one active model,
 * unloading whatever was loaded before (Section 23: one active model).
 * On the FIRST load of this model, runs the real context-recommendation
 * pipeline against `first_prompt` to decide gpu_layers/KV precision/ctx
 * (Section 21) -- this is a REAL, disclosed limitation (documented in
 * docs/server.md): that initial plan is sized for whichever request
 * happens to trigger the load, not a hypothetical future largest prompt;
 * gpu_layers/KV precision cannot change again without a reload. */
static bool	ensure_model_loaded(s_membrane_server_model_state *st,
				const membrane_registry_entry_t &entry,
				const std::string &first_prompt, int gen_tokens,
				std::string *err_code, std::string *err_message,
				int *http_status)
{
	if (st->model_loaded && st->loaded_name == entry.name)
		return (true);
	if (st->model_loaded)
	{
		membrane_model_close(&st->session);
		st->model_loaded = false;
		st->loaded_name.clear();
		st->cached_chat_template.clear();
	}
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

static void	handle_health(const httplib::Request &, httplib::Response &res)
{
	json	j;

	j["status"] = "ok";
	j["version"] = MEMBRANE_VERSION;
	res.set_content(j.dump(), "application/json");
}

static void	handle_models(s_membrane_server_model_state *st,
				const membrane_registry_t &reg, const httplib::Request &,
				httplib::Response &res)
{
	(void)st;
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

static void	handle_chat_completions(s_membrane_server_model_state *st,
				const membrane_registry_t &reg, const httplib::Request &req,
				httplib::Response &res)
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
		|| !body["model"].is_string() || !body.contains("messages")
		|| !body["messages"].is_array() || body["messages"].empty())
	{
		send_json_error(res, 400, "INVALID_REQUEST", "request must be a "
			"JSON object with a string \"model\" and a non-empty "
			"\"messages\" array");
		return ;
	}
	if (body.contains("stream") && body["stream"].is_boolean()
		&& body["stream"].get<bool>() == true)
	{
		send_json_error(res, 400, "STREAMING_NOT_SUPPORTED", "stream=true "
			"is not currently supported -- omit \"stream\" or set it to "
			"false");
		return ;
	}
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
	std::string	model_name = body["model"];
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

	std::lock_guard<std::mutex>	lock(st->mtx);
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

	if (!ensure_model_loaded(st, *entry, prompt_text, max_tokens, &err_code,
			&err_message, &err_status))
	{
		send_json_error(res, err_status, err_code, err_message);
		return ;
	}
	st->cached_chat_template = tmpl;

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
	membrane_session_generate(&st->session, gen_req, &gen_res);
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
	char	id_buf[64];

	snprintf(id_buf, sizeof(id_buf), "chatcmpl-%llx",
		(unsigned long long)time(NULL) ^ (unsigned long long)(size_t)&res);
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
	std::string	registry_path = opts.registry_path.empty()
			? membrane_registry_resolve_path() : opts.registry_path;
	membrane_registry_t		reg;
	membrane_registry_error_t	reg_err;

	if (registry_path.empty())
	{
		fprintf(stderr, "membrane serve: neither XDG_DATA_HOME nor HOME "
			"is set -- cannot locate the model registry\n");
		return (1);
	}
	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		fprintf(stderr, "membrane serve: could not load the model "
			"registry: %s\n", reg_err.message.c_str());
		return (1);
	}
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

	s_membrane_server_model_state	state;

	membrane_runtime_init(&state.rt);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	httplib::Server	svr;

	svr.Get("/health", handle_health);
	svr.Get("/v1/models", [&](const httplib::Request &rq,
			httplib::Response &rs) { handle_models(&state, reg, rq, rs); });
	svr.Post("/v1/chat/completions", [&](const httplib::Request &rq,
			httplib::Response &rs)
		{ handle_chat_completions(&state, reg, rq, rs); });

	int	port = opts.port;

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
