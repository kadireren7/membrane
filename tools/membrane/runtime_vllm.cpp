#include "runtime_vllm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "runtime_ollama.h"

using json = nlohmann::json;

/* See runtime_vllm.h's own top comment for the full contract. */

#define VLLM_SMALL_MAX_BYTES	(64u * 1024u)
#define VLLM_MODELS_MAX_BYTES	(4u * 1024u * 1024u)

static void	copy_cstr(char *dst, size_t n, const std::string &s)
{
	if (n == 0)
		return ;
	snprintf(dst, n, "%s", s.c_str());
}

/* ---------------------------------------------------------------- */
/* Capability contract (STATIC_CONTRACT, audited against v0.31.0)    */
/* ---------------------------------------------------------------- */

void	membrane_vllm_capabilities(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));
	/* MODEL / LIFECYCLE. GET /v1/models lists the model(s) the process
	 * serves. The model is chosen when `vllm serve` starts: no load,
	 * unload or switch API exists. Metadata: only max_model_len. */
	c->model_enumeration = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_load_unload = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->model_switch = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->model_metadata = MEMBRANE_CAPABILITY_PARTIAL;
	/* PLANNING / CONTROL. Context (--max-model-len), KV dtype (--kv-cache-
	 * dtype), memory fraction (--gpu-memory-utilization), parallelism
	 * (--tensor-parallel-size) are launch flags, not an API. */
	c->context_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->quant_variant_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->concurrency_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->device_selection = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->memory_headroom_telemetry = MEMBRANE_CAPABILITY_UNSUPPORTED;
	/* INFERENCE. OpenAI-compatible endpoints are documented; MEMBRANE
	 * never calls them from this adapter. */
	c->chat_completions = MEMBRANE_CAPABILITY_SUPPORTED;
	c->streaming = MEMBRANE_CAPABILITY_SUPPORTED;
	c->cancellation = MEMBRANE_CAPABILITY_UNKNOWN;
	/* OBSERVABILITY. current model: yes. Active context: only the
	 * configured maximum (PARTIAL). GET /load gives in-flight request
	 * count only. RAM/VRAM/KV usage exist solely on the Prometheus
	 * /metrics surface, which MEMBRANE does not read. */
	c->current_model = MEMBRANE_CAPABILITY_SUPPORTED;
	c->active_context = MEMBRANE_CAPABILITY_PARTIAL;
	c->ram_usage = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->vram_usage = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_cache_usage = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->loaded_model_memory = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->live_kv_migration = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->dynamic_reconfiguration = MEMBRANE_CAPABILITY_UNSUPPORTED;
}

bool	membrane_vllm_request_allowed(const std::string &method,
			const std::string &path)
{
	return (method == "GET" && (path == "/health" || path == "/version"
		|| path == "/v1/models" || path == "/load"));
}

/* ---------------------------------------------------------------- */
/* Pure parsers                                                      */
/* ---------------------------------------------------------------- */

bool	membrane_vllm_parse_version(const std::string &body,
			std::string *version)
{
	json	j = json::parse(body, nullptr, false);

	if (j.is_discarded() || !j.is_object() || !j.contains("version")
		|| !j["version"].is_string())
		return (false);
	*version = j["version"].get<std::string>();
	return (!version->empty() && version->size() < MEMBRANE_RUNTIME_VERSION_MAX);
}

bool	membrane_vllm_parse_load(const std::string &body, long long *load)
{
	json	j = json::parse(body, nullptr, false);

	if (j.is_discarded() || !j.is_object() || !j.contains("server_load")
		|| !j["server_load"].is_number_integer())
		return (false);
	*load = j["server_load"].get<long long>();
	return (*load >= 0);
}

bool	membrane_vllm_parse_models(const std::string &body,
			std::vector<membrane_external_model_t> *out, std::string *err)
{
	json	j = json::parse(body, nullptr, false);

	out->clear();
	if (j.is_discarded() || !j.is_object() || !j.contains("data")
		|| !j["data"].is_array())
	{
		*err = "expected {\"data\": [...]}";
		return (false);
	}
	for (const auto &e : j["data"])
	{
		membrane_external_model_t	m;

		if (!e.is_object() || !e.contains("id") || !e["id"].is_string()
			|| e["id"].get<std::string>().empty())
		{
			*err = "a model card has no string id";
			return (false);
		}
		membrane_external_model_init(&m);
		m.runtime_id = MEMBRANE_RUNTIME_ID_VLLM;
		m.runtime_model_id = e["id"].get<std::string>();
		m.display_name = m.runtime_model_id;
		if (e.contains("root") && e["root"].is_string())
			m.family = "";			/* root is a path/id, not a family */
		/* max_model_len: the served model's CONFIGURED maximum context */
		if (e.contains("max_model_len") && e["max_model_len"]
			.is_number_unsigned())
		{
			m.context_length_known = true;
			m.context_length = e["max_model_len"].get<uint64_t>();
		}
		m.provenance = MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE;
		out->push_back(m);
	}
	return (true);
}

/* ---------------------------------------------------------------- */
/* The one HTTP helper: allowlisted, bounded, GET only               */
/* ---------------------------------------------------------------- */

typedef enum e_vllm_http_kind
{
	VLLM_HTTP_RESPONSE = 0,
	VLLM_HTTP_TRANSPORT_FAILED,
	VLLM_HTTP_TOO_LARGE,
	VLLM_HTTP_REFUSED
}	vllm_http_kind_t;

typedef struct s_vllm_http_result
{
	vllm_http_kind_t	kind;
	int					status;
	std::string			body;
}	vllm_http_result_t;

static membrane_ollama_endpoint_t	resolve_endpoint(void)
{
	const char	*env = getenv(MEMBRANE_VLLM_ENDPOINT_ENV);

	/* The generic http://HOST[:PORT] parser is shared with the Ollama
	 * adapter (no credentials, no https, no paths). Only the default
	 * differs; the parser's own default-port rule applies to Ollama, so
	 * the vLLM default is always spelled out in full. */
	return (membrane_ollama_parse_endpoint(env != NULL && env[0] != '\0'
		? env : MEMBRANE_VLLM_DEFAULT_ENDPOINT));
}

static vllm_http_result_t	vllm_http(const membrane_ollama_endpoint_t &ep,
								const std::string &path, size_t cap,
								time_t read_timeout_s)
{
	vllm_http_result_t		r;
	bool					too_large = false;

	r.kind = VLLM_HTTP_REFUSED;
	r.status = 0;
	if (!ep.ok || !membrane_vllm_request_allowed("GET", path))
		return (r);
	httplib::Client			cli(ep.host, ep.port);
	httplib::ContentReceiver	recv = [&](const char *data, size_t len)
	{
		if (r.body.size() + len > cap)
		{
			too_large = true;
			return (false);
		}
		r.body.append(data, len);
		return (true);
	};

	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(read_timeout_s, 0);
	cli.set_write_timeout(2, 0);
	cli.set_max_timeout((read_timeout_s + 1) * 1000);
	cli.set_follow_location(false);
	cli.set_keep_alive(false);
	httplib::Result	res = cli.Get(path, httplib::Headers(), recv);

	if (too_large)
	{
		r.kind = VLLM_HTTP_TOO_LARGE;
		r.body.clear();
		return (r);
	}
	if (!res)
	{
		r.kind = VLLM_HTTP_TRANSPORT_FAILED;
		r.body.clear();
		return (r);
	}
	r.kind = VLLM_HTTP_RESPONSE;
	r.status = res->status;
	return (r);
}

static void	set_err(membrane_runtime_error_t *err, const char *code,
				const std::string &message)
{
	err->code = code;
	err->message = message;
}

/* 0 = ok to parse, else error set. */
static bool	classify(const membrane_ollama_endpoint_t &ep,
				const vllm_http_result_t &r, const char *what,
				membrane_runtime_error_t *err)
{
	if (r.kind == VLLM_HTTP_TRANSPORT_FAILED || r.kind == VLLM_HTTP_REFUSED)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_UNAVAILABLE,
			"vLLM API is not reachable at " + ep.url);
		return (false);
	}
	if (r.kind == VLLM_HTTP_TOO_LARGE)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_TOO_LARGE, std::string("vLLM's ")
			+ what + " response exceeded MEMBRANE's size limit");
		return (false);
	}
	if (r.status < 200 || r.status > 299)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_HTTP, std::string("vLLM answered ")
			+ what + " with HTTP " + std::to_string(r.status));
		return (false);
	}
	return (true);
}

void	membrane_vllm_describe(membrane_runtime_descriptor_t *out)
{
	membrane_ollama_endpoint_t	ep = resolve_endpoint();
	vllm_http_result_t			r;
	std::string					version;

	memset(out, 0, sizeof(*out));
	copy_cstr(out->id, sizeof(out->id), MEMBRANE_RUNTIME_ID_VLLM);
	copy_cstr(out->display_name, sizeof(out->display_name), "vLLM");
	out->type = MEMBRANE_RUNTIME_TYPE_EXTERNAL;
	out->execution_mode = MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL;
	out->capability_provenance = MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT;
	membrane_vllm_capabilities(&out->capabilities);
	if (!ep.ok)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_NOT_PROBED;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			std::string(MEMBRANE_VLLM_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return ;
	}
	out->execution_mode = ep.loopback ? MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL
		: MEMBRANE_RUNTIME_EXEC_REMOTE_EXTERNAL;
	out->endpoint_known = 1;
	copy_cstr(out->endpoint, sizeof(out->endpoint), ep.url);
	r = vllm_http(ep, "/version", VLLM_SMALL_MAX_BYTES, 2);
	if (r.kind == VLLM_HTTP_TRANSPORT_FAILED || r.kind == VLLM_HTTP_REFUSED)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_UNREACHABLE;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"vLLM API is not reachable.");
		return ;
	}
	if (r.kind == VLLM_HTTP_TOO_LARGE)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN;
		out->health = MEMBRANE_RUNTIME_HEALTH_UNKNOWN;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"the version response exceeded MEMBRANE's size limit.");
		return ;
	}
	if (r.status < 200 || r.status > 299)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_INCOMPATIBLE;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"endpoint answered GET /version with HTTP "
			+ std::to_string(r.status) + "; not a compatible vLLM API.");
		return ;
	}
	if (!membrane_vllm_parse_version(r.body, &version))
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN;
		out->health = MEMBRANE_RUNTIME_HEALTH_UNKNOWN;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"GET /version answered, but not with a recognizable vLLM "
			"version document.");
		return ;
	}
	copy_cstr(out->version, sizeof(out->version), version);
	out->version_known = 1;
	out->version_provenance = MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE;
	/* /version answering says the HTTP server is up; /health says whether
	 * the ENGINE is alive (503 = engine dead). Both must be good. */
	r = vllm_http(ep, "/health", VLLM_SMALL_MAX_BYTES, 2);
	if (r.kind == VLLM_HTTP_RESPONSE && r.status == 503)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_INCOMPATIBLE;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"vLLM reports its engine is not healthy (GET /health = 503).");
		return ;
	}
	out->availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	out->health = MEMBRANE_RUNTIME_HEALTH_HEALTHY;
}

bool	membrane_vllm_list_models(std::vector<membrane_external_model_t> *out,
			membrane_runtime_error_t *err)
{
	membrane_ollama_endpoint_t	ep = resolve_endpoint();
	vllm_http_result_t			r;
	std::string					perr;

	if (!ep.ok)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_INVALID_ENDPOINT,
			std::string(MEMBRANE_VLLM_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return (false);
	}
	r = vllm_http(ep, "/v1/models", VLLM_MODELS_MAX_BYTES, 5);
	if (!classify(ep, r, "GET /v1/models", err))
		return (false);
	if (!membrane_vllm_parse_models(r.body, out, &perr))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_MALFORMED,
			"vLLM returned a malformed model list: " + perr);
		return (false);
	}
	return (true);
}

bool	membrane_vllm_inspect_model(const std::string &model,
			membrane_external_model_detail_t *out,
			membrane_runtime_error_t *err)
{
	std::vector<membrane_external_model_t>	models;

	if (!membrane_vllm_list_models(&models, err))
		return (false);
	for (const auto &m : models)
		if (m.runtime_model_id == model)
		{
			out->model = m;
			return (true);
		}
	set_err(err, MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND,
		"vLLM is not serving a model named '" + model + "'");
	return (false);
}

bool	membrane_vllm_server_load(long long *load,
			membrane_runtime_error_t *err)
{
	membrane_ollama_endpoint_t	ep = resolve_endpoint();
	vllm_http_result_t			r;

	if (!ep.ok)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_INVALID_ENDPOINT,
			std::string(MEMBRANE_VLLM_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return (false);
	}
	r = vllm_http(ep, "/load", VLLM_SMALL_MAX_BYTES, 3);
	if (!classify(ep, r, "GET /load", err))
		return (false);
	if (!membrane_vllm_parse_load(r.body, load))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_MALFORMED,
			"vLLM returned a malformed /load document");
		return (false);
	}
	return (true);
}
