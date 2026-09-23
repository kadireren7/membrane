#include "runtime_ollama.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/* See runtime_ollama.h's own top comment for the full contract. */

/* Bounded response sizes -- a response larger than this is reported as
 * RESPONSE_TOO_LARGE, never buffered further. /api/tags lists one small
 * object per installed model; /api/show without "verbose" omits the
 * tokenizer arrays but may still carry a license/template of some KiB. */
#define OLLAMA_VERSION_MAX_BYTES	(64u * 1024u)
#define OLLAMA_TAGS_MAX_BYTES		(16u * 1024u * 1024u)
#define OLLAMA_SHOW_MAX_BYTES		(8u * 1024u * 1024u)
#define OLLAMA_MODEL_NAME_MAX		256
#define OLLAMA_PARAMETERS_MAX		1024
#define OLLAMA_ERROR_TEXT_MAX		200

static std::string	lower(const std::string &s)
{
	std::string	out(s);

	for (char &c : out)
		c = (char)std::tolower((unsigned char)c);
	return (out);
}

static std::string	trim(const std::string &s)
{
	size_t	b;
	size_t	e;

	b = 0;
	while (b < s.size() && std::isspace((unsigned char)s[b]))
		b++;
	e = s.size();
	while (e > b && std::isspace((unsigned char)s[e - 1]))
		e--;
	return (s.substr(b, e - b));
}

static bool	all_digits(const std::string &s)
{
	if (s.empty())
		return (false);
	for (char c : s)
		if (!std::isdigit((unsigned char)c))
			return (false);
	return (true);
}

static bool	is_loopback_host(const std::string &host)
{
	std::string	h = lower(host);

	if (h == "localhost" || h == "::1")
		return (true);
	if (h.compare(0, 4, "127.") != 0)
		return (false);
	for (char c : h)
		if (!std::isdigit((unsigned char)c) && c != '.')
			return (false);
	return (true);
}

static membrane_ollama_endpoint_t	endpoint_error(const std::string &msg)
{
	membrane_ollama_endpoint_t	ep;

	ep.ok = false;
	ep.port = 0;
	ep.loopback = false;
	ep.from_env = false;
	ep.error = msg;
	return (ep);
}

membrane_ollama_endpoint_t	membrane_ollama_parse_endpoint(
								const std::string &raw)
{
	membrane_ollama_endpoint_t	ep;
	std::string					s = trim(raw);
	std::string					rest;
	std::string					port_str;
	bool						ipv6 = false;

	if (s.size() > 100)
		return (endpoint_error("endpoint is too long"));
	if (lower(s).compare(0, 8, "https://") == 0)
		return (endpoint_error("https is not supported (this build has no "
			"TLS); use a local http:// endpoint"));
	if (lower(s).compare(0, 7, "http://") != 0)
		return (endpoint_error("endpoint must start with http://"));
	rest = s.substr(7);
	if (!rest.empty() && rest[rest.size() - 1] == '/')
		rest.erase(rest.size() - 1);
	if (rest.find('@') != std::string::npos)
		return (endpoint_error("credentials in the endpoint URL are not "
			"supported"));
	if (rest.find_first_of("/?#") != std::string::npos)
		return (endpoint_error("endpoint must be http://HOST[:PORT] with "
			"no path or query"));
	if (rest.empty())
		return (endpoint_error("endpoint has no host"));
	ep.ok = true;
	ep.port = MEMBRANE_OLLAMA_DEFAULT_PORT;
	ep.from_env = false;
	if (rest[0] == '[')
	{
		size_t	close = rest.find(']');

		if (close == std::string::npos || close == 1)
			return (endpoint_error("malformed bracketed IPv6 host"));
		ep.host = rest.substr(1, close - 1);
		for (char c : ep.host)
			if (!std::isxdigit((unsigned char)c) && c != ':' && c != '.')
				return (endpoint_error("malformed bracketed IPv6 host"));
		rest = rest.substr(close + 1);
		if (!rest.empty() && rest[0] != ':')
			return (endpoint_error("unexpected text after IPv6 host"));
		if (!rest.empty())
			port_str = rest.substr(1);
		ipv6 = true;
	}
	else
	{
		size_t	colon = rest.find(':');

		if (colon != std::string::npos
			&& rest.find(':', colon + 1) != std::string::npos)
			return (endpoint_error("an IPv6 host must be in brackets, "
				"e.g. http://[::1]:11434"));
		ep.host = rest.substr(0, colon);
		if (colon != std::string::npos)
			port_str = rest.substr(colon + 1);
		for (char c : ep.host)
			if (!std::isalnum((unsigned char)c) && c != '.' && c != '-')
				return (endpoint_error("invalid character in host"));
	}
	if (ep.host.empty())
		return (endpoint_error("endpoint has no host"));
	if (!port_str.empty() || (rest.size() > 0 && rest[rest.size() - 1] == ':'))
	{
		if (!all_digits(port_str) || port_str.size() > 5
			|| std::atoi(port_str.c_str()) < 1
			|| std::atoi(port_str.c_str()) > 65535)
			return (endpoint_error("invalid port"));
		ep.port = std::atoi(port_str.c_str());
	}
	ep.loopback = is_loopback_host(ep.host);
	ep.url = std::string("http://") + (ipv6 ? "[" + ep.host + "]" : ep.host)
		+ ":" + std::to_string(ep.port);
	return (ep);
}

membrane_ollama_endpoint_t	membrane_ollama_resolve_endpoint(void)
{
	const char					*env = getenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
	membrane_ollama_endpoint_t	ep;

	if (env == NULL || trim(env).empty())
		return (membrane_ollama_parse_endpoint(
			MEMBRANE_OLLAMA_DEFAULT_ENDPOINT));
	ep = membrane_ollama_parse_endpoint(env);
	ep.from_env = true;
	return (ep);
}

/*
 * The Ollama capability contract. Every value answers ONE question:
 * "can MEMBRANE explicitly observe/control this through Ollama's
 * DOCUMENTED local HTTP API" -- never "does Ollama do this internally".
 * A server-side environment variable read once when `ollama serve`
 * starts (OLLAMA_KV_CACHE_TYPE, OLLAMA_NUM_PARALLEL, CUDA_VISIBLE_DEVICES,
 * ...) is NOT an API control: MEMBRANE cannot set it on a daemon it does
 * not own. Audited against ollama/ollama v0.34.3 (docs/api.md, docs/
 * openapi.yaml, api/types.go, server/routes.go); the per-field rationale
 * is tabulated in docs/runtime-ollama.md. STATIC_CONTRACT provenance:
 * nothing here is probed from the running daemon.
 *
 * Note H2 itself only exercises model_enumeration and model_metadata;
 * every other SUPPORTED/PARTIAL value describes the documented API an
 * eventual adapter could use, not something H2 calls.
 */
void	membrane_ollama_capabilities(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));

	/* MODEL / LIFECYCLE. Enumeration: GET /api/tags. Metadata: POST
	 * /api/show. Switch: every inference request names its model, and
	 * the scheduler loads it on demand. Load/unload: PARTIAL -- documented
	 * only as side effects of /api/generate (empty prompt loads; keep_alive
	 * 0 unloads), and Ollama's own scheduler still evicts independently
	 * (OLLAMA_MAX_LOADED_MODELS, memory pressure). */
	c->model_enumeration = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_load_unload = MEMBRANE_CAPABILITY_PARTIAL;
	c->model_switch = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_metadata = MEMBRANE_CAPABILITY_SUPPORTED;

	/* PLANNING / CONTROL. Context: the documented per-request
	 * options.num_ctx. GPU layers: PARTIAL -- options.num_gpu is accepted
	 * per request (api/types.go, docs/api.md example) but its semantics
	 * are not specified by the OpenAPI document and Ollama's own memory
	 * fitting decides the final offload. Quant/variant: PARTIAL -- one can
	 * only choose among variants ALREADY pulled (each tag is one quant);
	 * obtaining another variant is a pull, i.e. a mutation. KV precision
	 * (OLLAMA_KV_CACHE_TYPE), concurrency (OLLAMA_NUM_PARALLEL): server
	 * start-time env only -> UNSUPPORTED through the API. KV placement:
	 * no control at all. Device selection: options.main_gpu appears only
	 * in a docs example, device visibility is start-time env -> UNKNOWN.
	 * Headroom telemetry: no free-memory endpoint exists. */
	c->context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->quant_variant_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->concurrency_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->device_selection = MEMBRANE_CAPABILITY_UNKNOWN;
	c->memory_headroom_telemetry = MEMBRANE_CAPABILITY_UNSUPPORTED;

	/* INFERENCE. /api/chat (and /v1/chat/completions) with NDJSON/SSE
	 * streaming are documented -- H2 NEVER calls them. Cancellation: no
	 * documented cancel endpoint or semantics (closing the connection is
	 * not documented as cancelling) -> UNKNOWN. */
	c->chat_completions = MEMBRANE_CAPABILITY_SUPPORTED;
	c->streaming = MEMBRANE_CAPABILITY_SUPPORTED;
	c->cancellation = MEMBRANE_CAPABILITY_UNKNOWN;

	/* OBSERVABILITY. GET /api/ps lists loaded models with context_length,
	 * size and size_vram (not called by H2). RAM/VRAM/loaded-model memory:
	 * PARTIAL -- per-loaded-model totals from Ollama's scheduler, no
	 * host/device-wide view and no per-device breakdown. KV-cache usage:
	 * not exposed anywhere. */
	c->current_model = MEMBRANE_CAPABILITY_SUPPORTED;
	c->active_context = MEMBRANE_CAPABILITY_SUPPORTED;
	c->ram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->vram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->kv_cache_usage = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->loaded_model_memory = MEMBRANE_CAPABILITY_PARTIAL;

	/* ADVANCED. No KV export/import API; changing num_ctx reloads the
	 * model rather than reconfiguring it live. */
	c->live_kv_migration = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->dynamic_reconfiguration = MEMBRANE_CAPABILITY_UNSUPPORTED;
}

bool	membrane_ollama_request_allowed(const std::string &method,
			const std::string &path)
{
	if (method == "GET")
		return (path == "/api/version" || path == "/api/tags");
	if (method == "POST")
		return (path == "/api/show");
	return (false);
}

bool	membrane_ollama_is_cloud_model_ref(const std::string &name)
{
	std::string	raw = trim(name);
	size_t		idx = raw.rfind(':');
	std::string	suffix_raw;
	std::string	suffix;

	if (idx == std::string::npos)
		return (false);
	suffix_raw = trim(raw.substr(idx + 1));
	suffix = lower(suffix_raw);
	if (suffix == "cloud")
		return (true);
	if (suffix_raw.find('/') == std::string::npos && suffix.size() >= 6
		&& suffix.compare(suffix.size() - 6, 6, "-cloud") == 0)
		return (true);
	return (false);
}

/* ---------------------------------------------------------------- */
/* Pure parsers                                                     */
/* ---------------------------------------------------------------- */

static bool	valid_version_string(const std::string &v)
{
	if (v.empty() || v.size() >= MEMBRANE_RUNTIME_VERSION_MAX)
		return (false);
	for (char c : v)
		if (!std::isalnum((unsigned char)c) && c != '.' && c != '-'
			&& c != '+' && c != '_')
			return (false);
	return (true);
}

bool	membrane_ollama_parse_version(const std::string &body,
			std::string *version)
{
	json	j = json::parse(body, nullptr, false);

	if (j.is_discarded() || !j.is_object() || !j.contains("version")
		|| !j["version"].is_string())
		return (false);
	if (!valid_version_string(j["version"].get<std::string>()))
		return (false);
	*version = j["version"].get<std::string>();
	return (true);
}

static std::string	str_field(const json &o, const char *key)
{
	if (o.is_object() && o.contains(key) && o[key].is_string())
		return (o[key].get<std::string>());
	return (std::string());
}

static bool	uint_field(const json &o, const char *key, uint64_t *out)
{
	if (!o.is_object() || !o.contains(key))
		return (false);
	if (o[key].is_number_unsigned())
	{
		*out = o[key].get<uint64_t>();
		return (true);
	}
	if (o[key].is_number_integer() && o[key].get<int64_t>() >= 0)
	{
		*out = (uint64_t)o[key].get<int64_t>();
		return (true);
	}
	return (false);
}

static std::vector<std::string>	str_array_field(const json &o,
									const char *key)
{
	std::vector<std::string>	out;

	if (!o.is_object() || !o.contains(key) || !o[key].is_array())
		return (out);
	for (const auto &v : o[key])
		if (v.is_string())
			out.push_back(v.get<std::string>());
	return (out);
}

/* Fills the fields /api/tags and /api/show share (the ModelDetails
 * object plus capabilities/remote/modified_at). */
static void	fill_common(const json &o, membrane_external_model_t *m)
{
	const json	empty = json::object();
	const json	&d = (o.contains("details") && o["details"].is_object())
		? o["details"] : empty;
	uint64_t	n;

	m->family = str_field(d, "family");
	m->families = str_array_field(d, "families");
	m->format = str_field(d, "format");
	m->parameter_size = str_field(d, "parameter_size");
	m->quantization = str_field(d, "quantization_level");
	if (uint_field(d, "context_length", &n) && n > 0)
	{
		m->context_length_known = true;
		m->context_length = n;
	}
	m->modified_at = str_field(o, "modified_at");
	m->remote_host = str_field(o, "remote_host");
	m->remote = !m->remote_host.empty();
	m->runtime_capabilities = str_array_field(o, "capabilities");
	m->runtime_id = MEMBRANE_RUNTIME_ID_OLLAMA;
	m->provenance = MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE;
}

bool	membrane_ollama_parse_tags(const std::string &body,
			std::vector<membrane_external_model_t> *out, std::string *err)
{
	json								j = json::parse(body, nullptr, false);
	std::vector<membrane_external_model_t>	models;
	size_t								i;

	if (j.is_discarded())
	{
		*err = "response is not valid JSON";
		return (false);
	}
	if (!j.is_object() || !j.contains("models") || !j["models"].is_array())
	{
		*err = "response has no \"models\" array";
		return (false);
	}
	i = 0;
	for (const auto &o : j["models"])
	{
		membrane_external_model_t	m;
		uint64_t					n;

		membrane_external_model_init(&m);
		if (!o.is_object())
		{
			*err = "models[" + std::to_string(i) + "] is not an object";
			return (false);
		}
		m.display_name = str_field(o, "name");
		m.runtime_model_id = str_field(o, "model");
		if (m.runtime_model_id.empty())
			m.runtime_model_id = m.display_name;
		if (m.display_name.empty())
			m.display_name = m.runtime_model_id;
		if (m.runtime_model_id.empty())
		{
			*err = "models[" + std::to_string(i) + "] has no name";
			return (false);
		}
		if (uint_field(o, "size", &n))
		{
			m.size_known = true;
			m.size_bytes = n;
		}
		m.digest = str_field(o, "digest");
		fill_common(o, &m);
		models.push_back(m);
		i++;
	}
	*out = models;
	return (true);
}

bool	membrane_ollama_parse_show(const std::string &model,
			const std::string &body, membrane_external_model_detail_t *out,
			std::string *err)
{
	json								j = json::parse(body, nullptr, false);
	membrane_external_model_detail_t	d;
	uint64_t							n;

	membrane_external_model_detail_init(&d);
	if (j.is_discarded())
	{
		*err = "response is not valid JSON";
		return (false);
	}
	if (!j.is_object())
	{
		*err = "response is not a JSON object";
		return (false);
	}
	d.model.runtime_model_id = model;
	d.model.display_name = model;
	fill_common(j, &d.model);
	if (j.contains("model_info") && j["model_info"].is_object())
	{
		const json	&mi = j["model_info"];

		d.architecture = str_field(mi, "general.architecture");
		if (uint_field(mi, "general.parameter_count", &n) && n > 0)
		{
			d.parameter_count_known = true;
			d.parameter_count = n;
		}
		if (!d.model.context_length_known && !d.architecture.empty()
			&& uint_field(mi, (d.architecture + ".context_length").c_str(),
				&n) && n > 0)
		{
			d.model.context_length_known = true;
			d.model.context_length = n;
		}
	}
	d.parameters = str_field(j, "parameters");
	if (d.parameters.size() > OLLAMA_PARAMETERS_MAX)
	{
		d.parameters.resize(OLLAMA_PARAMETERS_MAX);
		d.parameters_truncated = true;
	}
	d.has_template = !str_field(j, "template").empty();
	d.has_system = !str_field(j, "system").empty();
	d.has_license = !str_field(j, "license").empty();
	*out = d;
	return (true);
}

/* ---------------------------------------------------------------- */
/* The ONE HTTP helper -- allowlisted, bounded, no retries          */
/* ---------------------------------------------------------------- */

typedef enum e_ollama_http_kind
{
	OLLAMA_HTTP_RESPONSE = 0,
	OLLAMA_HTTP_TRANSPORT_FAILED,
	OLLAMA_HTTP_TOO_LARGE,
	OLLAMA_HTTP_REFUSED
}	ollama_http_kind_t;

typedef struct s_ollama_http_result
{
	ollama_http_kind_t	kind;
	int					status;
	std::string			body;
}	ollama_http_result_t;

static ollama_http_result_t	ollama_http(const membrane_ollama_endpoint_t &ep,
								const std::string &method,
								const std::string &path,
								const std::string &req_body, size_t cap,
								time_t read_timeout_s)
{
	ollama_http_result_t	r;
	bool					too_large = false;

	r.kind = OLLAMA_HTTP_REFUSED;
	r.status = 0;
	if (!ep.ok || !membrane_ollama_request_allowed(method, path))
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

	/* Short, bounded, single attempt: no retry loop, no redirects (a
	 * redirect could otherwise lead off the configured endpoint), no
	 * keep-alive, no credentials, no proxy. */
	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(read_timeout_s, 0);
	cli.set_write_timeout(2, 0);
	cli.set_max_timeout((read_timeout_s + 1) * 1000);
	cli.set_follow_location(false);
	cli.set_keep_alive(false);

	httplib::Result	res = (method == "GET")
		? cli.Get(path, httplib::Headers(), recv)
		: cli.Post(path, httplib::Headers(), req_body, "application/json",
			recv);

	if (too_large)
	{
		r.kind = OLLAMA_HTTP_TOO_LARGE;
		r.body.clear();
		return (r);
	}
	if (!res)
	{
		r.kind = OLLAMA_HTTP_TRANSPORT_FAILED;
		r.body.clear();
		return (r);
	}
	r.kind = OLLAMA_HTTP_RESPONSE;
	r.status = res->status;
	return (r);
}

/* Ollama's documented error shape is {"error": "..."}; surfaced only as
 * a short, printable excerpt -- never a raw body dump. */
static std::string	ollama_error_text(const std::string &body)
{
	std::string	msg = str_field(json::parse(body, nullptr, false), "error");
	std::string	clean;

	for (char c : msg)
	{
		if (clean.size() >= OLLAMA_ERROR_TEXT_MAX)
			break ;
		clean.push_back(std::isprint((unsigned char)c) ? c : ' ');
	}
	return (clean);
}

static void	set_err(membrane_runtime_error_t *err, const char *code,
				const std::string &message)
{
	err->code = code;
	err->message = message;
}

static void	copy_cstr(char *dst, size_t dst_size, const std::string &src)
{
	size_t	len = src.size();

	if (dst_size == 0)
		return ;
	if (len >= dst_size)
		len = dst_size - 1;
	memcpy(dst, src.data(), len);
	dst[len] = '\0';
}

void	membrane_ollama_describe(membrane_runtime_descriptor_t *out)
{
	membrane_ollama_endpoint_t	ep = membrane_ollama_resolve_endpoint();
	ollama_http_result_t		r;
	std::string					version;

	memset(out, 0, sizeof(*out));
	copy_cstr(out->id, sizeof(out->id), MEMBRANE_RUNTIME_ID_OLLAMA);
	copy_cstr(out->display_name, sizeof(out->display_name), "Ollama");
	out->type = MEMBRANE_RUNTIME_TYPE_EXTERNAL;
	out->execution_mode = MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL;
	out->capability_provenance = MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT;
	membrane_ollama_capabilities(&out->capabilities);
	if (!ep.ok)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_NOT_PROBED;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			std::string(MEMBRANE_OLLAMA_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return ;
	}
	/* An explicitly configured non-loopback endpoint is reported as what
	 * it is, never silently as a local runtime. */
	out->execution_mode = ep.loopback ? MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL
		: MEMBRANE_RUNTIME_EXEC_REMOTE_EXTERNAL;
	out->endpoint_known = 1;
	copy_cstr(out->endpoint, sizeof(out->endpoint), ep.url);
	r = ollama_http(ep, "GET", "/api/version", "", OLLAMA_VERSION_MAX_BYTES,
			2);
	if (r.kind == OLLAMA_HTTP_TRANSPORT_FAILED
		|| r.kind == OLLAMA_HTTP_REFUSED)
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
		out->health = MEMBRANE_RUNTIME_HEALTH_UNREACHABLE;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"Ollama API is not reachable.");
		return ;
	}
	if (r.kind == OLLAMA_HTTP_TOO_LARGE)
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
			"endpoint answered GET /api/version with HTTP "
			+ std::to_string(r.status) + "; not a compatible Ollama API.");
		return ;
	}
	if (!membrane_ollama_parse_version(r.body, &version))
	{
		out->availability = MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN;
		out->health = MEMBRANE_RUNTIME_HEALTH_UNKNOWN;
		copy_cstr(out->unavailable_reason, sizeof(out->unavailable_reason),
			"endpoint answered, but its version response was malformed.");
		return ;
	}
	out->availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	out->health = MEMBRANE_RUNTIME_HEALTH_HEALTHY;
	out->version_known = 1;
	copy_cstr(out->version, sizeof(out->version), version);
	out->version_provenance = MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE;
}

/* Shared transport/HTTP-status classification for inventory/metadata. */
static bool	classify_http(const membrane_ollama_endpoint_t &ep,
				const ollama_http_result_t &r, const char *what,
				membrane_runtime_error_t *err)
{
	std::string	detail;

	if (r.kind == OLLAMA_HTTP_TRANSPORT_FAILED
		|| r.kind == OLLAMA_HTTP_REFUSED)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_UNAVAILABLE,
			"Ollama API is not reachable at " + ep.url);
		return (false);
	}
	if (r.kind == OLLAMA_HTTP_TOO_LARGE)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_TOO_LARGE,
			std::string("Ollama's ") + what + " response exceeded "
			"MEMBRANE's size limit");
		return (false);
	}
	if (r.status < 200 || r.status > 299)
	{
		detail = ollama_error_text(r.body);
		set_err(err, MEMBRANE_RUNTIME_ERR_HTTP, std::string("Ollama answered ")
			+ what + " with HTTP " + std::to_string(r.status)
			+ (detail.empty() ? "" : ": " + detail));
		return (false);
	}
	return (true);
}

bool	membrane_ollama_list_models(std::vector<membrane_external_model_t> *out,
			membrane_runtime_error_t *err)
{
	membrane_ollama_endpoint_t	ep = membrane_ollama_resolve_endpoint();
	ollama_http_result_t		r;
	std::string					perr;

	if (!ep.ok)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_INVALID_ENDPOINT,
			std::string(MEMBRANE_OLLAMA_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return (false);
	}
	r = ollama_http(ep, "GET", "/api/tags", "", OLLAMA_TAGS_MAX_BYTES, 5);
	if (!classify_http(ep, r, "GET /api/tags", err))
		return (false);
	if (!membrane_ollama_parse_tags(r.body, out, &perr))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_MALFORMED,
			"Ollama returned a malformed model list: " + perr);
		return (false);
	}
	return (true);
}

static bool	valid_model_name(const std::string &name)
{
	if (name.empty() || name.size() > OLLAMA_MODEL_NAME_MAX)
		return (false);
	for (char c : name)
		if (!std::isgraph((unsigned char)c))
			return (false);
	return (true);
}

bool	membrane_ollama_inspect_model(const std::string &model,
			membrane_external_model_detail_t *out,
			membrane_runtime_error_t *err)
{
	membrane_ollama_endpoint_t	ep = membrane_ollama_resolve_endpoint();
	ollama_http_result_t		r;
	json						body;
	std::string					perr;

	if (!valid_model_name(model))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_INVALID_MODEL,
			"invalid Ollama model name");
		return (false);
	}
	if (membrane_ollama_is_cloud_model_ref(model))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_CLOUD_REFUSED,
			"refusing to inspect '" + model + "': Ollama resolves cloud "
			"model references against ollama.com, and this adapter never "
			"contacts cloud endpoints");
		return (false);
	}
	if (!ep.ok)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_INVALID_ENDPOINT,
			std::string(MEMBRANE_OLLAMA_ENDPOINT_ENV) + " is invalid: "
			+ ep.error);
		return (false);
	}
	body["model"] = model;
	r = ollama_http(ep, "POST", "/api/show", body.dump(),
			OLLAMA_SHOW_MAX_BYTES, 10);
	if (r.kind == OLLAMA_HTTP_RESPONSE && r.status == 404)
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND,
			"Ollama has no local model '" + model + "'");
		return (false);
	}
	if (!classify_http(ep, r, "POST /api/show", err))
		return (false);
	if (!membrane_ollama_parse_show(model, r.body, out, &perr))
	{
		set_err(err, MEMBRANE_RUNTIME_ERR_MALFORMED,
			"Ollama returned malformed model metadata: " + perr);
		return (false);
	}
	return (true);
}
