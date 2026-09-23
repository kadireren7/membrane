#ifndef MEMBRANE_RUNTIME_OLLAMA_H
# define MEMBRANE_RUNTIME_OLLAMA_H

# include <string>
# include <vector>

# include "runtime_adapter.h"

/*
 * Milestone H2: the first external-runtime adapter -- Ollama, READ-ONLY.
 * The one place in this codebase that speaks Ollama's HTTP API (all of
 * it through cpp-httplib, the HTTP client MEMBRANE already vendors).
 *
 * API surface used (audited against ollama/ollama v0.34.3's own
 * server/routes.go, api/types.go and docs/openapi.yaml -- see
 * docs/runtime-ollama.md), and NOTHING else:
 *   GET  /api/version   -> {"version": "..."}          (discovery/health)
 *   GET  /api/tags      -> {"models": [ ... ]}          (inventory)
 *   POST /api/show      <- {"model": NAME}              (metadata; a POST
 *                          only because Ollama defines it that way -- it
 *                          reads the local manifest, it changes nothing)
 * membrane_ollama_request_allowed() is the enforced allowlist: the one
 * internal HTTP helper refuses any other method/path before a socket is
 * opened. In particular /api/generate, /api/chat, /api/embed(dings),
 * /api/pull, /api/push, /api/create, /api/copy, /api/delete, /api/blobs,
 * the OpenAI-compatible /v1 routes, and every cloud/account route are
 * unreachable from this module.
 *
 * H2 does NOT send inference requests and does NOT change Ollama state.
 */

# define MEMBRANE_OLLAMA_ENDPOINT_ENV		"MEMBRANE_OLLAMA_ENDPOINT"
/* Ollama's own documented default (envconfig.Host(): 127.0.0.1:11434). */
# define MEMBRANE_OLLAMA_DEFAULT_ENDPOINT	"http://127.0.0.1:11434"
# define MEMBRANE_OLLAMA_DEFAULT_PORT		11434

typedef struct s_membrane_ollama_endpoint
{
	bool		ok;
	std::string	url;		/* normalized "http://HOST:PORT" */
	std::string	host;		/* without IPv6 brackets */
	int			port;
	bool		loopback;	/* only loopback -> LOCAL_EXTERNAL */
	bool		from_env;
	std::string	error;		/* set iff !ok */
}	membrane_ollama_endpoint_t;

/* Pure. Accepts only "http://HOST[:PORT][/]" (HOST may be a bracketed
 * IPv6 literal). Rejects https (no TLS in this build), any other scheme,
 * userinfo (credentials are never sent), paths, queries, fragments. */
membrane_ollama_endpoint_t	membrane_ollama_parse_endpoint(
								const std::string &raw);
/* MEMBRANE_OLLAMA_ENDPOINT if set and non-empty, else the default. */
membrane_ollama_endpoint_t	membrane_ollama_resolve_endpoint(void);

/* Pure: the static capability contract (see runtime_ollama.cpp). */
void	membrane_ollama_capabilities(membrane_runtime_capabilities_t *out);

/* Pure: true iff method+path is one of the 3 read-only calls above. */
bool	membrane_ollama_request_allowed(const std::string &method,
			const std::string &path);

/* Pure: mirrors Ollama's own internal/modelref parseSourceSuffix() --
 * true for "NAME:cloud" and "NAME:TAG-cloud" (case-insensitive), which
 * Ollama's /api/show would PROXY to ollama.com instead of answering
 * locally. MEMBRANE refuses these before any request is made. */
bool	membrane_ollama_is_cloud_model_ref(const std::string &name);

/* Pure parsers over a raw response body (exposed for fixture tests). */
bool	membrane_ollama_parse_version(const std::string &body,
			std::string *version);
bool	membrane_ollama_parse_tags(const std::string &body,
			std::vector<membrane_external_model_t> *out, std::string *err);
bool	membrane_ollama_parse_show(const std::string &model,
			const std::string &body, membrane_external_model_detail_t *out,
			std::string *err);

/* Live, bounded, read-only probes (runtime_adapter.h signatures). */
void	membrane_ollama_describe(membrane_runtime_descriptor_t *out);
bool	membrane_ollama_list_models(std::vector<membrane_external_model_t> *out,
			membrane_runtime_error_t *err);
bool	membrane_ollama_inspect_model(const std::string &model,
			membrane_external_model_detail_t *out,
			membrane_runtime_error_t *err);

#endif
