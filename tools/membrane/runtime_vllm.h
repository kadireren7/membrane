#ifndef MEMBRANE_RUNTIME_VLLM_H
# define MEMBRANE_RUNTIME_VLLM_H

# include <string>
# include <vector>

# include "runtime_adapter.h"

/*
 * Milestone L: a READ-ONLY vLLM adapter. vLLM is a supported
 * *inspect / advise / orchestrate* runtime -- MEMBRANE does not control it.
 *
 * Contract audited against vllm-project/vllm release v0.31.0 (published
 * 2026-10-05), source files vllm/entrypoints/serve/instrumentator/
 * health.py, basic.py and vllm/entrypoints/openai/models/api_router.py,
 * vllm/entrypoints/serve/engine/protocol.py:
 *
 *   GET /health      200 (engine healthy) / 503 (engine dead)  -- liveness
 *   GET /version     {"version": "<vllm version>"}
 *   GET /v1/models   OpenAI model list; each card: id, root,
 *                    max_model_len (the served model's configured maximum
 *                    context, NOT an active/allocated context)
 *   GET /load        {"server_load": N} -- requests currently using the
 *                    GPU (in-flight)
 *
 * and NOTHING else. membrane_vllm_request_allowed() is the enforced
 * allowlist (exact GET + path); there is no POST at all, so no inference
 * request, no /sleep|/wake_up (dev-mode only upstream), no /reset_prefix_
 * cache, no /metrics (a Prometheus text surface MEMBRANE does not parse).
 *
 * Why no control: vLLM fixes its model, context (--max-model-len), KV dtype
 * (--kv-cache-dtype) and memory fraction (--gpu-memory-utilization) at
 * process start. There is no supported runtime API to change them, so every
 * control capability is UNSUPPORTED -- a control MEMBRANE could only
 * "apply" by restarting someone else's server is not a control.
 *
 * Endpoint: MEMBRANE_VLLM_ENDPOINT (same http://HOST[:PORT] rules as
 * Ollama: no https, no credentials, no paths), default
 * http://127.0.0.1:8000 (vLLM's default port). Never contacted unless a
 * command asks about vLLM.
 */

# define MEMBRANE_VLLM_ENDPOINT_ENV			"MEMBRANE_VLLM_ENDPOINT"
# define MEMBRANE_VLLM_DEFAULT_ENDPOINT		"http://127.0.0.1:8000"
# define MEMBRANE_VLLM_AUDITED_VERSION		"v0.31.0"

/* Pure: static capability contract (see runtime_vllm.cpp). */
void	membrane_vllm_capabilities(membrane_runtime_capabilities_t *out);

/* Pure: exact allowlist. Only GET /health, /version, /v1/models, /load. */
bool	membrane_vllm_request_allowed(const std::string &method,
			const std::string &path);

/* Pure parsers (fixture-tested). */
bool	membrane_vllm_parse_version(const std::string &body,
			std::string *version);
bool	membrane_vllm_parse_models(const std::string &body,
			std::vector<membrane_external_model_t> *out, std::string *err);
bool	membrane_vllm_parse_load(const std::string &body, long long *load);

/* Live, bounded, read-only (runtime_adapter.h signatures). */
void	membrane_vllm_describe(membrane_runtime_descriptor_t *out);
bool	membrane_vllm_list_models(std::vector<membrane_external_model_t> *out,
			membrane_runtime_error_t *err);
bool	membrane_vllm_inspect_model(const std::string &model,
			membrane_external_model_detail_t *out,
			membrane_runtime_error_t *err);
/* GET /load. *load = in-flight requests. */
bool	membrane_vllm_server_load(long long *load,
			membrane_runtime_error_t *err);

#endif
