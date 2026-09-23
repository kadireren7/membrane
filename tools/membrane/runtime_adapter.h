#ifndef MEMBRANE_RUNTIME_ADAPTER_H
# define MEMBRANE_RUNTIME_ADAPTER_H

# include <cstddef>
# include <cstdint>
# include <string>
# include <vector>

# include "runtime_capabilities.h"

/*
 * Milestone H2: the minimal boundary that lets ONE external runtime
 * (Ollama, runtime_ollama.h) sit next to the built-in native runtime
 * (runtime_capabilities.h). Deliberately not a plugin system: no shared-
 * library loading, no registration macros, no dynamic discovery -- just a
 * fixed, statically-linked table (runtime_registry.cpp) of plain function
 * pointers. Adding a later vLLM adapter means adding one more row.
 *
 * Every adapter entry point is READ-ONLY by contract: describe() may
 * probe (one bounded request), list_models()/inspect_model() may read the
 * runtime's own inventory, and nothing here can install, start, stop,
 * pull, delete, load, unload, reconfigure, or run inference on anything.
 */

/*
 * A model as an EXTERNAL runtime reports it -- kept strictly separate
 * from MEMBRANE's own model registry (registry_core.h). Nothing in H2
 * converts one into the other; a later milestone may compare them.
 * Backend-neutral on purpose: every field is "as the runtime reported
 * it", and anything the runtime did not report is left empty/unknown
 * rather than inferred (parameter_size stays Ollama's own string, e.g.
 * "8.0B", never parsed into a number MEMBRANE did not receive).
 */
typedef struct s_membrane_external_model
{
	std::string					runtime_id;
	std::string					runtime_model_id;	/* the id the runtime
									 * itself accepts back, verbatim */
	std::string					display_name;
	std::string					family;				/* "" = unknown */
	std::vector<std::string>	families;
	std::string					format;				/* e.g. "gguf" */
	std::string					parameter_size;		/* runtime's string */
	std::string					quantization;		/* runtime's string */
	bool						size_known;
	uint64_t					size_bytes;
	std::string					digest;				/* "" = unknown */
	std::string					modified_at;		/* runtime's timestamp
									 * string, verbatim */
	bool						context_length_known;
	uint64_t					context_length;		/* model's trained max,
									 * NOT a configured/active context */
	bool						remote;				/* runtime says this
									 * entry is a stub for a model hosted
									 * elsewhere (Ollama: remote_host) */
	std::string					remote_host;
	std::vector<std::string>	runtime_capabilities;	/* e.g.
									 * "completion"/"vision" -- the MODEL's
									 * own features per the runtime, NOT
									 * membrane_runtime_capabilities_t */
	membrane_capability_provenance_t	provenance;
}	membrane_external_model_t;

void	membrane_external_model_init(membrane_external_model_t *m);

typedef struct s_membrane_external_model_detail
{
	membrane_external_model_t	model;
	std::string					architecture;		/* "" = unknown */
	bool						parameter_count_known;
	uint64_t					parameter_count;
	std::string					parameters;			/* runtime default
									 * parameter text, capped */
	bool						parameters_truncated;
	bool						has_template;		/* presence only --
									 * never dumped by default */
	bool						has_system;
	bool						has_license;
}	membrane_external_model_detail_t;

void	membrane_external_model_detail_init(
			membrane_external_model_detail_t *d);

/* Stable, machine-readable error codes for adapter reads. */
# define MEMBRANE_RUNTIME_ERR_INVALID_ENDPOINT		"INVALID_ENDPOINT"
# define MEMBRANE_RUNTIME_ERR_UNAVAILABLE			"RUNTIME_UNAVAILABLE"
# define MEMBRANE_RUNTIME_ERR_HTTP					"RUNTIME_HTTP_ERROR"
# define MEMBRANE_RUNTIME_ERR_MALFORMED			"MALFORMED_RESPONSE"
# define MEMBRANE_RUNTIME_ERR_TOO_LARGE			"RESPONSE_TOO_LARGE"
# define MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND		"MODEL_NOT_FOUND"
# define MEMBRANE_RUNTIME_ERR_INVALID_MODEL		"INVALID_MODEL_NAME"
# define MEMBRANE_RUNTIME_ERR_CLOUD_REFUSED		"CLOUD_MODEL_REFUSED"

typedef struct s_membrane_runtime_error
{
	std::string	code;
	std::string	message;
}	membrane_runtime_error_t;

typedef struct s_membrane_runtime_adapter
{
	const char	*id;
	/* Always fills *out (never fails): an unreachable runtime is still a
	 * KNOWN runtime whose availability is UNAVAILABLE. */
	void		(*describe)(membrane_runtime_descriptor_t *out);
	/* NULL when the runtime's models are not exposed through `membrane
	 * runtime` (membrane-native: its models ARE MEMBRANE's own registry,
	 * `membrane model list`). */
	bool		(*list_models)(std::vector<membrane_external_model_t> *out,
					membrane_runtime_error_t *err);
	bool		(*inspect_model)(const std::string &model,
					membrane_external_model_detail_t *out,
					membrane_runtime_error_t *err);
}	membrane_runtime_adapter_t;

/* The fixed table, in display order: membrane-native first, then
 * external adapters. Never includes a reserved-only id (vllm). */
size_t								membrane_runtime_registry_count(void);
const membrane_runtime_adapter_t	*membrane_runtime_registry_at(size_t i);
const membrane_runtime_adapter_t	*membrane_runtime_registry_find(
										const std::string &id);

#endif
