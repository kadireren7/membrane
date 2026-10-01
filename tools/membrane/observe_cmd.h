#ifndef MEMBRANE_OBSERVE_CMD_H
# define MEMBRANE_OBSERVE_CMD_H

# include <cstdint>
# include <string>
# include <vector>

# include <nlohmann/json.hpp>

# include "observation.h"
# include "gpu_device.h"
# include "runtime_capabilities.h"
# include "registry_core.h"
# include "server_config.h"
# include "service_state.h"
# include "runtime_session.h"

/*
 * Milestone I1: `membrane observe` -- the membrane-native observation
 * provider plus its CLI. READ-ONLY: it gathers facts through EXISTING
 * probes only, and never starts/stops a service, loads/activates/
 * downloads a model, writes the registry or config, or runs inference.
 * The only network I/O is the same bounded GET /v1/status `membrane
 * status` already performs against MEMBRANE's own configured endpoint
 * (no Ollama route of any kind -- I2 owns external runtimes).
 *
 * Two stages, so the snapshot logic is testable without a host:
 *   1. membrane_observe_collect_inputs()  -- the only function that
 *      touches the machine; fills raw, unlabeled facts.
 *   2. membrane_observe_build_snapshot()  -- PURE: turns raw facts into
 *      observation.h's provenance-labeled snapshot.
 *
 * Probes reused (never re-implemented):
 *   host RAM/swap      runtime_session.h  membrane_read_host_meminfo()
 *   GPU devices/VRAM   gpu_device.h       membrane_gpu_list_devices()
 *   service manager    service_state.h    membrane_probe_service()
 *   server config      server_config.h    membrane_server_config_load()
 *   live server state  status_client.h    membrane_fetch_server_status()
 *   registry           registry_core.h    membrane_registry_load()
 *   GGUF metadata      gpu_device.h       membrane_gpu_estimate_model()
 *   Planner v2         plan_cmd.h         membrane_plan_resolve_installed_v2()
 *   runtime identity   runtime_capabilities.h membrane_runtime_describe()
 */

/* Raw facts, exactly as each probe returned them -- no provenance yet. */
typedef struct s_membrane_observe_inputs
{
	int64_t							timestamp_unix_ms;
	uint64_t						collection_duration_ms;

	bool							runtime_described;
	membrane_runtime_descriptor_t	runtime;

	bool							meminfo_probed;
	membrane_host_meminfo_t			meminfo;
	bool							swap_supported;	/* this platform's probe
									 * really reads swap (Linux only --
									 * membrane_read_host_meminfo() leaves
									 * the swap fields 0 elsewhere) */
	std::string						meminfo_source;

	bool							gpu_enumerated;
	std::vector<membrane_gpu_device_info_t>	devices;

	bool							service_probed;
	membrane_service_probe_t		service;

	bool							config_loaded;	/* false = present but
									 * invalid (a missing file loads as
									 * defaults, per server_config.h) */
	membrane_server_config_t		config;

	bool							server_probed;
	bool							server_reachable;
	nlohmann::json					server_status;

	bool							registry_loaded;
	bool							entry_found;
	membrane_registry_entry_t		entry;

	bool							gguf_read;
	membrane_gpu_model_estimate_t	gguf;

	bool							plan_resolved;
	std::string						plan_error_code;
	bool							plan_has_decisions;
	uint64_t						plan_context;
	int								plan_kv_precision;	/* MEMBRANE_JOINT_KV_* */
	int								plan_kv_placement;	/* MEMBRANE_JOINT_PLACEMENT_* */
	bool							plan_kv_bytes_known;
	uint64_t						plan_kv_bytes;
	bool							plan_quant_known;
	std::string						plan_quant;
}	membrane_observe_inputs_t;

void	membrane_observe_inputs_init(membrane_observe_inputs_t *in);

/* Real, read-only probes for runtime_id (only membrane-native in I1). */
void	membrane_observe_collect_inputs(const std::string &runtime_id,
			membrane_observe_inputs_t *in);

/* Pure. Always finalizes the snapshot (status computed). */
void	membrane_observe_build_snapshot(const std::string &runtime_id,
			const membrane_observe_inputs_t &in,
			membrane_observation_snapshot_t *out);

/* The stable JSON document (schema_version MEMBRANE_OBSERVATION_SCHEMA_
 * VERSION): every telemetry field is {value, known, provenance, source},
 * value null when unknown. */
nlohmann::json	membrane_observe_snapshot_json(
					const membrane_observation_snapshot_t &s);

void	membrane_observe_print_human(const membrane_observation_snapshot_t &s);

/* `membrane observe [--runtime ID]`. Exit 0 for complete/partial,
 * MEMBRANE_EXIT_RUNTIME_ERROR for unavailable, CLI_ERROR for a bad
 * option / unknown or not-yet-observable runtime id. */
int		membrane_observe_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
