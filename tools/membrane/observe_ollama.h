#ifndef MEMBRANE_OBSERVE_OLLAMA_H
# define MEMBRANE_OBSERVE_OLLAMA_H

# include <cstdint>
# include <string>
# include <vector>

# include "observation.h"
# include "gpu_device.h"
# include "runtime_session.h"
# include "runtime_capabilities.h"
# include "runtime_ollama.h"

/*
 * Milestone I2: `membrane observe --runtime ollama` -- the Ollama
 * observation provider. READ-ONLY, and it maps GET /api/ps onto the SAME
 * I1 snapshot model (observation.h) that observe_cmd.cpp's native
 * provider fills -- no second telemetry model (docs/observability.md,
 * docs/runtime-ollama.md).
 *
 * Same two-stage split as I1's native provider, for the same reason
 * (testable without a host or a real Ollama daemon):
 *   1. membrane_observe_ollama_collect_inputs()  -- the only function
 *      that touches the network/machine; fills raw, unlabeled facts.
 *   2. membrane_observe_ollama_build_snapshot()  -- PURE: turns raw facts
 *      into observation.h's provenance-labeled snapshot.
 *
 * Network I/O, in order, and NEVER more than this (Part 15/17): one
 * GET /api/version (membrane_ollama_describe(), H2's existing discovery
 * probe) and -- ONLY if that reports the runtime AVAILABLE -- one
 * GET /api/ps (membrane_ollama_list_running(), runtime_ollama.h). If
 * discovery fails, /api/ps is never called.
 *
 * Probes reused (never re-implemented):
 *   Ollama discovery + version   runtime_ollama.h  membrane_ollama_describe()
 *   Ollama loaded models         runtime_ollama.h  membrane_ollama_list_running()
 *   host RAM/swap                runtime_session.h membrane_read_host_meminfo()
 *   GPU devices/VRAM             gpu_device.h      membrane_gpu_list_devices()
 *   host/GPU -> snapshot mapping observe_shared.h  membrane_observe_build_
 *                                                   {host,gpu}_fields()
 * MEMBRANE's own registry, server config, service-manager probe, GGUF
 * reader and Planner v2 are never touched here: those are MEMBRANE-native
 * concepts with no meaning for an externally-owned Ollama model, so the
 * corresponding snapshot fields are left honestly unknown
 * ("not_applicable_external_runtime") rather than filled from a
 * MEMBRANE-side guess.
 */

/* Raw facts, exactly as each probe returned them -- no provenance yet. */
typedef struct s_membrane_observe_ollama_inputs
{
	int64_t							timestamp_unix_ms;
	uint64_t						collection_duration_ms;

	membrane_runtime_descriptor_t	runtime;	/* membrane_ollama_describe()
										 * always fills this (H2 contract:
										 * "always fills *out, never
										 * fails") */

	bool							meminfo_probed;
	membrane_host_meminfo_t			meminfo;
	bool							swap_supported;
	std::string						meminfo_source;

	bool							gpu_enumerated;
	std::vector<membrane_gpu_device_info_t>	devices;

	bool							ps_called;	/* false iff discovery did
										 * not report AVAILABLE -- /api/ps
										 * was never requested (Part 15) */
	bool							ps_ok;
	std::vector<membrane_ollama_process_model_t>	models;
	membrane_runtime_error_t		ps_error;	/* set iff ps_called &&
										 * !ps_ok */
}	membrane_observe_ollama_inputs_t;

void	membrane_observe_ollama_inputs_init(
			membrane_observe_ollama_inputs_t *in);

/* Real, read-only probes: membrane_ollama_describe(), and -- only when
 * that reports AVAILABLE -- membrane_read_host_meminfo(),
 * membrane_gpu_list_devices(), membrane_ollama_list_running(). */
void	membrane_observe_ollama_collect_inputs(
			membrane_observe_ollama_inputs_t *in);

/* Pure. Always finalizes the snapshot (status computed). runtime_id is
 * always MEMBRANE_RUNTIME_ID_OLLAMA. */
void	membrane_observe_ollama_build_snapshot(
			const membrane_observe_ollama_inputs_t &in,
			membrane_observation_snapshot_t *out);

#endif
