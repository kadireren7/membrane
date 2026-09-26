#ifndef MEMBRANE_OBSERVE_SHARED_H
# define MEMBRANE_OBSERVE_SHARED_H

# include <string>
# include <vector>

# include <nlohmann/json.hpp>

# include "observation.h"
# include "gpu_device.h"
# include "runtime_session.h"

/*
 * Milestone I2: the parts of the I1 observation provider that do NOT
 * belong to any one runtime -- host/GPU mapping and snapshot rendering --
 * pulled out of observe_cmd.cpp (the membrane-native provider) so
 * observe_ollama.cpp (the Ollama provider) can reuse them instead of
 * standing up a second copy. Neither function here touches the machine:
 * the caller has already probed (membrane_read_host_meminfo(),
 * membrane_gpu_list_devices()), and these only turn the raw result into
 * observation.h's provenance-labeled fields -- the SAME mapping for
 * every runtime, because "what RAM/GPU MEMBRANE itself measures" does
 * not depend on which runtime is being observed (Part 8 of I2: this is
 * the device-wide half of that split; a runtime's own per-model figures
 * are its provider's job, never this module's).
 *
 * membrane_observe_snapshot_json()/_print_human() are equally
 * runtime-agnostic: both operate only on the finished
 * membrane_observation_snapshot_t, so `membrane observe`'s dispatcher
 * (observe_cmd.cpp) calls them once regardless of which provider built
 * the snapshot.
 */

/* MEASURED (or unknown, honestly) from the host's own RAM/swap read --
 * membrane_read_host_meminfo()'s raw result, not yet probed here. */
void	membrane_observe_build_host_fields(bool meminfo_probed,
			const membrane_host_meminfo_t &meminfo, bool swap_supported,
			const std::string &meminfo_source,
			membrane_observation_snapshot_t *s);

/* MEASURED (or unknown) from membrane_gpu_list_devices()'s raw result --
 * the first GPU/iGPU ggml enumerates, the same device `membrane plan`
 * uses, regardless of which runtime is being observed. */
void	membrane_observe_build_gpu_fields(bool gpu_enumerated,
			const std::vector<membrane_gpu_device_info_t> &devices,
			membrane_observation_snapshot_t *s);

/* The stable JSON document (schema_version MEMBRANE_OBSERVATION_SCHEMA_
 * VERSION): every telemetry field is {value, known, provenance, source},
 * value null when unknown. Runtime-agnostic. */
nlohmann::json	membrane_observe_snapshot_json(
					const membrane_observation_snapshot_t &s);

void	membrane_observe_print_human(const membrane_observation_snapshot_t &s);

#endif
