#ifndef MEMBRANE_ADVISE_CMD_H
# define MEMBRANE_ADVISE_CMD_H

# include <string>
# include <vector>

/*
 * Milestone I3 (memory intelligence): `membrane advise` -- the CLI for
 * membrane_memory_intelligence_assemble() (tools/membrane-run/
 * memory_intelligence.h). READ-ONLY: collects EXACTLY the same bounded
 * facts `membrane observe` already collects for the same runtime id
 * (membrane_observe_collect_inputs()/membrane_observe_ollama_collect_
 * inputs(), observe_cmd.h/observe_ollama.h -- never a new probe, never a
 * new Ollama route beyond I2's existing allowlist), then hands the
 * resulting snapshot plus that runtime's own capability matrix (already
 * embedded in membrane_observe_inputs_t::runtime.capabilities/membrane_
 * observe_ollama_inputs_t::runtime.capabilities -- no second capability
 * lookup) to the pure intelligence layer.
 *
 * `membrane observe` stays facts-only, with no recommendations
 * (docs/observability.md's own top comment). `membrane advise` is the
 * SEPARATE interpretation layer Milestone I3 adds on top -- see
 * docs/memory-intelligence.md. Like `membrane observe`/`membrane plan`,
 * this command never starts/stops a service, loads/unloads/switches a
 * model, changes context/GPU-layers/KV settings, or performs inference.
 */

int	membrane_advise_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
