#ifndef MEMBRANE_DOCTOR_CMD_H
# define MEMBRANE_DOCTOR_CMD_H

# include <string>
# include <vector>

# include <nlohmann/json.hpp>

/*
 * Mega Phase C, PR C1, Sections 9-10 of the task: `membrane doctor` --
 * the unified product-facing diagnostic surface. Orchestrates EXISTING
 * capabilities (membrane-run --doctor for hardware, registry_core.h for
 * the model registry, server_config.h for the server config,
 * systemd_unit.h/subprocess.h for the installed service, status_client.h
 * for live HTTP status) rather than re-implementing any of them --
 * Section 5's own "orchestrate, not duplicate" rule applies here too,
 * not just to `membrane setup`.
 *
 * Never performs generation (Section 9: "Do NOT perform expensive
 * generation"). Warnings are never fatal (Section 10) -- the process
 * exit code is only nonzero when at least one check is a real ERROR.
 */
int	membrane_doctor_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

/* Mega Phase C, PR C1, Section 5: runs every doctor check and returns
 * the exact same JSON shape `membrane doctor --json` prints
 * ("schema_version"/"membrane_version"/"overall"/"checks"), without
 * printing anything itself -- so `membrane setup` can reuse the SAME
 * real hardware/registry/service/HTTP checks instead of a second,
 * drifting implementation ("orchestrate existing capabilities, not
 * duplicate them"). Returns the same "overall" string doctor itself
 * would report (OK/WARN/ERROR). */
std::string	membrane_doctor_collect(nlohmann::json *out_root);

#endif
