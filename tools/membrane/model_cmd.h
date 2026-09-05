#ifndef MEMBRANE_MODEL_CMD_H
# define MEMBRANE_MODEL_CMD_H

# include <vector>
# include <string>

/*
 * Mega Phase A, PR A2: `membrane model add|remove|list|inspect` -- the
 * llama-aware CLI layer on top of registry_core.h's pure module. This is
 * the ONE place this product validates a path is a real, readable GGUF
 * (via gpu_device.h's existing membrane_gpu_estimate_model(), already
 * used by --inspect-model/--doctor in membrane-run -- never a second
 * GGUF-parsing implementation) before remembering it under a name.
 *
 * Mega Phase D, PR D1: `membrane model search|info|install|uninstall` --
 * the built-in model catalog (model_catalog.h) and real HTTPS downloads
 * (download_manager.h). `search`/`info` are pure and offline; `install`
 * downloads, validates (the same GGUF check `add` uses), and registers
 * in one step (idempotent: a re-run against an already-correct,
 * already-downloaded file skips the download and re-verifies its
 * checksum instead); `uninstall` deletes the file AND the registry
 * entry together, refusing to touch a path it did not itself download
 * (use `remove` for a manually-`add`ed model instead).
 *
 * Exit codes match membrane-run's own convention (product_cli.h):
 * MEMBRANE_EXIT_SUCCESS/CLI_ERROR/MODEL_ERROR.
 */
int	membrane_model_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
