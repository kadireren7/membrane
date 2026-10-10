#ifndef MEMBRANE_MODELS_CMD_H
# define MEMBRANE_MODELS_CMD_H

# include <string>
# include <vector>

/*
 * Final productization (P2/P6/P7): `membrane models` -- find and understand
 * models without remembering names or passing --json.
 *
 *   membrane models              what is installed here (name, size, quant,
 *                                loaded / default)
 *   membrane models available    the curated catalog: what each model is
 *                                for, its size, license, and how much
 *                                MEMBRANE actually knows about it
 *   membrane models recommend    what suits THIS computer, from measured
 *                                hardware facts + catalog metadata
 *
 * `membrane model ...` (singular) keeps working unchanged for scripts.
 * Read-only: nothing is installed, loaded or changed.
 */

int	membrane_models_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json);

#endif
