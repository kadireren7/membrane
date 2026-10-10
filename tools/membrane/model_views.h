#ifndef MEMBRANE_MODEL_VIEWS_H
# define MEMBRANE_MODEL_VIEWS_H

# include <cstdint>
# include <string>
# include <vector>

# include "model_match.h"

/*
 * Final productization (P2/P3/P7): the one place that turns MEMBRANE's three
 * sources of model facts -- the local registry, the built-in catalog and the
 * running service -- into rows a human can read. `membrane models`, the
 * picker, the dashboard and the fuzzy resolver all use it, so they can never
 * disagree about what is installed, loaded or the default.
 */

typedef struct s_membrane_model_row
{
	std::string					id;			/* canonical / registry name */
	std::string					display;	/* "Qwen2.5-1.5B-Instruct" */
	std::string					runtime = "membrane-native";
	bool						installed = false;
	std::string					path;		/* installed only */
	uint64_t					size_bytes = 0;	/* installed file, or the
										 * smallest catalog variant */
	std::string					quant;		/* "" = unknown */
	std::string					params;		/* "1.5B" */
	std::string					arch;
	std::vector<std::string>	tags;
	std::string					purpose;
	std::string					tested_status;	/* validated|smoke-tested|
										 * metadata-only|"" (unlisted) */
	std::string					license;
	bool						in_catalog = false;
	bool						resident_known = false;
	bool						resident = false;
	bool						is_default = false;
}	membrane_model_row_t;

/* "1.1 GB" / "725 MB" (decimal, like download sizes) */
std::string	membrane_human_size(uint64_t bytes);

/* Quant label from a GGUF file name ("...-Q4_K_M.gguf" -> "Q4_K_M"); ""
 * when none is recognizable. Pure. */
std::string	membrane_quant_from_filename(const std::string &filename);

/* Installed models, sorted by display name then id. probe_server: also ask
 * the configured local service which are loaded (one bounded request). */
void	membrane_model_rows_installed(std::vector<membrane_model_row_t> *out,
			bool probe_server);

/* Every catalog family (installed or not), sorted by parameter size then
 * id; `installed`/`path` filled in where a registry entry matches. */
void	membrane_model_rows_catalog(std::vector<membrane_model_row_t> *out);

/* Matcher candidates built from the same sources. */
std::vector<membrane_model_cand_t>	membrane_model_candidates(
			bool include_catalog);

/* "tiny . smoke-test . 725 MB . Q4_K_M . loaded" -- one detail line. */
std::string	membrane_model_detail_line(const membrane_model_row_t &r);

#endif
