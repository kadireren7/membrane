#ifndef MEMBRANE_MODEL_CATALOG_H
# define MEMBRANE_MODEL_CATALOG_H

# include <cstdint>
# include <string>
# include <vector>

/*
 * Mega Phase D, PR D1: a curated, versioned, BUILT-IN model catalog --
 * `membrane model search`/`info` work with zero network access (Section
 * 8 of the task: "no network dependency for inference"; this extends
 * that same principle to catalog browsing itself). Only `install`
 * needs a real network call (download_manager.h).
 *
 * This is deliberately NOT a web marketplace and NOT a scraped index --
 * every entry here was individually verified against a real, reputable
 * Hugging Face repository (exact filename + real Content-Length via a
 * real HTTP HEAD/GET, license tag read from the repo's own metadata)
 * before being added. See model_catalog.cpp's own top comment for the
 * exact verification each entry went through, and docs/model-catalog.md
 * for the source policy this catalog follows.
 */

typedef struct s_membrane_catalog_variant
{
	std::string	quant;			/* e.g. "Q4_K_M", "F16" -- as the
								 * upstream repo itself names it, never
								 * renamed/normalized */
	std::string	filename;		/* exact upstream asset filename */
	std::string	download_url;	/* exact, real, HTTPS resolve URL */
	uint64_t	size_bytes;		/* real, verified Content-Length at the
								 * time this catalog entry was recorded
								 * (see recorded_at below) -- upstream
								 * files can change; install-time
								 * verification never trusts this
								 * number alone (download_manager.h
								 * checks the REAL response) */
	std::string	sha256;			/* real SHA-256, read directly from
								 * Hugging Face's own API
								 * (`?blobs=true`, `siblings[].lfs.sha256`)
								 * at recorded_at -- verified, never
								 * fabricated. Empty only if upstream
								 * genuinely does not provide one for a
								 * future catalog entry (Section 6:
								 * "checksum verification if upstream
								 * provides checksum" -- every entry in
								 * this PR's own catalog has one). */
}	membrane_catalog_variant_t;

typedef struct s_membrane_catalog_family
{
	std::string	name;			/* canonical id, e.g.
								 * "smollm2-135m-instruct" -- lowercase,
								 * hyphenated, stable */
	std::vector<std::string>		aliases;	/* e.g. {"smollm2:135m"} --
								 * every alias must resolve to exactly
								 * this one family, never ambiguous
								 * (Section 22 of the task) */
	std::string	display_name;	/* e.g. "SmolLM2-135M-Instruct" */
	std::string	arch;			/* real compat_check.c/gguf arch name,
								 * e.g. "llama", "qwen2" */
	std::string	parameter_count;	/* human label, e.g. "135M" --
								 * display only, never parsed */
	std::string	provider;		/* the real Hugging Face org/user that
								 * published these GGUF files, e.g.
								 * "unsloth" -- NOT necessarily the
								 * original model author */
	std::string	repo_url;		/* real, browsable repo URL */
	std::string	license;		/* real license tag read from the
								 * repo's own metadata, e.g.
								 * "apache-2.0" */
	std::string	chat_template_status;	/* "present" or "absent" -- a
								 * real, disclosed fact (Mega Phase C's
								 * own known-limitation: a model with no
								 * chat template only fails at
								 * chat-completion time today; the
								 * catalog at least surfaces this
								 * upfront now) */
	std::string	compatibility_status;	/* mirrors docs/compatibility.json's
								 * own SUPPORTED/UNSUPPORTED/
								 * NOT_YET_VALIDATED vocabulary */
	std::string	compatibility_evidence;	/* pointer to the real evidence,
								 * e.g. "docs/compatibility.json MC-01" */
	std::string	recorded_at;	/* ISO8601 date this entry's sizes/
								 * license were last verified against
								 * the real upstream repo */
	std::vector<membrane_catalog_variant_t>	variants;
}	membrane_catalog_family_t;

typedef struct s_membrane_catalog
{
	int32_t	schema_version;
	std::vector<membrane_catalog_family_t>	families;
}	membrane_catalog_t;

/* The built-in catalog, parsed once from the compiled-in JSON string
 * (model_catalog.cpp's own MEMBRANE_CATALOG_JSON constant) -- pure,
 * no filesystem/network access, safe to call repeatedly and in tests. */
membrane_catalog_t	membrane_catalog_load(void);

/* Case-insensitive substring search across name/aliases/display_name/
 * arch/provider. Empty query returns every family (used by `membrane
 * model search` with no argument -- "list everything"). */
std::vector<const membrane_catalog_family_t *>	membrane_catalog_search(
			const membrane_catalog_t &cat, const std::string &query);

/* Exact (case-insensitive) match against name OR any alias. Returns
 * NULL if no family matches -- the caller decides whether an unresolved
 * name is an error (Section 22: "no ambiguous fuzzy install without
 * confirmation" -- this function itself never guesses/fuzzy-matches). */
const membrane_catalog_family_t	*membrane_catalog_resolve(
			const membrane_catalog_t &cat, const std::string &name_or_alias);

/* Finds a specific variant within a family by exact (case-insensitive)
 * quant name (e.g. "Q4_K_M"). Returns NULL if not found. */
const membrane_catalog_variant_t	*membrane_catalog_find_variant(
			const membrane_catalog_family_t &family, const std::string &quant);

#endif
