#ifndef MEMBRANE_MODEL_MATCH_H
# define MEMBRANE_MODEL_MATCH_H

# include <string>
# include <vector>

/*
 * Final productization (P4): "membrane use qwen" must not force anyone to
 * know the canonical model id. This is the PURE matcher; nothing here
 * touches the registry, the network or the terminal.
 *
 * Tiers, evaluated in order -- the first tier that has any hit decides:
 *   1 EXACT      case-insensitive equality with an id, display name or
 *                alias, or equality after normalization (case, '-', '_',
 *                ':', '/', ' ' and a trailing ":latest" ignored)
 *   2 PREFIX     the normalized query is a prefix of a normalized name
 *   3 SUBSTRING  the normalized query is contained in a normalized name, or
 *                every space-separated token of the query is
 *   4 FUZZY      a typo-distance (Levenshtein on the normalized form) of at
 *                most max(1, min(3, len/4)), for queries of 4+ characters;
 *                only the candidates at the SMALLEST distance are returned
 * Hits are distinct candidates, installed first, then by id: the result is
 * deterministic and independent of input order.
 *
 * Policy (enforced by callers, tested in test_model_resolve): only EXACT may
 * proceed without asking. PREFIX / SUBSTRING / FUZZY are suggestions -- for
 * anything that can download, load or change state the user must confirm,
 * and a non-interactive run lists the candidates and fails instead.
 */

typedef struct s_membrane_model_cand
{
	std::string					id;			/* canonical id */
	std::string					display;	/* human name, may be "" */
	std::vector<std::string>	aliases;
	bool						installed = false;
	bool						catalog = false;
}	membrane_model_cand_t;

typedef enum e_membrane_match_kind
{
	MEMBRANE_MATCH_NONE = 0,
	MEMBRANE_MATCH_EXACT,
	MEMBRANE_MATCH_PREFIX,
	MEMBRANE_MATCH_SUBSTRING,
	MEMBRANE_MATCH_FUZZY
}	membrane_match_kind_t;

const char	*membrane_match_kind_name(membrane_match_kind_t k);

typedef struct s_membrane_match
{
	membrane_match_kind_t	kind = MEMBRANE_MATCH_NONE;
	std::vector<size_t>		hits;		/* indices into the candidate list */
	int						distance = 0;	/* FUZZY only */
}	membrane_match_t;

/* lowercase; trailing ":latest" dropped; '-' '_' ':' '/' ' ' removed */
std::string	membrane_model_normalize(const std::string &name);

membrane_match_t	membrane_model_match(const std::string &query,
						const std::vector<membrane_model_cand_t> &cands);

#endif
