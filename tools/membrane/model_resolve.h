#ifndef MEMBRANE_MODEL_RESOLVE_H
# define MEMBRANE_MODEL_RESOLVE_H

# include <iosfwd>
# include <string>
# include <vector>

# include "model_match.h"
# include "model_views.h"

/*
 * Final productization (P3/P4): turning what a user typed (or did not type)
 * into one canonical model id, safely.
 *
 *   exact match                      -> proceeds, no question
 *   one prefix/substring/fuzzy hit   -> `Did you mean "X"? [Y/n]` (a TTY)
 *   several hits                     -> a numbered choice (a TTY)
 *   anything but an exact match when there is no TTY
 *                                    -> NOTHING is chosen: the candidates
 *                                       are listed and the command fails
 *
 * The rule is deliberate: a fuzzy guess must never download, load or change
 * state without the user seeing and confirming it. `--yes` does not waive
 * this -- it consents to a download of a model you NAMED, not to MEMBRANE's
 * guess at which one you meant.
 */

typedef enum e_membrane_resolve_status
{
	MEMBRANE_RESOLVE_EXACT = 0,		/* an exact match; no question asked */
	MEMBRANE_RESOLVE_CONFIRMED,		/* the user accepted a suggestion */
	MEMBRANE_RESOLVE_DECLINED,		/* the user said no / quit */
	MEMBRANE_RESOLVE_NEEDS_CHOICE,	/* non-interactive: listed, not chosen */
	MEMBRANE_RESOLVE_NOT_FOUND
}	membrane_resolve_status_t;

typedef struct s_membrane_resolve_result
{
	membrane_resolve_status_t	status = MEMBRANE_RESOLVE_NOT_FOUND;
	size_t						index = 0;		/* into the candidates */
	std::string					id;				/* canonical id */
	bool						installed = false;
	membrane_match_kind_t		kind = MEMBRANE_MATCH_NONE;
	std::string					message;		/* for the non-OK cases */
}	membrane_resolve_result_t;

typedef struct s_membrane_resolve_io
{
	bool			interactive = false;
	std::istream	*in = nullptr;		/* nullptr = std::cin */
	std::ostream	*out = nullptr;		/* nullptr = std::cout */
}	membrane_resolve_io_t;

/* io.interactive for the real terminal: stdin is a TTY. */
membrane_resolve_io_t	membrane_resolve_io_for_terminal(void);

membrane_resolve_result_t	membrane_model_resolve(const std::string &query,
		const std::vector<membrane_model_cand_t> &cands,
		const std::string &command_for_hint, const membrane_resolve_io_t &io);

/* `membrane use` / `membrane chat` with no model: a numbered picker over the
 * given rows (the caller sorts them). Returns the chosen index, or -1 for
 * quit/EOF. */
int	membrane_model_pick(const std::string &title,
		const std::vector<membrane_model_row_t> &rows,
		const membrane_resolve_io_t &io);

#endif
