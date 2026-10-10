#ifndef MEMBRANE_PICKER_H
# define MEMBRANE_PICKER_H

# include <iosfwd>
# include <string>
# include <vector>

/*
 * Final productization (P3): a small numbered picker with pagination and
 * search, with no TUI dependency -- plain lines on a stream, so it works
 * over ssh, in tmux, and is tested with string streams.
 *
 *   Installed models -- Page 1/2
 *
 *     1. Qwen2.5 1.5B Instruct
 *        multilingual . general . 1.1 GB
 *     2. SmolLM2 360M Instruct
 *        tiny . fast smoke-test . 730 MB
 *
 *   [1-2] select   n next   p prev   / search   q quit
 *
 * Numbers are global within the (possibly filtered) list, so they stay
 * valid across pages. `/term` filters by a case-insensitive substring of the
 * label and detail; `/` alone clears the filter. End of input quits. The
 * caller decides the ORDER (deterministic) -- the picker never reorders.
 */

typedef struct s_membrane_picker_item
{
	std::string	label;
	std::string	detail;			/* one short line, may be "" */
}	membrane_picker_item_t;

typedef struct s_membrane_picker_opts
{
	std::string	title;
	size_t		page_size = 8;
	bool		allow_search = true;
	size_t		max_invalid = 6;	/* consecutive bad inputs before giving up */
}	membrane_picker_opts_t;

typedef enum e_membrane_picker_status
{
	MEMBRANE_PICK_SELECTED = 0,
	MEMBRANE_PICK_QUIT			/* q, EOF, or too many bad inputs */
}	membrane_picker_status_t;

typedef struct s_membrane_picker_result
{
	membrane_picker_status_t	status = MEMBRANE_PICK_QUIT;
	size_t						index = 0;	/* into the ORIGINAL items */
}	membrane_picker_result_t;

membrane_picker_result_t	membrane_picker_run(
								const std::vector<membrane_picker_item_t> &items,
								const membrane_picker_opts_t &opts,
								std::istream &in, std::ostream &out);

#endif
