#ifndef MEMBRANE_CLI_SHARED_H
# define MEMBRANE_CLI_SHARED_H

# include <string>
# include <vector>

/*
 * Mega Phase D, PR D6: small CLI-composition helpers originally written
 * (and still tested) as file-local statics in setup_cmd.cpp -- extracted
 * here, unchanged, so use_cmd.cpp can reuse the exact same interactive-
 * prompt/silent-reuse machinery rather than a second copy. setup_cmd.cpp
 * itself is refactored to call these instead of its own former statics
 * (Section 2 of the D6 task: "do not duplicate existing logic" applies to
 * this project's OWN CLI-composition code too, not just registry/install/
 * service).
 */

/* True iff stdin is a real terminal -- every non-interactive path below
 * (a CI job, a pipe, `< /dev/null`) is driven off this one check. */
bool	membrane_cli_is_interactive(void);

/* Section 6 of the D6 task (same convention setup_cmd.cpp's own consent
 * prompts already used): assume_yes OR non-interactive always returns
 * default_yes without ever blocking on stdin. NOTE: this makes non-
 * interactive runs succeed with default_yes, which is exactly what
 * `membrane setup`'s own optional prompts want -- it is deliberately NOT
 * used by `membrane use`'s own not-installed consent gate (Section 6 of
 * the D6 task requires FAILING clearly in that one specific case, never
 * silently defaulting to yes); see use_cmd.cpp's own bespoke consent
 * check for that path. */
bool	membrane_cli_ask_yes_no(const std::string &prompt, bool default_yes,
			bool assume_yes);

/* Returns "" (never blocks) when non-interactive. */
std::string	membrane_cli_ask_line(const std::string &prompt);

/* Suppresses ALL prose when want_json is true -- the one JSON document a
 * machine caller gets is assembled separately and printed exactly once. */
void	membrane_cli_narrate(bool want_json, const char *fmt, ...);

/* RAII: redirects real fd 1 to MEMBRANE_NULL_DEVICE for the duration of
 * one reused dispatch call, so a command that always prints SOMETHING
 * itself (no genuinely silent mode) can be reused internally without
 * polluting a caller's own single clean JSON summary. A no-op when
 * `active` is false. */
class stdout_silencer_t
{
	public:
		explicit stdout_silencer_t(bool active);
		~stdout_silencer_t(void);

		stdout_silencer_t(const stdout_silencer_t &) = delete;
		stdout_silencer_t	&operator=(const stdout_silencer_t &) = delete;

	private:
		bool	active_;
		int		saved_fd_;
};

/* Runs `fn(args, false)` (always non-JSON -- neither reused command has a
 * genuinely silent mode, only human-prose vs. its-own-JSON-line) under a
 * stdout_silencer_t active iff want_json, so the caller's own control
 * flow can rely purely on the real return code. */
int	membrane_cli_dispatch_silently_if_json(bool want_json,
			const std::vector<std::string> &args,
			int (*fn)(const std::vector<std::string> &, bool));

#endif
