#ifndef MEMBRANE_SUBPROCESS_H
# define MEMBRANE_SUBPROCESS_H

# include <string>
# include <vector>

/*
 * Mega Phase B, PR B1: a minimal, safe subprocess runner for `membrane
 * service`'s systemctl/journalctl calls. Deliberately fork()+execvp(),
 * NEVER system()/popen() with an interpolated shell string -- argv is
 * passed directly to execvp(), so there is no shell parsing step for
 * injected metacharacters to exploit, even if a caller-controlled value
 * (a unit name, in principle) ever ended up in argv (Section 11 of the
 * task: "No shell injection"). Linux-only, matching this project's
 * existing Linux-only scope (docs/install.md).
 */

typedef struct s_membrane_subprocess_result
{
	bool		spawn_failed;	/* fork()/execvp() itself failed -- exit_
								 * code/output below are meaningless */
	int			exit_code;		/* real WEXITSTATUS(), or -1 if the child
								 * was killed by a signal */
	std::string	stdout_output;
	std::string	stderr_output;
}	membrane_subprocess_result_t;

/* Runs argv[0] with the remaining elements as its own argv (argv[0]
 * itself is looked up via PATH, matching execvp()'s own contract --
 * every caller in this project passes a bare command name like
 * "systemctl", never a caller-assembled path). Blocks until the child
 * exits; stdout/stderr are captured via pipes, not any TTY-shared
 * behavior a caller-visible shell would introduce. timeout_seconds <= 0
 * means no timeout (only used for genuinely bounded commands like
 * daemon-reload/start/stop; logs uses a real bound via journalctl's own
 * -n flag instead of a wall-clock timeout). */
bool	membrane_run_subprocess(const std::vector<std::string> &argv,
			membrane_subprocess_result_t *out, int timeout_seconds = 10);

#endif
