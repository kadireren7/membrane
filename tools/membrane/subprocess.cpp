#include "subprocess.h"

#include <cerrno>
#include <cstring>
#include <ctime>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

/*
 * See subprocess.h's own top comment. Reads stdout/stderr concurrently
 * via poll() (never sequential blocking reads on two pipes at once --
 * that risks a real deadlock if the child fills one pipe's kernel buffer
 * while this process is blocked reading the other) with a wall-clock
 * deadline; a child that outlives it is SIGKILL'd, waited for
 * (never left a zombie), and reported as a timeout via spawn_failed.
 */

static void	set_nonblocking(int fd)
{
	int	flags = fcntl(fd, F_GETFL, 0);

	if (flags != -1)
		fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

bool	membrane_run_subprocess(const std::vector<std::string> &argv,
			membrane_subprocess_result_t *out, int timeout_seconds)
{
	*out = membrane_subprocess_result_t();
	if (argv.empty())
	{
		out->spawn_failed = true;
		return (false);
	}
	int	out_pipe[2];
	int	err_pipe[2];

	if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0)
	{
		out->spawn_failed = true;
		return (false);
	}
	pid_t	pid = fork();

	if (pid < 0)
	{
		close(out_pipe[0]);
		close(out_pipe[1]);
		close(err_pipe[0]);
		close(err_pipe[1]);
		out->spawn_failed = true;
		return (false);
	}
	if (pid == 0)
	{
		/* Child: wire pipes to stdout/stderr, exec, or _exit(127) if
		 * execvp() itself fails (e.g. command not found) -- never
		 * falls back into the parent's own code past this point. */
		dup2(out_pipe[1], STDOUT_FILENO);
		dup2(err_pipe[1], STDERR_FILENO);
		close(out_pipe[0]);
		close(out_pipe[1]);
		close(err_pipe[0]);
		close(err_pipe[1]);
		std::vector<char *>	c_argv;

		for (const auto &a : argv)
			c_argv.push_back(const_cast<char *>(a.c_str()));
		c_argv.push_back(NULL);
		execvp(c_argv[0], c_argv.data());
		_exit(127);
	}
	/* Parent. */
	close(out_pipe[1]);
	close(err_pipe[1]);
	set_nonblocking(out_pipe[0]);
	set_nonblocking(err_pipe[0]);

	struct timespec	deadline;
	struct timespec	now;

	clock_gettime(CLOCK_MONOTONIC, &deadline);
	deadline.tv_sec += (timeout_seconds > 0 ? timeout_seconds : 3600);

	bool	out_open = true;
	bool	err_open = true;
	bool	timed_out = false;
	char	buf[4096];

	while (out_open || err_open)
	{
		struct pollfd	fds[2];
		int				nfds = 0;
		int				out_idx = -1;
		int				err_idx = -1;

		if (out_open)
		{
			out_idx = nfds;
			fds[nfds].fd = out_pipe[0];
			fds[nfds].events = POLLIN;
			nfds++;
		}
		if (err_open)
		{
			err_idx = nfds;
			fds[nfds].fd = err_pipe[0];
			fds[nfds].events = POLLIN;
			nfds++;
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		long	remaining_ms = (deadline.tv_sec - now.tv_sec) * 1000
				+ (deadline.tv_nsec - now.tv_nsec) / 1000000;

		if (remaining_ms <= 0)
		{
			timed_out = true;
			break ;
		}
		int	pr = poll(fds, nfds, remaining_ms > 200 ? 200 : (int)remaining_ms);

		if (pr < 0 && errno != EINTR)
			break ;
		if (out_open && out_idx >= 0 && (fds[out_idx].revents
				& (POLLIN | POLLHUP | POLLERR)))
		{
			ssize_t	n = read(out_pipe[0], buf, sizeof(buf));

			if (n > 0)
				out->stdout_output.append(buf, (size_t)n);
			else if (n == 0)
				out_open = false;
			else if (errno != EAGAIN && errno != EWOULDBLOCK)
				out_open = false;
		}
		if (err_open && err_idx >= 0 && (fds[err_idx].revents
				& (POLLIN | POLLHUP | POLLERR)))
		{
			ssize_t	n = read(err_pipe[0], buf, sizeof(buf));

			if (n > 0)
				out->stderr_output.append(buf, (size_t)n);
			else if (n == 0)
				err_open = false;
			else if (errno != EAGAIN && errno != EWOULDBLOCK)
				err_open = false;
		}
	}
	close(out_pipe[0]);
	close(err_pipe[0]);
	if (timed_out)
	{
		kill(pid, SIGKILL);
		int	status;

		waitpid(pid, &status, 0);
		out->spawn_failed = true;
		out->stderr_output += "\n(membrane: subprocess timed out and was "
			"killed)";
		return (false);
	}
	int	status;

	if (waitpid(pid, &status, 0) < 0)
	{
		out->spawn_failed = true;
		return (false);
	}
	if (WIFEXITED(status))
		out->exit_code = WEXITSTATUS(status);
	else
		out->exit_code = -1;
	return (true);
}
