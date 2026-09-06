#include "subprocess.h"

#include <cerrno>
#include <cstring>
#include <ctime>

#ifdef _WIN32
# include <thread>
# include <windows.h>
#else
# include <fcntl.h>
# include <poll.h>
# include <signal.h>
# include <sys/wait.h>
# include <unistd.h>
#endif

/*
 * See subprocess.h's own top comment. On Linux/macOS: reads stdout/
 * stderr concurrently via poll() (never sequential blocking reads on
 * two pipes at once -- that risks a real deadlock if the child fills
 * one pipe's kernel buffer while this process is blocked reading the
 * other) with a wall-clock deadline; a child that outlives it is
 * SIGKILL'd, waited for (never left a zombie), and reported as a
 * timeout via spawn_failed.
 */

#ifdef _WIN32

/*
 * Mega Phase D, PR D5: CreateProcessA needs a single command-line
 * string, not an argv array -- this is the standard, well-documented
 * Win32 quoting algorithm (the same one the Windows CRT itself uses to
 * build argv from a command line, run in reverse): wrap an argument in
 * double quotes if it is empty or contains a space/tab/quote, doubling
 * any run of backslashes that is immediately followed by a quote (or
 * is at the very end of a quoted argument), and escaping the quote
 * itself with a backslash. Getting this wrong either breaks arguments
 * containing spaces (a real path like "C:\Program Files\membrane\...")
 * or -- far worse -- lets a backslash-quote sequence terminate the
 * argument early, which is exactly the class of bug this whole
 * function exists to get right once, correctly, rather than leave
 * every call site to improvise its own.
 */
static std::string	win32_quote_argument(const std::string &arg)
{
	bool	needs_quotes = arg.empty();

	for (char c : arg)
		if (c == ' ' || c == '\t' || c == '"')
			needs_quotes = true;
	if (!needs_quotes)
		return (arg);
	std::string	out = "\"";
	size_t		backslashes = 0;

	for (char c : arg)
	{
		if (c == '\\')
		{
			++backslashes;
			continue ;
		}
		if (c == '"')
		{
			out.append(backslashes * 2 + 1, '\\');
			backslashes = 0;
			out += '"';
			continue ;
		}
		out.append(backslashes, '\\');
		backslashes = 0;
		out += c;
	}
	out.append(backslashes * 2, '\\');
	out += '"';
	return (out);
}

static std::string	win32_build_command_line(
				const std::vector<std::string> &argv)
{
	std::string	out;

	for (size_t i = 0; i < argv.size(); ++i)
	{
		if (i > 0)
			out += ' ';
		out += win32_quote_argument(argv[i]);
	}
	return (out);
}

/* Reads a pipe to EOF into *out -- run on its own std::thread (one per
 * pipe) so stdout/stderr are drained concurrently, same reason the
 * POSIX path uses poll() on both fds at once: a child that fills one
 * pipe's kernel buffer while this process only reads the other would
 * otherwise deadlock. */
static void	win32_read_pipe_to_string(HANDLE h, std::string *out)
{
	char	buf[4096];
	DWORD	n;

	while (ReadFile(h, buf, sizeof(buf), &n, NULL) && n > 0)
		out->append(buf, n);
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

	SECURITY_ATTRIBUTES	sa;

	sa.nLength = sizeof(sa);
	sa.bInheritHandle = TRUE;
	sa.lpSecurityDescriptor = NULL;

	HANDLE	out_read;
	HANDLE	out_write;
	HANDLE	err_read;
	HANDLE	err_write;

	if (!CreatePipe(&out_read, &out_write, &sa, 0)
		|| !CreatePipe(&err_read, &err_write, &sa, 0))
	{
		out->spawn_failed = true;
		return (false);
	}
	/* The PARENT's own read-end handles must never be inherited by the
	 * child -- only the write ends (wired to the child's stdout/stderr
	 * below) are meant to cross into it. Leaving the read ends
	 * inheritable would leak a handle the child could otherwise hold
	 * open, which would (among other problems) prevent this process's
	 * own ReadFile() loop from ever seeing real EOF after the child
	 * exits. */
	SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
	SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0);

	STARTUPINFOA	si;

	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdOutput = out_write;
	si.hStdError = err_write;
	si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

	PROCESS_INFORMATION	pi;

	ZeroMemory(&pi, sizeof(pi));

	std::string	cmdline = win32_build_command_line(argv);
	std::vector<char>	cmdline_buf(cmdline.begin(), cmdline.end());

	cmdline_buf.push_back('\0');	/* CreateProcessA may WRITE to this
									 * buffer (in-place argument
									 * splitting) -- never pass a
									 * std::string's own internal buffer
									 * or a string literal directly. */
	BOOL	spawned = CreateProcessA(NULL, cmdline_buf.data(), NULL, NULL,
			TRUE, 0, NULL, NULL, &si, &pi);

	/* These two write-end handles belong to the CHILD's own stdout/
	 * stderr now (duplicated into it by CreateProcessA's own handle
	 * inheritance) -- closing the parent's copies here (whether or not
	 * spawning succeeded) is what lets win32_read_pipe_to_string()'s
	 * ReadFile() loop below ever see real EOF once the child exits;
	 * holding them open in the parent would hang that loop forever. */
	CloseHandle(out_write);
	CloseHandle(err_write);
	if (!spawned)
	{
		CloseHandle(out_read);
		CloseHandle(err_read);
		out->spawn_failed = true;
		return (false);
	}

	std::thread	out_reader(win32_read_pipe_to_string, out_read,
			&out->stdout_output);
	std::thread	err_reader(win32_read_pipe_to_string, err_read,
			&out->stderr_output);

	DWORD	wait_ms = (DWORD)((timeout_seconds > 0 ? timeout_seconds : 3600)
			* 1000);
	DWORD	wait_result = WaitForSingleObject(pi.hProcess, wait_ms);

	if (wait_result != WAIT_OBJECT_0)
	{
		TerminateProcess(pi.hProcess, 1);
		WaitForSingleObject(pi.hProcess, INFINITE);
		out_reader.join();
		err_reader.join();
		CloseHandle(out_read);
		CloseHandle(err_read);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		out->spawn_failed = true;
		out->stderr_output += "\n(membrane: subprocess timed out and was "
			"killed)";
		return (false);
	}
	out_reader.join();
	err_reader.join();
	CloseHandle(out_read);
	CloseHandle(err_read);

	DWORD	exit_code = 0;

	GetExitCodeProcess(pi.hProcess, &exit_code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	out->exit_code = (int)exit_code;
	return (true);
}

#else

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

#endif	/* _WIN32 */
