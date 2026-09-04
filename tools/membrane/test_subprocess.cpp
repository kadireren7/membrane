#include <cstdio>
#include <string>
#include <vector>

#include "subprocess.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B1: unit tests for subprocess.h's fork()+execvp()
 * runner. The injection-safety tests are the important ones here (Section
 * 11 of the task: "No shell injection") -- they prove a value containing
 * shell metacharacters reaches the child process VERBATIM, as a single
 * argv element, never interpreted by a shell.
 */

static void	test_basic_success_captures_stdout(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"echo", "hello"}, &result) == true,
		"running a real, simple command succeeds");
	TEST_ASSERT(!result.spawn_failed, "spawn_failed is false");
	TEST_ASSERT(result.exit_code == 0, "exit_code is 0");
	TEST_ASSERT(result.stdout_output == "hello\n",
		"stdout is captured exactly");
}

static void	test_exit_code_propagates(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"sh", "-c", "exit 7"}, &result)
		== true, "running a command that exits nonzero still returns "
		"true (the RUN itself succeeded; the command's own result is in "
		"exit_code)");
	TEST_ASSERT(result.exit_code == 7, "the real exit code is captured");
}

static void	test_stderr_captured_separately(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"sh", "-c",
		"echo out; echo err >&2"}, &result) == true, "command runs");
	TEST_ASSERT(result.stdout_output == "out\n", "stdout only has 'out'");
	TEST_ASSERT(result.stderr_output == "err\n",
		"stderr only has 'err' -- the two streams are never merged");
}

static void	test_command_not_found_is_exit_127_not_a_crash(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess(
		{"membrane-this-binary-does-not-exist-anywhere"}, &result) == true,
		"a nonexistent command still returns true from the RUN's own "
		"perspective -- execvp() failing inside the child is reported via "
		"exit_code 127, matching every shell's own convention, never a "
		"crash of the parent");
	TEST_ASSERT(result.exit_code == 127,
		"exit_code is 127, execvp()'s own child-side failure convention");
}

static void	test_empty_argv_is_spawn_failed(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({}, &result) == false,
		"an empty argv is rejected before any fork()/exec() is attempted");
	TEST_ASSERT(result.spawn_failed, "spawn_failed is true");
}

/* Section 11's central guarantee: argv is passed directly to execvp(),
 * with no shell parsing step -- a value containing shell metacharacters
 * ($(), ;, |, &&, backticks, quotes) is delivered to the child as a
 * single, literal, UNINTERPRETED argument. `echo` merely reflects its
 * argv[1] back on stdout; if a shell were involved, $(id) would have been
 * expanded and the semicolon would have started a second command. */
static void	test_argv_is_never_shell_interpreted(void)
{
	std::string	dangerous = "$(id); echo pwned && rm -rf / #`whoami`";
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"echo", dangerous}, &result)
		== true, "command runs");
	TEST_ASSERT(result.stdout_output == dangerous + "\n",
		"the dangerous string is echoed back byte-for-byte -- no "
		"substitution, no command chaining, no glob expansion occurred, "
		"proving no shell ever parsed it");
}

/* A value containing a literal newline is still one argv element -- a
 * shell command LINE could never carry this safely without careful
 * escaping, but execvp() has no such concept at all. */
static void	test_argv_element_with_embedded_newline(void)
{
	std::string	value = "line one\nline two; rm -rf /tmp/should-not-run";
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"echo", value}, &result) == true,
		"command runs");
	TEST_ASSERT(result.stdout_output == value + "\n",
		"the embedded newline and semicolon are both part of the single "
		"literal argument, never treated as a second shell command");
}

static void	test_timeout_kills_and_reports_spawn_failed(void)
{
	membrane_subprocess_result_t	result;

	TEST_ASSERT(membrane_run_subprocess({"sleep", "5"}, &result, 1)
		== false, "a command that outlives its timeout is killed and the "
		"run itself is reported as failed");
	TEST_ASSERT(result.spawn_failed,
		"spawn_failed is true for a timeout (no real exit_code to trust)");
}

int	main(void)
{
	test_basic_success_captures_stdout();
	test_exit_code_propagates();
	test_stderr_captured_separately();
	test_command_not_found_is_exit_127_not_a_crash();
	test_empty_argv_is_spawn_failed();
	test_argv_is_never_shell_interpreted();
	test_argv_element_with_embedded_newline();
	test_timeout_kills_and_reports_spawn_failed();
	printf("test_subprocess: all tests passed\n");
	return (0);
}
