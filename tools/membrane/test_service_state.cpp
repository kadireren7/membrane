#include <cstdio>
#include <cstdlib>

#include "service_state.h"
#include "test_helpers.h"

/*
 * Post-v1 product-polish, PR 1: unit tests for the shared service-
 * lifecycle probe (service_state.h's own top comment has the full real
 * v1.0.0-user-session bug this closes). Driven entirely through
 * MEMBRANE_SERVICE_PROBE_OVERRIDE (service_state.h's own documented
 * test-only hook) -- never a real systemctl/launchctl/schtasks call,
 * so this is CI-safe and hermetic regardless of whatever real per-user
 * service state happens to exist on the machine running the test.
 */

struct s_env_guard
{
	~s_env_guard(void)
	{
		unsetenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);
	}
};

static void	test_not_installed(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);

	membrane_service_probe_t	probe = membrane_probe_service();

	TEST_ASSERT(probe.manager_available,
		"the service manager itself is available");
	TEST_ASSERT(!probe.installed,
		"no MEMBRANE unit/plist/task is installed");
	TEST_ASSERT(!probe.active, "nothing installed means nothing active");
	TEST_ASSERT(membrane_service_start_should_be_blocked(probe),
		"`membrane service start` must refuse up front when nothing is "
		"installed -- never let raw systemd/launchctl/schtasks output be "
		"the primary UX for this real, common condition");
}

static void	test_installed_but_stopped(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "installed_inactive", 1);

	membrane_service_probe_t	probe = membrane_probe_service();

	TEST_ASSERT(probe.manager_available, "manager is available");
	TEST_ASSERT(probe.installed, "a real unit/plist/task exists");
	TEST_ASSERT(!probe.active, "it is not currently running");
	TEST_ASSERT(!membrane_service_start_should_be_blocked(probe),
		"`membrane service start` is exactly the right guidance once a "
		"real unit is installed but stopped -- must not be blocked");
}

static void	test_installed_and_running(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "installed_active", 1);

	membrane_service_probe_t	probe = membrane_probe_service();

	TEST_ASSERT(probe.installed && probe.active,
		"a real, currently-running installed service is reported as such");
	TEST_ASSERT(!membrane_service_start_should_be_blocked(probe),
		"an already-running service is never blocked from a redundant "
		"`start` (systemd/launchctl/schtasks themselves handle that "
		"idempotently)");
}

static void	test_manager_unavailable_never_reports_installed(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "manager_unavailable", 1);

	membrane_service_probe_t	probe = membrane_probe_service();

	TEST_ASSERT(!probe.manager_available,
		"a genuinely unusable service manager is reported honestly");
	TEST_ASSERT(!probe.installed,
		"installed is never true when the manager itself is unusable");
	/* The `start` short-circuit only ever targets the specific "manager
	 * works, nothing installed" condition -- a missing manager falls
	 * through to service_cmd.cpp's own pre-existing, already-tested
	 * IO_ERROR path instead ("could not run systemctl -- is it
	 * installed?"), never this new short-circuit. */
	TEST_ASSERT(!membrane_service_start_should_be_blocked(probe),
		"a missing service manager is not this guard's job -- it falls "
		"through to the existing IO_ERROR path instead");
}

int	main(void)
{
	test_not_installed();
	test_installed_but_stopped();
	test_installed_and_running();
	test_manager_unavailable_never_reports_installed();
	printf("test_service_state: all tests passed\n");
	return (0);
}
