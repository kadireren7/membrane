#include <cstdio>
#include <cstdlib>

#include "service_cmd.h"
#include "service_state.h"
#include "product_cli.h"
#include "test_helpers.h"

/*
 * Post-v1 product-polish, PR 1: regression coverage for the real
 * v1.0.0 user-session bug where `membrane service start` let systemd's
 * raw "Unit membrane.service not found." be the primary UX whenever the
 * background service had never been installed. Driven entirely through
 * MEMBRANE_SERVICE_PROBE_OVERRIDE (service_state.h's own test-only
 * hook) -- guarantees the real systemctl/launchctl/schtasks binary is
 * NEVER invoked for the "not installed" case (the short-circuit in
 * service_cmd.cpp's own dispatch returns before that call), so this
 * stays hermetic and never touches whatever real per-user service state
 * exists on the machine running the test. The complementary "installed"
 * pass-through path is intentionally NOT exercised here -- it would
 * require a real systemctl/launchctl/schtasks invocation against
 * whatever service state genuinely exists on the host, which is exactly
 * the real, un-mockable side effect this test suite must not risk.
 */

struct s_env_guard
{
	~s_env_guard(void)
	{
		unsetenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);
	}
};

static void	test_start_refuses_cleanly_when_not_installed(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);

	int	rc = membrane_service_cmd_dispatch({"start"}, false);

	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR,
		"`membrane service start` refuses with a clean MEMBRANE error "
		"when nothing is installed, instead of ever shelling out to the "
		"real service manager and surfacing its raw error text");
}

static void	test_start_refuses_in_json_mode_too(void)
{
	s_env_guard	guard;

	setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);

	int	rc = membrane_service_cmd_dispatch({"start"}, true);

	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR,
		"the same not-installed refusal holds in --json mode");
}

int	main(void)
{
	test_start_refuses_cleanly_when_not_installed();
	test_start_refuses_in_json_mode_too();
	printf("test_service_cmd: all tests passed\n");
	return (0);
}
