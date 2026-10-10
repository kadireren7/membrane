#include <cstdio>
#include <cstdlib>
#include <string>

#include <unistd.h>

#include "test_helpers.h"
#include "ui.h"

/* P12/P13: color and symbols are decided in one place and are never emitted
 * where they should not be. */

static void	test_forced_color_and_symbols(void)
{
	membrane_ui_force(true, true);
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_OK, "x") == "\033[32mx\033[0m",
		"green");
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_WARN, "x") == "\033[33mx\033[0m",
		"yellow");
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_ERR, "x") == "\033[31mx\033[0m",
		"red");
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_INFO, "x") == "\033[36mx\033[0m",
		"cyan");
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_DIM, "x") == "\033[90mx\033[0m",
		"gray");
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_OK, "") == "",
		"empty text is never wrapped");
	TEST_ASSERT(membrane_ui_sym_ok().find("\xE2\x9C\x93") != std::string::npos
		&& membrane_ui_sym_err().find("\xC3\x97") != std::string::npos
		&& membrane_ui_sym_dot().find("\xE2\x97\x8F") != std::string::npos
		&& membrane_ui_sym_warn().find('!') != std::string::npos,
		"unicode symbols when allowed");
	membrane_ui_force(false, false);
	TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_ERR, "boom") == "boom",
		"no ANSI when color is off");
	TEST_ASSERT(membrane_ui_sym_ok() == "[ok]" && membrane_ui_sym_err()
			== "[x]" && membrane_ui_sym_dot() == "*"
		&& membrane_ui_sym_warn() == "!", "ASCII fallback, no escapes");
	membrane_ui_unforce();
}

static void	test_environment_decisions(void)
{
	/* the test harness runs with stdout redirected (not a TTY) */
	membrane_ui_unforce();
	setenv("NO_COLOR", "1", 1);
	membrane_ui_configure(false);
	TEST_ASSERT(!membrane_ui_color(), "NO_COLOR");
	unsetenv("NO_COLOR");
	membrane_ui_configure(true);
	TEST_ASSERT(!membrane_ui_color(), "--no-color");
	membrane_ui_configure(false);
	if (!isatty(1))		/* only meaningful when stdout is redirected */
	{
		TEST_ASSERT(!membrane_ui_color() && !membrane_ui_unicode(),
			"redirected / non-TTY output is never colored and uses ASCII "
			"symbols");
		TEST_ASSERT(membrane_ui_paint(MEMBRANE_UI_ERR, "e").find('\033')
			== std::string::npos, "no raw ANSI in redirected output");
	}
}

int	main(void)
{
	test_forced_color_and_symbols();
	test_environment_decisions();
	printf("test_ui: all tests passed\n");
	return (0);
}
