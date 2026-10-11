#include <cstdio>
#include <set>
#include <string>

#include "platform_paths.h"
#include "test_helpers.h"

/* Milestone M: per-platform path rules, over an injected env and an injected
 * directory-exists probe. Runs the same on every host. */

static std::set<std::string>	g_dirs;

static bool	fake_exists(const std::string &p)
{
	return (g_dirs.count(p) != 0);
}

static membrane_path_env_t	env(const char *home)
{
	membrane_path_env_t	e;

	e.home = home;
	return (e);
}

static void	test_linux_unchanged(void)
{
	g_dirs.clear();
	membrane_path_env_t	e = env("/home/u");

	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_LINUX, e,
		fake_exists) == "/home/u/.local/share/membrane", "linux data");
	TEST_ASSERT(membrane_platform_config_dir(MEMBRANE_PLATFORM_LINUX, e,
		fake_exists) == "/home/u/.config/membrane", "linux config");
	TEST_ASSERT(membrane_platform_state_dir(MEMBRANE_PLATFORM_LINUX, e,
		fake_exists) == "/home/u/.local/state/membrane", "linux state");
	e.xdg_data_home = "/x/data";
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_LINUX, e,
		fake_exists) == "/x/data/membrane", "XDG override");
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_LINUX,
		membrane_path_env_t(), fake_exists).empty(),
		"no home, no override: unknown, not a guess");
}

static void	test_macos(void)
{
	g_dirs.clear();
	membrane_path_env_t	e = env("/Users/u");
	const char			*native = "/Users/u/Library/Application Support/membrane";

	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_MACOS, e,
		fake_exists) == native, "fresh macOS user: Application Support");
	TEST_ASSERT(membrane_platform_config_dir(MEMBRANE_PLATFORM_MACOS, e,
		fake_exists) == native, "config in the same place");
	g_dirs.insert("/Users/u/.local/share/membrane");
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_MACOS, e,
		fake_exists) == "/Users/u/.local/share/membrane",
		"existing legacy registry keeps being used");
	g_dirs.insert(native);
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_MACOS, e,
		fake_exists) == native, "once native exists it wins");
	e.xdg_data_home = "/custom";
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_MACOS, e,
		fake_exists) == "/custom/membrane", "explicit XDG always wins");
}

static void	test_windows(void)
{
	g_dirs.clear();
	membrane_path_env_t	e = env("C:\\Users\\u");

	e.localappdata = "C:\\Users\\u\\AppData\\Local";
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_WINDOWS, e,
		fake_exists) == "C:/Users/u/AppData/Local/membrane", "windows data");
	TEST_ASSERT(membrane_platform_config_dir(MEMBRANE_PLATFORM_WINDOWS, e,
		fake_exists) == "C:/Users/u/AppData/Local/membrane/config",
		"windows config");
	TEST_ASSERT(membrane_platform_state_dir(MEMBRANE_PLATFORM_WINDOWS, e,
		fake_exists) == "C:/Users/u/AppData/Local/membrane/state",
		"windows state");
	e.localappdata.clear();
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_WINDOWS, e,
		fake_exists) == "C:/Users/u/AppData/Local/membrane",
		"no LOCALAPPDATA: derived from the profile");
	g_dirs.insert("C:/Users/u/.local/share/membrane");
	TEST_ASSERT(membrane_platform_data_dir(MEMBRANE_PLATFORM_WINDOWS, e,
		fake_exists) == "C:/Users/u/.local/share/membrane",
		"existing legacy Windows registry is kept");
}

static void	test_helpers(void)
{
	TEST_ASSERT(membrane_path_normalize("C:\\a\\b") == "C:/a/b", "normalize");
	TEST_ASSERT(membrane_path_display_for(MEMBRANE_PLATFORM_WINDOWS, "C:/a/b")
		== "C:\\a\\b", "windows display");
	TEST_ASSERT(membrane_path_display_for(MEMBRANE_PLATFORM_LINUX, "/a/b")
		== "/a/b", "posix display unchanged");
	TEST_ASSERT(membrane_path_dirname("C:\\a\\membrane.exe") == "C:\\a"
		&& membrane_path_dirname("/usr/bin/membrane") == "/usr/bin"
		&& membrane_path_dirname("membrane").empty(), "dirname, both separators");
	TEST_ASSERT(membrane_path_list_separator(MEMBRANE_PLATFORM_WINDOWS) == ';'
		&& membrane_path_list_separator(MEMBRANE_PLATFORM_MACOS) == ':',
		"PATH separator");
	TEST_ASSERT(std::string(membrane_platform_name(
		MEMBRANE_PLATFORM_WINDOWS)) == "windows", "name");
	TEST_ASSERT(std::string(membrane_arch_name()) != "", "arch");
}

int	main(void)
{
	test_linux_unchanged();
	test_macos();
	test_windows();
	test_helpers();
	printf("test_platform_paths: all tests passed\n");
	return (0);
}
