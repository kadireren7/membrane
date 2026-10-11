#ifndef MEMBRANE_PLATFORM_PATHS_H
# define MEMBRANE_PLATFORM_PATHS_H

# include <string>

/*
 * Milestone M: one place that knows where MEMBRANE keeps its per-user data,
 * config and state on each OS. Pure logic over an injected environment and
 * an injected "does this directory exist" probe, so every platform's rules
 * are unit-testable on any host.
 *
 * Internal convention (unchanged): paths are kept with '/' separators on
 * every platform -- Win32 accepts them -- and membrane_path_display() turns a
 * path into the native spelling only for text shown to a person.
 *
 *   Linux    data   $XDG_DATA_HOME/membrane          | ~/.local/share/membrane
 *            config $XDG_CONFIG_HOME/membrane        | ~/.config/membrane
 *            state  $XDG_STATE_HOME/membrane         | ~/.local/state/membrane
 *   macOS    all three: ~/Library/Application Support/membrane, unless the
 *            XDG_* variable is set, or the legacy ~/.local/... directory
 *            already exists and the native one does not (existing source-
 *            build users keep their registry; nothing is moved or copied).
 *   Windows  all three: %LOCALAPPDATA%\membrane (config in ...\config,
 *            state in ...\state), same XDG override and the same
 *            legacy-directory rule against %USERPROFILE%\.local\...
 *
 * An explicit XDG_*_HOME variable wins on every platform. Linux is never
 * changed. Nothing here creates, moves or deletes anything.
 */

typedef enum e_membrane_platform
{
	MEMBRANE_PLATFORM_LINUX = 0,
	MEMBRANE_PLATFORM_MACOS,
	MEMBRANE_PLATFORM_WINDOWS
}	membrane_platform_t;

typedef struct s_membrane_path_env
{
	std::string	home;			/* HOME, else USERPROFILE */
	std::string	xdg_data_home;
	std::string	xdg_config_home;
	std::string	xdg_state_home;
	std::string	localappdata;	/* Windows only */
}	membrane_path_env_t;

typedef bool	(*membrane_dir_exists_fn)(const std::string &path);

membrane_platform_t	membrane_platform_current(void);
const char			*membrane_platform_name(membrane_platform_t p);
/* "x86_64", "arm64", ... -- compile-time, from the toolchain. */
const char			*membrane_arch_name(void);

membrane_path_env_t	membrane_path_env_from_process(void);
bool				membrane_dir_exists(const std::string &path);

/* "" when no location can be determined (no home, no override). */
std::string	membrane_platform_data_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists);
std::string	membrane_platform_config_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists);
std::string	membrane_platform_state_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists);

/* Convenience: the above for the running process. */
std::string	membrane_data_dir(void);
std::string	membrane_config_dir(void);
std::string	membrane_state_dir(void);

/* '\\' -> '/' (Windows spelling to the internal one). */
std::string	membrane_path_normalize(const std::string &p);
/* Native spelling for display: '/' -> '\\' on Windows, unchanged elsewhere.
 * The platform is a parameter so it can be tested everywhere. */
std::string	membrane_path_display_for(membrane_platform_t p,
				const std::string &path);
std::string	membrane_path_display(const std::string &path);

/* Directory part of a path accepting both separators ("" if none). */
std::string	membrane_path_dirname(const std::string &path);

/* PATH list separator for a platform: ':' or ';'. */
char		membrane_path_list_separator(membrane_platform_t p);

#endif
