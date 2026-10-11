#include "platform_paths.h"

#include <cstdlib>

#include <sys/stat.h>

membrane_platform_t	membrane_platform_current(void)
{
#if defined(_WIN32)
	return (MEMBRANE_PLATFORM_WINDOWS);
#elif defined(__APPLE__)
	return (MEMBRANE_PLATFORM_MACOS);
#else
	return (MEMBRANE_PLATFORM_LINUX);
#endif
}

const char	*membrane_platform_name(membrane_platform_t p)
{
	switch (p)
	{
		case MEMBRANE_PLATFORM_MACOS: return ("macos");
		case MEMBRANE_PLATFORM_WINDOWS: return ("windows");
		default: return ("linux");
	}
}

const char	*membrane_arch_name(void)
{
#if defined(__aarch64__) || defined(_M_ARM64)
	return ("arm64");
#elif defined(__x86_64__) || defined(_M_X64)
	return ("x86_64");
#elif defined(__i386__) || defined(_M_IX86)
	return ("x86");
#else
	return ("unknown");
#endif
}

static std::string	getenv_str(const char *name)
{
	const char	*v = getenv(name);

	return (v != NULL ? std::string(v) : std::string());
}

std::string	membrane_path_normalize(const std::string &p)
{
	std::string	out = p;

	for (size_t i = 0; i < out.size(); ++i)
		if (out[i] == '\\')
			out[i] = '/';
	return (out);
}

membrane_path_env_t	membrane_path_env_from_process(void)
{
	membrane_path_env_t	e;

	e.home = getenv_str("HOME");
	if (e.home.empty())
		e.home = getenv_str("USERPROFILE");
	e.xdg_data_home = getenv_str("XDG_DATA_HOME");
	e.xdg_config_home = getenv_str("XDG_CONFIG_HOME");
	e.xdg_state_home = getenv_str("XDG_STATE_HOME");
	e.localappdata = getenv_str("LOCALAPPDATA");
	return (e);
}

bool	membrane_dir_exists(const std::string &path)
{
	struct stat	st;

	return (stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR);
}

typedef enum e_kind { KIND_DATA, KIND_CONFIG, KIND_STATE } kind_t;

static std::string	resolve(kind_t kind, membrane_platform_t p,
						const membrane_path_env_t &env,
						membrane_dir_exists_fn exists)
{
	const std::string	&xdg = kind == KIND_DATA ? env.xdg_data_home
			: kind == KIND_CONFIG ? env.xdg_config_home : env.xdg_state_home;
	const char			*legacy_sub = kind == KIND_DATA ? "/.local/share"
			: kind == KIND_CONFIG ? "/.config" : "/.local/state";
	std::string			home = membrane_path_normalize(env.home);

	if (!xdg.empty())
		return (membrane_path_normalize(xdg) + "/membrane");
	if (p == MEMBRANE_PLATFORM_LINUX)
		return (home.empty() ? "" : home + legacy_sub + "/membrane");

	std::string	legacy = home.empty() ? "" : home + legacy_sub + "/membrane";
	std::string	native;

	if (p == MEMBRANE_PLATFORM_MACOS)
	{
		if (!home.empty())
			native = home + "/Library/Application Support/membrane";
	}
	else
	{
		std::string	base = membrane_path_normalize(env.localappdata);

		if (base.empty() && !home.empty())
			base = home + "/AppData/Local";
		if (!base.empty())
		{
			native = base + "/membrane";
			if (kind == KIND_CONFIG)
				native += "/config";
			else if (kind == KIND_STATE)
				native += "/state";
		}
	}
	if (native.empty())
		return (legacy);
	/* Existing users keep their data: the legacy directory wins only when
	 * it is already there and the native one is not. Never moved/copied. */
	if (!legacy.empty() && exists != NULL && exists(legacy)
		&& !exists(native))
		return (legacy);
	return (native);
}

std::string	membrane_platform_data_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists)
{
	return (resolve(KIND_DATA, p, env, exists));
}

std::string	membrane_platform_config_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists)
{
	return (resolve(KIND_CONFIG, p, env, exists));
}

std::string	membrane_platform_state_dir(membrane_platform_t p,
				const membrane_path_env_t &env, membrane_dir_exists_fn exists)
{
	return (resolve(KIND_STATE, p, env, exists));
}

std::string	membrane_data_dir(void)
{
	return (membrane_platform_data_dir(membrane_platform_current(),
		membrane_path_env_from_process(), membrane_dir_exists));
}

std::string	membrane_config_dir(void)
{
	return (membrane_platform_config_dir(membrane_platform_current(),
		membrane_path_env_from_process(), membrane_dir_exists));
}

std::string	membrane_state_dir(void)
{
	return (membrane_platform_state_dir(membrane_platform_current(),
		membrane_path_env_from_process(), membrane_dir_exists));
}

std::string	membrane_path_display_for(membrane_platform_t p,
				const std::string &path)
{
	std::string	out = path;

	if (p != MEMBRANE_PLATFORM_WINDOWS)
		return (out);
	for (size_t i = 0; i < out.size(); ++i)
		if (out[i] == '/')
			out[i] = '\\';
	return (out);
}

std::string	membrane_path_display(const std::string &path)
{
	return (membrane_path_display_for(membrane_platform_current(), path));
}

std::string	membrane_path_dirname(const std::string &path)
{
	size_t	i = path.find_last_of("/\\");

	if (i == std::string::npos)
		return ("");
	return (path.substr(0, i));
}

char	membrane_path_list_separator(membrane_platform_t p)
{
	return (p == MEMBRANE_PLATFORM_WINDOWS ? ';' : ':');
}
