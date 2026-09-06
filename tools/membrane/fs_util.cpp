#include "fs_util.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <sys/stat.h>

#ifdef _WIN32
# include <direct.h>
# include <io.h>
# include <process.h>
# include <windows.h>
#else
# include <unistd.h>
# include <limits.h>
# ifdef __APPLE__
#  include <mach-o/dyld.h>
# endif
#endif

/*
 * Mega Phase D, PR D5: every POSIX call this file makes has a real,
 * behaviorally-different Windows equivalent -- isolated into these few
 * small platform_*() wrappers rather than #ifdef'd inline at each call
 * site, so membrane_mkdir_parents()/membrane_atomic_write_file()'s own
 * logic reads identically on every platform. The one call that is NOT
 * a drop-in replacement: Windows CRT rename() fails with EEXIST if the
 * destination already exists (unlike POSIX rename()'s atomic-replace
 * contract, which this whole atomic-write pattern depends on) --
 * platform_atomic_replace() uses MoveFileExA(..., MOVEFILE_REPLACE_
 * EXISTING) on Windows instead, restoring the real atomic-replace
 * semantics this file's own callers already rely on.
 */

#ifdef _WIN32
static int	platform_mkdir(const char *path)
{
	return (_mkdir(path));
}

static bool	platform_atomic_replace(const char *from, const char *to)
{
	return (MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) != 0);
}
#else
static int	platform_mkdir(const char *path)
{
	return (mkdir(path, 0755));
}

static bool	platform_atomic_replace(const char *from, const char *to)
{
	return (rename(from, to) == 0);
}
#endif

bool	membrane_mkdir_parents(const std::string &dir,
			membrane_fs_error_t *err)
{
	std::string	partial;
	size_t		pos = 0;

	if (dir.empty())
		return (true);
	if (dir[0] == '/')
	{
		partial = "/";
		pos = 1;
	}
	while (pos <= dir.size())
	{
		size_t		next = dir.find('/', pos);
		std::string	component = dir.substr(pos, next == std::string::npos
				? std::string::npos : next - pos);

		if (!component.empty())
		{
			partial += component;
			if (platform_mkdir(partial.c_str()) != 0 && errno != EEXIST)
			{
				err->set = true;
				err->code = "IO_ERROR";
				err->message = std::string("could not create directory '")
					+ partial + "': " + strerror(errno);
				return (false);
			}
			partial += "/";
		}
		if (next == std::string::npos)
			break ;
		pos = next + 1;
	}
	return (true);
}

bool	membrane_atomic_write_file(const std::string &path,
			const std::string &content, membrane_fs_error_t *err)
{
	*err = membrane_fs_error_t();
	size_t		slash = path.find_last_of('/');
	std::string	dir = slash == std::string::npos ? "." : path.substr(0, slash);

	if (!membrane_mkdir_parents(dir, err))
		return (false);
#ifdef _WIN32
	std::string	tmp_path = path + ".tmp." + std::to_string((long long)_getpid());
#else
	std::string	tmp_path = path + ".tmp." + std::to_string((long long)getpid());
#endif
	FILE		*f = fopen(tmp_path.c_str(), "wb");

	if (f == NULL)
	{
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not create temp file '")
			+ tmp_path + "': " + strerror(errno);
		return (false);
	}
	size_t	written = fwrite(content.data(), 1, content.size(), f);
	bool	flush_ok = (fflush(f) == 0);
#ifdef _WIN32
	int		fd = _fileno(f);
	bool	sync_ok = (fd >= 0 && _commit(fd) == 0);
#else
	int		fd = fileno(f);
	bool	sync_ok = (fd >= 0 && fsync(fd) == 0);
#endif

	fclose(f);
	if (written != content.size() || !flush_ok || !sync_ok)
	{
		remove(tmp_path.c_str());
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not write '") + tmp_path + "': "
			+ strerror(errno);
		return (false);
	}
	if (!platform_atomic_replace(tmp_path.c_str(), path.c_str()))
	{
		remove(tmp_path.c_str());
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not atomically replace '")
			+ path + "': " + strerror(errno);
		return (false);
	}
	return (true);
}

int64_t	membrane_stat_mtime_ns(const struct stat &st)
{
#ifdef _WIN32
	/* MSVC's `struct stat` (via <sys/stat.h>) carries only whole-second
	 * st_mtime -- no sub-second timespec field exists at all (real,
	 * disclosed precision loss vs. Linux/macOS, not a bug: there is no
	 * more-precise field to read here). */
	return ((int64_t)st.st_mtime * 1000000000LL);
#elif defined(__APPLE__)
	return ((int64_t)st.st_mtimespec.tv_sec * 1000000000LL
		+ (int64_t)st.st_mtimespec.tv_nsec);
#else
	return ((int64_t)st.st_mtim.tv_sec * 1000000000LL
		+ (int64_t)st.st_mtim.tv_nsec);
#endif
}

std::string	membrane_resolve_home_dir(void)
{
	const char	*home = getenv("HOME");

	if (home != NULL && home[0] != '\0')
		return (home);
	const char	*userprofile = getenv("USERPROFILE");

	if (userprofile != NULL && userprofile[0] != '\0')
		return (userprofile);
	return ("");
}

bool	membrane_resolve_own_exe_path(std::string *out_path)
{
#ifdef _WIN32
	char	buf[MAX_PATH];
	DWORD	n = GetModuleFileNameA(NULL, buf, MAX_PATH);

	if (n == 0 || n == MAX_PATH)
		return (false);
	std::string	path(buf, n);

	/* Normalize to forward slashes -- every caller in this project
	 * locates the basename via find_last_of('/'); Win32 APIs accept
	 * '/' just as readily as '\\' in the paths they're later given
	 * (fopen/CreateFile), so this keeps every existing caller's own
	 * path-splitting logic platform-uniform rather than special-cased
	 * per caller. */
	for (auto &c : path)
		if (c == '\\')
			c = '/';
	*out_path = path;
	return (true);
#elif defined(__APPLE__)
	char		buf[PATH_MAX];
	uint32_t	size = sizeof(buf);

	if (_NSGetExecutablePath(buf, &size) != 0)
		return (false);
	char	resolved[PATH_MAX];

	if (realpath(buf, resolved) == NULL)
		return (false);
	*out_path = resolved;
	return (true);
#else
	char	buf[PATH_MAX];
	ssize_t	n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

	if (n < 0)
		return (false);
	buf[n] = '\0';
	*out_path = buf;
	return (true);
#endif
}
