#include "fs_util.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <sys/stat.h>
#include <unistd.h>

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
			if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
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
	std::string	tmp_path = path + ".tmp." + std::to_string((long long)getpid());
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
	int		fd = fileno(f);
	bool	sync_ok = (fd >= 0 && fsync(fd) == 0);

	fclose(f);
	if (written != content.size() || !flush_ok || !sync_ok)
	{
		unlink(tmp_path.c_str());
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not write '") + tmp_path + "': "
			+ strerror(errno);
		return (false);
	}
	if (rename(tmp_path.c_str(), path.c_str()) != 0)
	{
		unlink(tmp_path.c_str());
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not atomically replace '")
			+ path + "': " + strerror(errno);
		return (false);
	}
	return (true);
}
