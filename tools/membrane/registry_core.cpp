#include "registry_core.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

/*
 * See registry_core.h's own top comment for the full contract. This file
 * touches the filesystem only for load()/save() themselves (open/read/
 * write/rename/mkdir) -- add()/remove()/find()/check_identity() are pure
 * functions over already-loaded in-memory state, unit-testable without
 * any real file.
 */

std::string	membrane_registry_default_path(void)
{
	const char	*xdg_data_home = getenv("XDG_DATA_HOME");

	if (xdg_data_home != NULL && xdg_data_home[0] != '\0')
		return (std::string(xdg_data_home) + "/membrane/models.json");
	const char	*home = getenv("HOME");

	if (home != NULL && home[0] != '\0')
		return (std::string(home) + "/.local/share/membrane/models.json");
	return ("");
}

std::string	membrane_registry_resolve_path(void)
{
	const char	*override_path = getenv("MEMBRANE_MODELS_PATH");

	if (override_path != NULL && override_path[0] != '\0')
		return (std::string(override_path));
	return (membrane_registry_default_path());
}

static void	set_err(membrane_registry_error_t *err, const char *code,
				const std::string &message)
{
	err->set = true;
	err->code = code;
	err->message = message;
}

/* mkdir -p semantics for the immediate parent of `path` only (registry
 * files live one directory level under XDG_DATA_HOME, never deeper) --
 * creates each missing path component in order; EEXIST on any component
 * is not an error (another process/a prior run already created it). */
static bool	mkdir_parents(const std::string &dir, membrane_registry_error_t
				*err)
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
		size_t	next = dir.find('/', pos);
		std::string	component = dir.substr(pos, next == std::string::npos
				? std::string::npos : next - pos);

		if (!component.empty())
		{
			partial += component;
			if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
			{
				set_err(err, "IO_ERROR", std::string("could not create "
					"directory '") + partial + "': " + strerror(errno));
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

static json	entry_to_json(const membrane_registry_entry_t &e)
{
	json	j;

	j["name"] = e.name;
	j["path"] = e.path;
	j["basename"] = e.basename;
	j["arch_name"] = e.arch_name;
	j["model_max_context"] = e.model_max_context;
	j["file_size_bytes"] = e.file_size_bytes;
	j["file_mtime_ns"] = e.file_mtime_ns;
	j["added_at_unix"] = e.added_at_unix;
	return (j);
}

static bool	entry_from_json(const json &j, membrane_registry_entry_t *out)
{
	if (!j.is_object() || !j.contains("name") || !j.contains("path"))
		return (false);
	out->name = j.value("name", "");
	out->path = j.value("path", "");
	out->basename = j.value("basename", "");
	out->arch_name = j.value("arch_name", "");
	out->model_max_context = j.value("model_max_context", (uint64_t)0);
	out->file_size_bytes = j.value("file_size_bytes", (uint64_t)0);
	out->file_mtime_ns = j.value("file_mtime_ns", (int64_t)0);
	out->added_at_unix = j.value("added_at_unix", (int64_t)0);
	return (out->name.size() > 0 && out->path.size() > 0);
}

bool	membrane_registry_load(const std::string &registry_path,
			membrane_registry_t *out, membrane_registry_error_t *err)
{
	*err = membrane_registry_error_t();
	out->entries.clear();
	FILE	*f = fopen(registry_path.c_str(), "rb");

	if (f == NULL)
	{
		if (errno == ENOENT)
			return (true);	/* Section 10: no file yet -- empty registry,
							 * not an error. */
		set_err(err, "IO_ERROR", std::string("could not open '")
			+ registry_path + "': " + strerror(errno));
		return (false);
	}
	std::string	content;
	char		buf[8192];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		content.append(buf, n);
	fclose(f);
	json	root;

	try
	{
		root = json::parse(content);
	}
	catch (const json::parse_error &e)
	{
		set_err(err, "PARSE_ERROR", std::string("'") + registry_path
			+ "' is not valid JSON: " + e.what());
		return (false);
	}
	if (!root.is_object() || !root.contains("schema_version")
		|| !root.contains("models") || !root["models"].is_array())
	{
		set_err(err, "PARSE_ERROR", std::string("'") + registry_path
			+ "' does not have the expected {schema_version, models[]} "
			"shape");
		return (false);
	}
	for (const auto &j : root["models"])
	{
		membrane_registry_entry_t	entry;

		if (!entry_from_json(j, &entry))
		{
			set_err(err, "PARSE_ERROR", std::string("'") + registry_path
				+ "' contains a malformed model entry");
			return (false);
		}
		out->entries.push_back(entry);
	}
	return (true);
}

bool	membrane_registry_save(const std::string &registry_path,
			const membrane_registry_t &reg, membrane_registry_error_t *err)
{
	*err = membrane_registry_error_t();
	size_t	slash = registry_path.find_last_of('/');
	std::string	dir = slash == std::string::npos ? "."
			: registry_path.substr(0, slash);

	if (!mkdir_parents(dir, err))
		return (false);
	json	root;

	root["schema_version"] = 1;
	root["models"] = json::array();
	for (const auto &e : reg.entries)
		root["models"].push_back(entry_to_json(e));
	std::string	tmp_path = registry_path + ".tmp."
			+ std::to_string((long long)getpid());
	FILE	*f = fopen(tmp_path.c_str(), "wb");

	if (f == NULL)
	{
		set_err(err, "IO_ERROR", std::string("could not create temp file "
			"'") + tmp_path + "': " + strerror(errno));
		return (false);
	}
	std::string	dump = root.dump(2);
	size_t		written = fwrite(dump.data(), 1, dump.size(), f);
	bool		flush_ok = (fflush(f) == 0);
	int			fd = fileno(f);
	bool		sync_ok = (fd >= 0 && fsync(fd) == 0);

	fclose(f);
	if (written != dump.size() || !flush_ok || !sync_ok)
	{
		unlink(tmp_path.c_str());
		set_err(err, "IO_ERROR", std::string("could not write '") + tmp_path
			+ "': " + strerror(errno));
		return (false);
	}
	if (rename(tmp_path.c_str(), registry_path.c_str()) != 0)
	{
		unlink(tmp_path.c_str());
		set_err(err, "IO_ERROR", std::string("could not atomically "
			"replace '") + registry_path + "': " + strerror(errno));
		return (false);
	}
	return (true);
}

bool	membrane_registry_add(membrane_registry_t *reg,
			const membrane_registry_entry_t &entry,
			membrane_registry_error_t *err)
{
	*err = membrane_registry_error_t();
	if (entry.name.empty())
	{
		set_err(err, "INVALID_NAME", "model name must not be empty");
		return (false);
	}
	if (entry.path.empty())
	{
		set_err(err, "INVALID_PATH", "model path must not be empty");
		return (false);
	}
	for (const auto &e : reg->entries)
	{
		if (e.name == entry.name)
		{
			set_err(err, "DUPLICATE_NAME", std::string("a model named '")
				+ entry.name + "' is already registered (path: " + e.path
				+ ") -- remove it first, or choose a different name");
			return (false);
		}
	}
	reg->entries.push_back(entry);
	return (true);
}

bool	membrane_registry_remove(membrane_registry_t *reg,
			const std::string &name, membrane_registry_error_t *err)
{
	*err = membrane_registry_error_t();
	for (size_t i = 0; i < reg->entries.size(); ++i)
	{
		if (reg->entries[i].name == name)
		{
			reg->entries.erase(reg->entries.begin() + (long)i);
			return (true);
		}
	}
	set_err(err, "NOT_FOUND", std::string("no model named '") + name
		+ "' is registered");
	return (false);
}

const membrane_registry_entry_t	*membrane_registry_find(
			const membrane_registry_t &reg, const std::string &name)
{
	for (const auto &e : reg.entries)
		if (e.name == name)
			return (&e);
	return (NULL);
}

const char	*membrane_registry_check_identity(
				const membrane_registry_entry_t &entry,
				e_membrane_registry_stat_status stat_status,
				uint64_t current_size_bytes, int64_t current_mtime_ns)
{
	if (stat_status == MEMBRANE_REGISTRY_STAT_MISSING)
		return (MEMBRANE_REGISTRY_CHECK_MISSING);
	if (stat_status == MEMBRANE_REGISTRY_STAT_ERROR)
		return (MEMBRANE_REGISTRY_CHECK_UNREADABLE);
	if (current_size_bytes != entry.file_size_bytes
		|| current_mtime_ns != entry.file_mtime_ns)
		return (MEMBRANE_REGISTRY_CHECK_MODIFIED);
	return (MEMBRANE_REGISTRY_CHECK_OK);
}
