#include "model_cmd.h"
#include "registry_core.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>

#include <nlohmann/json.hpp>

#include "gpu_device.h"
#include "runtime_core.h"
#include "product_cli.h"
#include "server_config.h"

using json = nlohmann::json;


static bool	stat_file(const std::string &path,
				e_membrane_registry_stat_status *status,
				uint64_t *size_bytes, int64_t *mtime_ns)
{
	struct stat	st;

	if (stat(path.c_str(), &st) != 0)
	{
		*status = (errno == ENOENT) ? MEMBRANE_REGISTRY_STAT_MISSING
			: MEMBRANE_REGISTRY_STAT_ERROR;
		*size_bytes = 0;
		*mtime_ns = 0;
		return (*status == MEMBRANE_REGISTRY_STAT_OK);
	}
	*status = MEMBRANE_REGISTRY_STAT_OK;
	*size_bytes = (uint64_t)st.st_size;
	*mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000LL
		+ (int64_t)st.st_mtim.tv_nsec;
	return (true);
}

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", code}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane: %s\n", message.c_str());
}

static int	cmd_add(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 2)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model add NAME "
			"PATH");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	const std::string	&name = args[0];
	const std::string	&raw_path = args[1];
	char				resolved[PATH_MAX];

	if (realpath(raw_path.c_str(), resolved) == NULL)
	{
		print_err(want_json, "NOT_FOUND", std::string("'") + raw_path
			+ "' does not exist or is not accessible: " + strerror(errno));
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	std::string	canonical(resolved);
	e_membrane_registry_stat_status	stat_status;
	uint64_t							size_bytes;
	int64_t								mtime_ns;

	if (!stat_file(canonical, &stat_status, &size_bytes, &mtime_ns))
	{
		print_err(want_json, "UNREADABLE", std::string("'") + canonical
			+ "' could not be read");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	membrane_gpu_model_estimate_t	est;

	if (!membrane_gpu_estimate_model(canonical.c_str(), &est))
	{
		print_err(want_json, "INVALID_GGUF", std::string("'") + canonical
			+ "' is not a readable GGUF file (no recognizable tensor "
			"data)");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	membrane_registry_entry_t	entry;

	entry.name = name;
	entry.path = canonical;
	const char	*base = membrane_runtime_safe_basename(canonical.c_str());

	entry.basename = base != NULL ? base : "";
	entry.arch_name = est.hparams_available ? est.arch_name : "";
	entry.model_max_context = est.model_max_context_available
		? est.model_max_context : 0;
	entry.file_size_bytes = size_bytes;
	entry.file_mtime_ns = mtime_ns;
	entry.added_at_unix = (int64_t)time(NULL);

	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	if (registry_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_DATA_HOME nor HOME "
			"is set -- cannot locate the registry file");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_load(registry_path, &reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_add(&reg, entry, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!membrane_registry_save(registry_path, reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (want_json)
	{
		json	j;

		j["ok"] = true;
		j["name"] = entry.name;
		j["path"] = entry.path;
		j["architecture"] = entry.arch_name;
		j["model_max_context"] = entry.model_max_context;
		printf("%s\n", j.dump().c_str());
	}
	else
		printf("Added '%s' -> %s%s%s\n", name.c_str(), canonical.c_str(),
			entry.arch_name.empty() ? "" : " (architecture: ",
			entry.arch_name.empty() ? "" : (entry.arch_name + ")").c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_remove(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model remove "
			"NAME");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	if (!membrane_registry_load(registry_path, &reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_remove(&reg, args[0], &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!membrane_registry_save(registry_path, reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (want_json)
		printf("{\"ok\":true,\"removed\":\"%s\"}\n", args[0].c_str());
	else
		printf("Removed '%s'\n", args[0].c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

/* Shared by `list` (every entry) and `inspect` (one entry, more detail) --
 * a real stat() identity check every time, never assumed still valid from
 * whatever was cached at `add` time (Section 13). */
static json	entry_status_json(const membrane_registry_entry_t &e)
{
	e_membrane_registry_stat_status	stat_status;
	uint64_t							size_bytes;
	int64_t								mtime_ns;

	stat_file(e.path, &stat_status, &size_bytes, &mtime_ns);
	const char	*status = membrane_registry_check_identity(e, stat_status,
			size_bytes, mtime_ns);
	json		j;

	j["name"] = e.name;
	j["path"] = e.path;
	j["basename"] = e.basename;
	j["architecture"] = e.arch_name;
	j["model_max_context"] = e.model_max_context;
	j["status"] = status;
	return (j);
}

static int	cmd_list(const std::vector<std::string> &args, bool want_json)
{
	(void)args;
	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	if (!membrane_registry_load(registry_path, &reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (want_json)
	{
		json	arr = json::array();

		for (const auto &e : reg.entries)
			arr.push_back(entry_status_json(e));
		printf("%s\n", arr.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (reg.entries.empty())
	{
		printf("No models registered. Add one: membrane model add NAME "
			"PATH\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	for (const auto &e : reg.entries)
	{
		json		j = entry_status_json(e);
		std::string	status = j["status"];

		printf("%-20s %-10s %s%s\n", e.name.c_str(),
			e.arch_name.empty() ? "?" : e.arch_name.c_str(), e.path.c_str(),
			status == MEMBRANE_REGISTRY_CHECK_OK ? ""
				: (std::string(" [") + status + "]").c_str());
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_inspect(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model inspect "
			"NAME");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	err;

	if (!membrane_registry_load(registry_path, &reg, &err))
	{
		print_err(want_json, err.code, err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	const membrane_registry_entry_t	*entry = membrane_registry_find(reg,
			args[0]);

	if (entry == NULL)
	{
		print_err(want_json, "NOT_FOUND", std::string("no model named '")
			+ args[0] + "' is registered");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	json		j = entry_status_json(*entry);
	std::string	status = j["status"];

	/* Section 13: a MODIFIED (stale) entry gets a real rescan here --
	 * never continues to report the OLD cached architecture/context as
	 * if it were still current. */
	if (status == MEMBRANE_REGISTRY_CHECK_MODIFIED)
	{
		membrane_gpu_model_estimate_t	est;

		if (membrane_gpu_estimate_model(entry->path.c_str(), &est))
		{
			j["architecture"] = est.hparams_available
				? std::string(est.arch_name) : std::string("");
			j["model_max_context"] = est.model_max_context_available
				? est.model_max_context : 0;
			j["rescanned"] = true;
		}
		else
			j["rescan_failed"] = true;
	}
	if (want_json)
		printf("%s\n", j.dump().c_str());
	else
	{
		printf("Model: %s\n", entry->name.c_str());
		printf("  path:         %s\n", entry->path.c_str());
		printf("  status:       %s\n", status.c_str());
		printf("  architecture: %s\n",
			std::string(j["architecture"]).empty() ? "(unknown)"
				: std::string(j["architecture"]).c_str());
		printf("  max context:  %llu\n",
			(unsigned long long)j["model_max_context"].get<uint64_t>());
		if (j.contains("rescanned"))
			printf("  note:         file changed since it was added -- "
				"metadata above is freshly rescanned\n");
		else if (j.contains("rescan_failed"))
			printf("  note:         file changed since it was added, and "
				"could not be rescanned\n");
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

/* Mega Phase B, PR B1, Section 9 of the task: sets the persistent server
 * config's default_model -- a fallback the server MAY use for a chat
 * request that omits "model" (Section 21 of Mega Phase A's own server.md
 * still requires "model" when no default is configured; this only adds
 * an optional fallback, never forces one -- "Server may start: healthy,
 * no model loaded" stays true even with a default_model configured,
 * since this never proactively loads anything). */
static int	cmd_use(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model use "
			"NAME");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	reg_err;

	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		print_err(want_json, reg_err.code, reg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (membrane_registry_find(reg, args[0]) == NULL)
	{
		print_err(want_json, "NOT_FOUND", std::string("no model named '")
			+ args[0] + "' is registered -- add it first with `membrane "
			"model add`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;

	if (config_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_server_config_load(config_path, &cfg, &cfg_err))
	{
		print_err(want_json, cfg_err.code, cfg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	cfg.default_model = args[0];
	if (!membrane_server_config_save(config_path, cfg, &cfg_err))
	{
		print_err(want_json, cfg_err.code, cfg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (want_json)
		printf("{\"ok\":true,\"default_model\":\"%s\"}\n", args[0].c_str());
	else
		printf("Default model set to '%s'. Takes effect on the next "
			"`membrane serve`/service restart.\n", args[0].c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_model_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	if (args.empty())
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model "
			"add|remove|list|inspect|use ...");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::vector<std::string>	rest(args.begin() + 1, args.end());

	if (args[0] == "add")
		return (cmd_add(rest, want_json));
	if (args[0] == "remove")
		return (cmd_remove(rest, want_json));
	if (args[0] == "list")
		return (cmd_list(rest, want_json));
	if (args[0] == "inspect")
		return (cmd_inspect(rest, want_json));
	if (args[0] == "use")
		return (cmd_use(rest, want_json));
	print_err(want_json, "CLI_ERROR", std::string("unknown subcommand '")
		+ args[0] + "' -- expected add|remove|list|inspect|use");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
