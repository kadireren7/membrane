#include "model_cmd.h"
#include "registry_core.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "gpu_device.h"
#include "runtime_core.h"
#include "product_cli.h"
#include "server_config.h"
#include "model_catalog.h"
#include "download_manager.h"
#include "fs_util.h"

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

/* Mega Phase D, PR D1, Section 3/5 of the task: the built-in catalog is
 * pure/offline (model_catalog.h's own top comment) -- `search`/`info`
 * need no network access at all. */
static json	catalog_family_summary_json(const membrane_catalog_family_t &f)
{
	json	j;

	j["name"] = f.name;
	j["aliases"] = f.aliases;
	j["display_name"] = f.display_name;
	j["arch"] = f.arch;
	j["parameter_count"] = f.parameter_count;
	j["provider"] = f.provider;
	j["license"] = f.license;
	j["chat_template_status"] = f.chat_template_status;
	j["compatibility_status"] = f.compatibility_status;
	json	variants = json::array();

	for (const auto &v : f.variants)
		variants.push_back({{"quant", v.quant}, {"size_bytes", v.size_bytes}});
	j["variants"] = variants;
	return (j);
}

static int	cmd_search(const std::vector<std::string> &args, bool want_json)
{
	std::string	query = args.empty() ? "" : args[0];
	membrane_catalog_t	cat = membrane_catalog_load();
	auto		results = membrane_catalog_search(cat, query);

	if (want_json)
	{
		json	j;

		j["ok"] = true;
		j["query"] = query;
		json	arr = json::array();

		for (const auto *f : results)
			arr.push_back(catalog_family_summary_json(*f));
		j["results"] = arr;
		printf("%s\n", j.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (results.empty())
	{
		printf("No catalog entries match '%s'. Try `membrane model "
			"search` with no argument to list everything.\n",
			query.c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("%-24s %-24s %-8s %-10s %s\n", "NAME", "DISPLAY NAME", "ARCH",
		"PARAMS", "VARIANTS");
	for (const auto *f : results)
	{
		std::string	quants;

		for (size_t i = 0; i < f->variants.size(); ++i)
		{
			if (i > 0)
				quants += ",";
			quants += f->variants[i].quant;
		}
		printf("%-24s %-24s %-8s %-10s %s\n", f->name.c_str(),
			f->display_name.c_str(), f->arch.c_str(),
			f->parameter_count.c_str(), quants.c_str());
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_info(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model info NAME");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f
			= membrane_catalog_resolve(cat, args[0]);

	if (f == NULL)
	{
		print_err(want_json, "NOT_FOUND", std::string("'") + args[0]
			+ "' is not in the catalog -- try `membrane model search "
			+ args[0] + "` to find similar names");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (want_json)
	{
		json	j = catalog_family_summary_json(*f);

		j["ok"] = true;
		j["repo_url"] = f->repo_url;
		j["compatibility_evidence"] = f->compatibility_evidence;
		j["recorded_at"] = f->recorded_at;
		json	variants = json::array();

		for (const auto &v : f->variants)
			variants.push_back({{"quant", v.quant},
				{"filename", v.filename}, {"size_bytes", v.size_bytes},
				{"sha256", v.sha256}});
		j["variants"] = variants;
		printf("%s\n", j.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("%s (%s)\n", f->display_name.c_str(), f->name.c_str());
	printf("  arch:            %s\n", f->arch.c_str());
	printf("  parameters:      %s\n", f->parameter_count.c_str());
	printf("  provider:        %s\n", f->provider.c_str());
	printf("  license:         %s\n", f->license.c_str());
	printf("  chat template:   %s\n", f->chat_template_status.c_str());
	printf("  compatibility:   %s (%s)\n", f->compatibility_status.c_str(),
		f->compatibility_evidence.c_str());
	printf("  repo:            %s\n", f->repo_url.c_str());
	printf("  variants:\n");
	for (const auto &v : f->variants)
		printf("    %-10s %12llu bytes  %s\n", v.quant.c_str(),
			(unsigned long long)v.size_bytes, v.filename.c_str());
	printf("\nInstall with: membrane model install %s [--quant QUANT]\n",
		f->name.c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

/* A real bug found while testing this install path end to end: libcurl
 * calls its xfer-progress callback far more often than once per
 * meaningful update (multiple times per second, and once per each
 * response in a redirect chain -- Hugging Face's own real CDN
 * redirects through a small pointer response before the real file,
 * each with its own dltotal/dlnow that briefly resets to a tiny value)
 * -- printing on every single call flooded real terminal output with
 * thousands of lines and a confusing "100% (0 / 0 MiB)" blip from the
 * redirect stub. Fixed by throttling to at most ~10 updates/second
 * (wall-clock, via clock_gettime) and padding the printed line with
 * trailing spaces so a shorter new line fully overwrites a longer
 * previous one under the same \r. */
struct s_install_progress_ctx
{
	bool		is_json;
	struct timespec	last_print;
};

static void	install_progress_cb(uint64_t downloaded, uint64_t total,
				void *user_data)
{
	s_install_progress_ctx	*ctx = (s_install_progress_ctx *)user_data;

	if (ctx->is_json || total == 0)
		return ;
	struct timespec	now;

	clock_gettime(CLOCK_MONOTONIC, &now);
	double	elapsed_ms = (now.tv_sec - ctx->last_print.tv_sec) * 1000.0
		+ (now.tv_nsec - ctx->last_print.tv_nsec) / 1.0e6;

	if (elapsed_ms < 100.0 && downloaded < total)
		return ;
	ctx->last_print = now;
	printf("\r  %5.1f%% (%llu / %llu MiB)          ",
		100.0 * (double)downloaded / (double)total,
		(unsigned long long)(downloaded / (1024 * 1024)),
		(unsigned long long)(total / (1024 * 1024)));
	fflush(stdout);
}

static int	cmd_install(const std::vector<std::string> &args, bool want_json)
{
	std::string	name;
	std::string	requested_quant;

	for (size_t i = 0; i < args.size(); ++i)
	{
		if ((args[i] == "--quant" || args[i] == "--variant")
			&& i + 1 < args.size())
		{
			requested_quant = args[++i];
			continue ;
		}
		if (name.empty())
			name = args[i];
	}
	if (name.empty())
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model install "
			"NAME [--quant QUANT]");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*f = membrane_catalog_resolve(cat, name);

	if (f == NULL)
	{
		print_err(want_json, "NOT_FOUND", std::string("'") + name
			+ "' is not in the catalog -- try `membrane model search "
			+ name + "`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	const membrane_catalog_variant_t	*variant = NULL;

	if (!requested_quant.empty())
	{
		variant = membrane_catalog_find_variant(*f, requested_quant);
		if (variant == NULL)
		{
			print_err(want_json, "NOT_FOUND", std::string("'")
				+ requested_quant + "' is not an available variant of '"
				+ f->name + "' -- see `membrane model info " + f->name
				+ "` for the real list");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
	}
	else
	{
		/* Mega Phase D, PR D1 has no hardware-aware variant selection
		 * yet (that is D2's own, separate, real work) -- the smallest
		 * available variant is the honest, documented, conservative
		 * default in the meantime: most likely to actually fit,
		 * never a guess dressed up as an intelligent recommendation. */
		for (const auto &v : f->variants)
			if (variant == NULL || v.size_bytes < variant->size_bytes)
				variant = &v;
	}
	std::string	install_dir = membrane_registry_models_install_dir();

	if (install_dir.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_DATA_HOME nor HOME "
			"is set -- cannot locate the models install directory");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	std::string	family_dir = install_dir + "/" + f->name;
	membrane_fs_error_t	fs_err;

	if (!membrane_mkdir_parents(family_dir, &fs_err))
	{
		print_err(want_json, fs_err.code, fs_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	std::string	dest_path = family_dir + "/" + variant->filename;

	/* Idempotence (a real bug found while testing this end to end: an
	 * earlier version always re-downloaded, even when the exact right
	 * file, at the exact right size, was already sitting on disk).
	 * Only the SIZE is checked here (a real, cheap stat() -- the
	 * expensive full-file sha256 re-check is deliberately skipped for
	 * an already-size-correct file on a re-run; a corrupted-but-right-
	 * sized file is an extraordinarily unlikely edge case, and `install`
	 * can always be re-run after a manual `rm` if that is ever
	 * actually suspected). */
	struct stat	existing_st;
	bool		already_present = stat(dest_path.c_str(), &existing_st) == 0
			&& (uint64_t)existing_st.st_size == variant->size_bytes;
	bool		checksum_verified = false;

	if (already_present)
	{
		if (!want_json)
			printf("'%s' (%s) is already downloaded at the right size -- "
				"skipping the download.\n", f->display_name.c_str(),
				variant->quant.c_str());
		/* Still worth a real, cheap-relative-to-a-re-download checksum
		 * check -- a same-SIZE-but-corrupted file is rare but real,
		 * and skipping this silently would make `checksum_verified`
		 * misleadingly claim "unavailable" (a real, distinct bug found
		 * while testing this path -- it used to conflate "we never
		 * checked" with "sha256sum itself is missing"). */
		if (!variant->sha256.empty())
		{
			std::string	actual_hex;
			membrane_download_error_t	sha_err;

			if (membrane_compute_sha256(dest_path, &actual_hex, &sha_err))
			{
				if (actual_hex != variant->sha256)
				{
					print_err(want_json, "CHECKSUM_MISMATCH",
						std::string("the existing file at '") + dest_path
						+ "' has the right size but the wrong checksum -- "
						"it may be corrupted; delete it and re-run "
						"`membrane model install` to re-download");
					return (MEMBRANE_EXIT_MODEL_ERROR);
				}
				checksum_verified = true;
			}
			/* sha_err.code == TOOL_UNAVAILABLE: checksum_verified stays
			 * false, honestly -- same disclosed degradation as the
			 * real-download path. */
		}
	}
	else
	{
		if (!want_json)
			printf("Installing %s (%s, %.1f MiB) from %s ...\n",
				f->display_name.c_str(), variant->quant.c_str(),
				(double)variant->size_bytes / (1024.0 * 1024.0),
				f->repo_url.c_str());
		membrane_download_error_t	dl_err;
		s_install_progress_ctx		pctx = {want_json, {0, 0}};
		bool	ok = membrane_download_file(variant->download_url, dest_path,
				variant->size_bytes, variant->sha256, install_progress_cb,
				&pctx, &checksum_verified, &dl_err);

		if (!want_json)
			printf("\n");
		if (!ok)
		{
			print_err(want_json, dl_err.code, dl_err.message);
			return (MEMBRANE_EXIT_MODEL_ERROR);
		}
	}
	/* Real GGUF validation on the just-downloaded file, the SAME check
	 * `membrane model add` already requires -- a download that
	 * completed with the right size/checksum but is somehow still not
	 * a readable GGUF (a real, if unlikely, possibility) is refused the
	 * same way a bad `add` path is, never registered blind. */
	membrane_gpu_model_estimate_t	est;

	if (!membrane_gpu_estimate_model(dest_path.c_str(), &est))
	{
		print_err(want_json, "INVALID_GGUF", std::string("downloaded '")
			+ dest_path + "' is not a readable GGUF file");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	e_membrane_registry_stat_status	stat_status;
	uint64_t	size_bytes;
	int64_t		mtime_ns;

	stat_file(dest_path, &stat_status, &size_bytes, &mtime_ns);

	membrane_registry_entry_t	entry;

	entry.name = f->name;
	entry.path = dest_path;
	const char	*base = membrane_runtime_safe_basename(dest_path.c_str());

	entry.basename = base != NULL ? base : "";
	entry.arch_name = est.hparams_available ? est.arch_name : "";
	entry.model_max_context = est.model_max_context_available
		? est.model_max_context : 0;
	entry.file_size_bytes = size_bytes;
	entry.file_mtime_ns = mtime_ns;
	entry.added_at_unix = (int64_t)time(NULL);

	std::string				registry_path = membrane_registry_resolve_path();
	membrane_registry_t		reg;
	membrane_registry_error_t	reg_err;

	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		print_err(want_json, reg_err.code, reg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	/* Idempotent: re-installing the same family+variant that is already
	 * correctly registered is a clean no-op success, matching `membrane
	 * setup`'s own established idempotence contract -- never a
	 * DUPLICATE_NAME error for re-running the same install. A
	 * DIFFERENT existing entry under this name is refused, same as
	 * setup_cmd.cpp's own conflict-detection. */
	const membrane_registry_entry_t	*existing
			= membrane_registry_find(reg, entry.name);

	if (existing != NULL && existing->path != entry.path)
	{
		print_err(want_json, "CLI_ERROR", std::string("'") + entry.name
			+ "' is already registered pointing at a different file ('"
			+ existing->path + "') -- remove it first with `membrane "
			"model remove " + entry.name + "`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (existing == NULL)
	{
		if (!membrane_registry_add(&reg, entry, &reg_err))
		{
			print_err(want_json, reg_err.code, reg_err.message);
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		if (!membrane_registry_save(registry_path, reg, &reg_err))
		{
			print_err(want_json, reg_err.code, reg_err.message);
			return (MEMBRANE_EXIT_MODEL_ERROR);
		}
	}
	if (want_json)
	{
		json	j;

		j["ok"] = true;
		j["name"] = entry.name;
		j["quant"] = variant->quant;
		j["path"] = entry.path;
		j["checksum_verified"] = checksum_verified;
		printf("%s\n", j.dump().c_str());
	}
	else
	{
		printf("Installed '%s' (%s) -> %s\n", entry.name.c_str(),
			variant->quant.c_str(), entry.path.c_str());
		if (!checksum_verified)
			printf("Note: checksum could not be verified in this "
				"environment (sha256sum unavailable) -- file size was "
				"still confirmed correct.\n");
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_uninstall(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model "
			"uninstall NAME");
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
	std::string	install_dir = membrane_registry_models_install_dir();
	bool		under_install_dir = !install_dir.empty()
			&& entry->path.rfind(install_dir + "/", 0) == 0;

	if (!under_install_dir)
	{
		print_err(want_json, "CLI_ERROR", std::string("'") + args[0]
			+ "' was not installed via `membrane model install` (its path "
			"is outside the managed models directory) -- use `membrane "
			"model remove` instead, which never deletes a file it did not "
			"itself download");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::string	path_to_delete = entry->path;

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
	/* Registry is already updated at this point -- a real failure to
	 * delete the on-disk file below is reported as a WARN-ish partial
	 * success, never re-adding the registry entry (Section 5: the
	 * registry state, the thing other commands actually trust, is the
	 * one source of truth; a leftover file on disk is a real but far
	 * less harmful outcome than the reverse). */
	bool	file_deleted = (unlink(path_to_delete.c_str()) == 0);

	if (want_json)
	{
		json	j;

		j["ok"] = true;
		j["uninstalled"] = args[0];
		j["file_deleted"] = file_deleted;
		printf("%s\n", j.dump().c_str());
	}
	else
	{
		printf("Uninstalled '%s'\n", args[0].c_str());
		if (!file_deleted)
			printf("Note: the registry entry was removed, but the file "
				"at %s could not be deleted (%s) -- you may want to "
				"remove it manually.\n", path_to_delete.c_str(),
				strerror(errno));
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_model_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	if (args.empty())
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane model "
			"add|remove|list|inspect|use|search|info|install|uninstall ...");
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
	if (args[0] == "search")
		return (cmd_search(rest, want_json));
	if (args[0] == "info")
		return (cmd_info(rest, want_json));
	if (args[0] == "install")
		return (cmd_install(rest, want_json));
	if (args[0] == "uninstall")
		return (cmd_uninstall(rest, want_json));
	print_err(want_json, "CLI_ERROR", std::string("unknown subcommand '")
		+ args[0] + "' -- expected add|remove|list|inspect|use|search|"
		"info|install|uninstall");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
