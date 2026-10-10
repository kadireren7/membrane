#include "dashboard_cmd.h"

#include <cstdio>
#include <cstring>
#include <memory>

#include <nlohmann/json.hpp>

#include "adaptive.h"
#include "adaptive_controller.h"
#include "advise_cmd.h"
#include "gpu_device.h"
#include "model_views.h"
#include "product_cli.h"
#include "server_config.h"
#include "service_state.h"
#include "status_client.h"
#include "ui.h"

using json = nlohmann::json;

/* See dashboard_cmd.h's top comment. */

static void	set(char *dst, size_t n, const std::string &s)
{
	snprintf(dst, n, "%s", s.c_str());
}

void	membrane_dashboard_next_steps(membrane_dashboard_t *d)
{
	d->next_count = 0;
	if (d->installed_count == 0)
	{
		set(d->next[d->next_count++], sizeof(d->next[0]),
			"membrane models recommend");
		set(d->next[d->next_count++], sizeof(d->next[0]), "membrane use");
		return ;
	}
	if (!d->model_known)
	{
		set(d->next[d->next_count++], sizeof(d->next[0]), "membrane use");
		set(d->next[d->next_count++], sizeof(d->next[0]), "membrane models");
		return ;
	}
	if (!d->server_running)
	{
		set(d->next[d->next_count++], sizeof(d->next[0]),
			d->service_installed ? "membrane service start"
				: "membrane serve");
		set(d->next[d->next_count++], sizeof(d->next[0]), "membrane models");
		return ;
	}
	set(d->next[d->next_count++], sizeof(d->next[0]), "membrane chat");
	set(d->next[d->next_count++], sizeof(d->next[0]), "membrane models");
}

static std::string	status_word(const char *s)
{
	std::string	w = s;

	if (w == "healthy")
		return (membrane_ui_paint(MEMBRANE_UI_OK, "Healthy"));
	if (w == "notice")
		return (membrane_ui_paint(MEMBRANE_UI_OK, "Healthy")
			+ membrane_ui_paint(MEMBRANE_UI_DIM, " (a notice: see `membrane "
				"advise`)"));
	if (w == "warning")
		return (membrane_ui_paint(MEMBRANE_UI_WARN, "Warning"));
	if (w == "critical")
		return (membrane_ui_paint(MEMBRANE_UI_ERR, "Critical"));
	return (membrane_ui_paint(MEMBRANE_UI_DIM, "Unknown"));
}

int	membrane_dashboard_cmd_dispatch(bool want_json)
{
	std::unique_ptr<membrane_dashboard_t>	d(new membrane_dashboard_t());
	std::vector<membrane_model_row_t>		rows;
	membrane_server_config_t				cfg = membrane_server_config_defaults();
	membrane_server_config_error_t			cerr;
	std::string								cpath
		= membrane_server_config_resolve_path();
	json									status;

	memset(d.get(), 0, sizeof(*d));
	set(d->memory_status, sizeof(d->memory_status), "unknown");
	if (!cpath.empty())
		membrane_server_config_load(cpath, &cfg, &cerr);
	membrane_model_rows_installed(&rows, false);
	d->installed_count = (int)rows.size();
	for (const auto &r : rows)
		if (r.is_default)
		{
			d->model_known = true;
			set(d->model, sizeof(d->model), r.display);
		}
	snprintf(d->endpoint, sizeof(d->endpoint), "http://%s:%d",
		cfg.listen_address.c_str(), cfg.port);
	d->server_running = membrane_fetch_server_status(cfg.listen_address,
			cfg.port, &status);
	{
		membrane_service_probe_t	sp = membrane_probe_service();

		d->service_installed = sp.installed;
		d->service_active = sp.active;
	}
	d->gpu_backend_compiled = membrane_gpu_backend_available() != 0;
	{
		membrane_gpu_device_info_t	devs[MEMBRANE_GPU_MAX_DEVICES];
		size_t						n = membrane_gpu_list_devices(devs,
				MEMBRANE_GPU_MAX_DEVICES);

		for (size_t i = 0; i < n && d->gpu[0] == '\0'; ++i)
			if (devs[i].type == MEMBRANE_DEV_TYPE_GPU
				|| devs[i].type == MEMBRANE_DEV_TYPE_IGPU)
				snprintf(d->gpu, sizeof(d->gpu), "%s (%s)",
					devs[i].description[0] ? devs[i].description
						: devs[i].name, devs[i].backend);
	}
	{
		membrane_adaptive_policy_t	pol;

		membrane_adaptive_policy_defaults(&pol);
		if (!membrane_adaptive_policy_from_config(cfg.adaptive, &pol).empty())
			membrane_adaptive_policy_defaults(&pol);
		set(d->adaptive_mode, sizeof(d->adaptive_mode),
			membrane_adaptive_mode_name(
				membrane_adaptive_effective_mode(&pol)));
	}
	if (d->model_known && d->server_running)
	{
		std::unique_ptr<membrane_observation_snapshot_t>	snap(
				new membrane_observation_snapshot_t());
		std::unique_ptr<membrane_memory_intelligence_t>	intel(
				new membrane_memory_intelligence_t());
		membrane_runtime_capabilities_t					caps;
		std::string										err;

		if (membrane_advise_collect(MEMBRANE_RUNTIME_ID_NATIVE, snap.get(),
				&caps, intel.get(), &err)
			&& intel->status != MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA)
			set(d->memory_status, sizeof(d->memory_status),
				membrane_intel_status_name(intel->status) == std::string("ok")
				? "healthy" : membrane_intel_status_name(intel->status));
	}
	membrane_dashboard_next_steps(d.get());
	if (want_json)
	{
		json	next = json::array();

		for (int i = 0; i < d->next_count; ++i)
			next.push_back(d->next[i]);
		printf("%s\n", json{{"schema_version", 1}, {"mode", "dashboard"},
			{"membrane_version", MEMBRANE_VERSION},
			{"model", d->model_known ? json(d->model) : json(nullptr)},
			{"installed_models", d->installed_count},
			{"runtime", "membrane-native"},
			{"gpu", d->gpu[0] ? json(d->gpu) : json(nullptr)},
			{"gpu_backend_compiled", d->gpu_backend_compiled},
			{"memory", d->memory_status},
			{"server", {{"running", d->server_running},
				{"endpoint", d->endpoint}}},
			{"service", {{"installed", d->service_installed},
				{"active", d->service_active}}},
			{"adaptive", d->adaptive_mode}, {"next", next}}.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("%s\n\n", membrane_ui_paint(MEMBRANE_UI_BOLD, "MEMBRANE").c_str());
	printf("  %-9s %s\n", "Model", d->model_known ? d->model
		: membrane_ui_paint(MEMBRANE_UI_DIM, d->installed_count
			? "none selected" : "none installed").c_str());
	printf("  %-9s %s\n", "Runtime", "Native");
	printf("  %-9s %s\n", "GPU", d->gpu[0] ? d->gpu
		: membrane_ui_paint(MEMBRANE_UI_DIM, d->gpu_backend_compiled
			? "none detected" : "none (this build is CPU-only)").c_str());
	printf("  %-9s %s\n", "Memory", d->model_known && d->server_running
		? status_word(d->memory_status).c_str()
		: membrane_ui_paint(MEMBRANE_UI_DIM, "not measured (no running "
			"server)").c_str());
	printf("  %-9s %s\n", "Server", d->server_running
		? (membrane_ui_paint(MEMBRANE_UI_OK, "Running") + " "
			+ membrane_ui_paint(MEMBRANE_UI_DIM, d->endpoint)).c_str()
		: membrane_ui_paint(MEMBRANE_UI_DIM, "Stopped").c_str());
	{
		std::string	am = d->adaptive_mode;

		am[0] = (char)toupper((unsigned char)am[0]);
		printf("  %-9s %s\n", "Adaptive", am.c_str());
	}
	printf("\nNext:\n");
	for (int i = 0; i < d->next_count; ++i)
		printf("  %s\n", membrane_ui_paint(MEMBRANE_UI_INFO,
			d->next[i]).c_str());
	printf("\n%s\n", membrane_ui_paint(MEMBRANE_UI_DIM, "membrane --help lists "
		"every command.").c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}
