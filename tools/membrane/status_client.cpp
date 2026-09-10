#include "status_client.h"

#include <cstdio>

#include <httplib.h>

using json = nlohmann::json;

bool	membrane_fetch_server_status(const std::string &bind, int port,
			json *out)
{
	httplib::Client	cli(bind, port);

	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(2, 0);
	auto	res = cli.Get("/v1/status");

	if (!res || res->status != 200)
		return (false);
	try
	{
		*out = json::parse(res->body);
	}
	catch (const json::parse_error &)
	{
		return (false);
	}
	return (true);
}

void	membrane_print_server_status_human(const json &j)
{
	printf("  running: yes\n");
	printf("  endpoint: %s\n", j.value("endpoint", std::string("?")).c_str());
	/* Mega Phase D, PR D6, Section 20: default vs. active is a real,
	 * deliberate distinction (Section 11 of the task) -- printed as two
	 * separate lines rather than folded into one, so a default that is
	 * NOT currently active (service just started, or a live switch
	 * failed and recovered a different model) is never misread as a
	 * contradiction. */
	if (j.contains("default_model") && !j["default_model"].is_null())
		printf("  default model: %s\n",
			j["default_model"].get<std::string>().c_str());
	else
		printf("  default model: (none configured)\n");
	/* Mega Phase E, PR E2, Section 15 of the task: "resident models,
	 * pinned, evictable, backend, memory estimate" -- replaces PR E1's
	 * own single "active model" line, since more than one model can now
	 * be resident simultaneously (see server.cpp's own handle_status()
	 * comment for the /v1/status field this reads). */
	bool	printed_any = false;

	if (j.contains("resident_models") && j["resident_models"].is_array())
	{
		for (const auto &m : j["resident_models"])
		{
			printed_any = true;
			printf("  resident model: %s (%s%s)\n",
				m.value("model", std::string("?")).c_str(),
				m.value("state", std::string("?")).c_str(),
				m.value("pinned", false) ? ", pinned" : "");
			printf("    backend: %s\n",
				m.value("backend", std::string("?")).c_str());
			printf("    kv precision: %s\n",
				m.value("kv_precision", std::string("?")).c_str());
			printf("    evictable: %s\n",
				m.value("evictable", false) ? "yes" : "no");
		}
	}
	if (!printed_any)
		printf("  resident models: (none loaded yet)\n");
	printf("  resident model limit: %s\n",
		j.contains("resident_model_limit")
			? std::to_string(j.value("resident_model_limit", 0)).c_str()
			: "?");
	printf("  context policy: %s\n",
		j.value("context_policy", std::string("?")).c_str());
}

bool	membrane_activate_model(const std::string &bind, int port,
			const std::string &model_name,
			membrane_activate_result_t *out)
{
	httplib::Client	cli(bind, port);

	/* A model load is real, disk-bound work (Section 21/22 of the Mega
	 * Phase A task's own context/host-memory pipeline) -- a much longer
	 * read timeout than membrane_fetch_server_status()'s own 2s plain
	 * status read, but still bounded (never an indefinite hang if the
	 * server itself is wedged). 60s comfortably covers every model this
	 * project's own built-in catalog currently offers on ordinary
	 * consumer disks; a genuinely slower load is reported as a real
	 * transport failure here rather than blocking `membrane use` forever. */
	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(60, 0);

	nlohmann::json	body;

	body["model"] = model_name;
	auto	res = cli.Post("/membrane/v1/models/activate", body.dump(),
			"application/json");

	if (!res)
		return (false);
	nlohmann::json	parsed;

	try
	{
		parsed = nlohmann::json::parse(res->body);
	}
	catch (const nlohmann::json::parse_error &)
	{
		return (false);
	}
	out->ok = parsed.value("ok", false);
	out->already_active = parsed.value("already_active", false);
	out->active_model = (parsed.contains("active_model")
			&& !parsed["active_model"].is_null())
		? parsed["active_model"].get<std::string>() : std::string();
	out->backend = parsed.value("backend", std::string());
	if (parsed.contains("error") && parsed["error"].is_object())
	{
		out->error_code = parsed["error"].value("code", std::string());
		out->error_message = parsed["error"].value("message", std::string());
	}
	else
	{
		out->error_code.clear();
		out->error_message.clear();
	}
	return (true);
}

bool	membrane_set_model_pin(const std::string &bind, int port,
			const std::string &model_name, bool pin,
			membrane_pin_result_t *out)
{
	httplib::Client	cli(bind, port);

	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(5, 0);

	json	body;

	body["model"] = model_name;
	auto	res = cli.Post(pin ? "/membrane/v1/models/pin"
			: "/membrane/v1/models/unpin", body.dump(), "application/json");

	if (!res)
		return (false);
	json	parsed;

	try
	{
		parsed = json::parse(res->body);
	}
	catch (const json::parse_error &)
	{
		return (false);
	}
	out->ok = parsed.value("ok", false);
	out->pinned = parsed.value("pinned", false);
	if (parsed.contains("error") && parsed["error"].is_object())
	{
		out->error_code = parsed["error"].value("code", std::string());
		out->error_message = parsed["error"].value("message", std::string());
	}
	else
	{
		out->error_code.clear();
		out->error_message.clear();
	}
	return (true);
}
