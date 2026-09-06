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
	if (j.contains("loaded_model") && !j["loaded_model"].is_null())
	{
		printf("  active model: %s\n",
			j["loaded_model"].get<std::string>().c_str());
		printf("  backend: %s\n", j.value("backend", std::string("?")).c_str());
		printf("  kv precision: %s\n",
			j.value("kv_precision", std::string("?")).c_str());
	}
	else
		printf("  active model: (none loaded yet)\n");
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
