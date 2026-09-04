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
	if (j.contains("loaded_model") && !j["loaded_model"].is_null())
	{
		printf("  loaded model: %s\n",
			j["loaded_model"].get<std::string>().c_str());
		printf("  backend: %s\n", j.value("backend", std::string("?")).c_str());
		printf("  kv precision: %s\n",
			j.value("kv_precision", std::string("?")).c_str());
	}
	else
		printf("  loaded model: (none yet)\n");
	printf("  context policy: %s\n",
		j.value("context_policy", std::string("?")).c_str());
}
