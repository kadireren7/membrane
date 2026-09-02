#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "model_cmd.h"
#include "server.h"
#include "product_cli.h"

using json = nlohmann::json;

/*
 * Mega Phase A: `membrane`, the new product CONTROL CLI (Section 10) --
 * distinct from `membrane-run`, which stays the inference entry point
 * (backwards compatible, unchanged since PR A2). `membrane model ...`
 * (PR A2), `membrane serve` (PR A3, the local OpenAI-compatible HTTP
 * server -- see server.h), and `membrane status` (PR A4, Section 37 --
 * a thin HTTP CLIENT against a `serve` instance already running in the
 * foreground elsewhere; this project has no daemon/process-management
 * capability and does not claim one) are all here.
 */

static void	print_usage(FILE *out)
{
	fprintf(out, "membrane %s -- MEMBRANE control CLI\n\n", MEMBRANE_VERSION);
	fprintf(out, "Usage:\n");
	fprintf(out, "  membrane model add NAME PATH      register a local "
		"GGUF model under NAME\n");
	fprintf(out, "  membrane model remove NAME        unregister NAME\n");
	fprintf(out, "  membrane model list                list every "
		"registered model\n");
	fprintf(out, "  membrane model inspect NAME       show one "
		"registered model's details\n");
	fprintf(out, "  membrane serve                    start a local "
		"OpenAI-compatible HTTP server\n");
	fprintf(out, "  membrane status                    check a running "
		"`membrane serve` instance\n");
	fprintf(out, "\n");
	fprintf(out, "serve options:\n");
	fprintf(out, "  --port N                           listen port "
		"(default 8642)\n");
	fprintf(out, "  --bind ADDRESS                     bind address "
		"(default 127.0.0.1 -- loopback only)\n");
	fprintf(out, "  --allow-non-loopback                required to bind "
		"any address other than 127.0.0.1/localhost;\n");
	fprintf(out, "                                      the server has "
		"NO authentication, see --help output above\n");
	fprintf(out, "\n");
	fprintf(out, "status options:\n");
	fprintf(out, "  --port N                           port to check "
		"(default 8642)\n");
	fprintf(out, "  --bind ADDRESS                     address to check "
		"(default 127.0.0.1)\n");
	fprintf(out, "\n");
	fprintf(out, "Options:\n");
	fprintf(out, "  --json                             machine-readable "
		"JSON output\n");
	fprintf(out, "  --help                             show this help\n");
	fprintf(out, "  --version                          show version\n");
	fprintf(out, "\n");
	fprintf(out, "The registry is stored at $XDG_DATA_HOME/membrane/"
		"models.json\n");
	fprintf(out, "(or $HOME/.local/share/membrane/models.json if "
		"XDG_DATA_HOME is unset).\n");
	fprintf(out, "\n");
	fprintf(out, "membrane-run remains the inference entry point -- see "
		"membrane-run --help.\n");
}

int	main(int argc, char **argv)
{
	std::vector<std::string>	args;
	bool						want_json = false;
	int							i;

	i = 1;
	while (i < argc)
	{
		std::string	a = argv[i];

		if (a == "--json")
			want_json = true;
		else if (a == "--help" || a == "-h")
			return (print_usage(stdout), MEMBRANE_EXIT_SUCCESS);
		else if (a == "--version")
			return (printf("membrane %s\n", MEMBRANE_VERSION),
				MEMBRANE_EXIT_SUCCESS);
		else
			args.push_back(a);
		i++;
	}
	if (args.empty())
	{
		print_usage(stderr);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (args[0] == "model")
	{
		std::vector<std::string>	model_args(args.begin() + 1,
				args.end());

		return (membrane_model_cmd_dispatch(model_args, want_json));
	}
	if (args[0] == "serve")
	{
		membrane_server_options_t	opts;

		opts.bind_address = "127.0.0.1";
		opts.port = 8642;
		opts.allow_non_loopback = false;
		for (i = 1; i < (int)args.size(); ++i)
		{
			if (args[i] == "--port" && i + 1 < (int)args.size())
				opts.port = atoi(args[++i].c_str());
			else if (args[i] == "--bind" && i + 1 < (int)args.size())
				opts.bind_address = args[++i];
			else if (args[i] == "--allow-non-loopback")
				opts.allow_non_loopback = true;
			else
			{
				fprintf(stderr, "membrane serve: unknown option '%s'\n",
					args[i].c_str());
				return (MEMBRANE_EXIT_CLI_ERROR);
			}
		}
		return (membrane_server_run(opts));
	}
	if (args[0] == "status")
	{
		std::string	bind = "127.0.0.1";
		int			port = 8642;

		for (i = 1; i < (int)args.size(); ++i)
		{
			if (args[i] == "--port" && i + 1 < (int)args.size())
				port = atoi(args[++i].c_str());
			else if (args[i] == "--bind" && i + 1 < (int)args.size())
				bind = args[++i];
			else
			{
				fprintf(stderr, "membrane status: unknown option '%s'\n",
					args[i].c_str());
				return (MEMBRANE_EXIT_CLI_ERROR);
			}
		}
		httplib::Client	cli(bind, port);

		cli.set_connection_timeout(0, 500000);
		cli.set_read_timeout(2, 0);
		auto	res = cli.Get("/v1/status");

		if (!res || res->status != 200)
		{
			if (want_json)
				printf("{\"running\":false}\n");
			else
				printf("MEMBRANE server\n  running: no (no response from "
					"http://%s:%d)\n", bind.c_str(), port);
			return (MEMBRANE_EXIT_SUCCESS);
		}
		json	j;

		try
		{
			j = json::parse(res->body);
		}
		catch (const json::parse_error &)
		{
			fprintf(stderr, "membrane status: server returned an "
				"unparseable response\n");
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
		if (want_json)
			printf("%s\n", j.dump().c_str());
		else
		{
			printf("MEMBRANE server\n");
			printf("  running: yes\n");
			printf("  endpoint: %s\n",
				j.value("endpoint", std::string("?")).c_str());
			if (j.contains("loaded_model") && !j["loaded_model"].is_null())
			{
				printf("  loaded model: %s\n",
					j["loaded_model"].get<std::string>().c_str());
				printf("  backend: %s\n",
					j.value("backend", std::string("?")).c_str());
				printf("  kv precision: %s\n",
					j.value("kv_precision", std::string("?")).c_str());
			}
			else
				printf("  loaded model: (none yet)\n");
			printf("  context policy: %s\n",
				j.value("context_policy", std::string("?")).c_str());
		}
		return (MEMBRANE_EXIT_SUCCESS);
	}
	fprintf(stderr, "membrane: unknown command '%s'\n", args[0].c_str());
	print_usage(stderr);
	return (MEMBRANE_EXIT_CLI_ERROR);
}
