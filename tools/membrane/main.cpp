#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_cmd.h"
#include "server.h"
#include "server_config.h"
#include "service_cmd.h"
#include "status_client.h"
#include "doctor_cmd.h"
#include "setup_cmd.h"
#include "product_cli.h"

using json = nlohmann::json;

/*
 * `membrane`, the product CONTROL CLI -- distinct from `membrane-run`,
 * which stays the inference entry point (backwards compatible, unchanged
 * since Mega Phase A, PR A2). `membrane model ...` (A2), `membrane serve`
 * (A3, the local OpenAI-compatible HTTP server -- see server.h),
 * `membrane status` (A4 -- a thin HTTP client against a running
 * instance), and `membrane service ...` (Mega Phase B, PR B1 -- manages
 * membrane.service as a systemd --user unit, see service_cmd.h) are all
 * here. `membrane serve` itself is completely unchanged by PR B1 --
 * still the direct foreground/debug entry point service_cmd.cpp's
 * generated unit's own ExecStart calls.
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
	fprintf(out, "  membrane model use NAME           set NAME as the "
		"default model for `membrane serve`\n");
	fprintf(out, "  membrane serve                    start a local "
		"OpenAI-compatible HTTP server (foreground)\n");
	fprintf(out, "  membrane status                    check a running "
		"`membrane serve`/service instance\n");
	fprintf(out, "  membrane doctor                    unified diagnostic: "
		"installation, hardware, registry, config, service, HTTP\n");
	fprintf(out, "  membrane setup                     guided first-run: "
		"register a model, install/start the service, verify it\n");
	fprintf(out, "  membrane service install           install membrane."
		"service as a systemd --user unit\n");
	fprintf(out, "  membrane service uninstall         remove it\n");
	fprintf(out, "  membrane service start|stop|restart\n");
	fprintf(out, "                                      control the "
		"installed service\n");
	fprintf(out, "  membrane service status             systemd state + "
		"live HTTP status combined\n");
	fprintf(out, "  membrane service logs [-n N]       recent journalctl "
		"output (default 50 lines)\n");
	fprintf(out, "\n");
	fprintf(out, "service install options:\n");
	fprintf(out, "  --exec-path PATH                    use PATH instead "
		"of this binary's own real path\n");
	fprintf(out, "  --force                              overwrite a "
		"same-named unit MEMBRANE did not create\n");
	fprintf(out, "\n");
	fprintf(out, "setup options:\n");
	fprintf(out, "  --model PATH                        GGUF to register "
		"(prompted for interactively if omitted)\n");
	fprintf(out, "  --model-name NAME                   name to register "
		"it under (derived from the filename if omitted)\n");
	fprintf(out, "  --yes                                assume safe "
		"defaults, never prompt (for automation)\n");
	fprintf(out, "  --no-service                         skip service "
		"install/start (e.g. no systemd in this environment)\n");
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
		/* Section 8 of the Mega Phase B task: `membrane serve` (no
		 * flags) reads listen_address/port from the persistent config
		 * so a `membrane service install`-generated unit's own
		 * ExecStart can stay a plain "<path> serve" -- changing the
		 * port/bind later never requires regenerating/reinstalling the
		 * unit. An explicit --port/--bind on the command line always
		 * wins (checked via want_port/want_bind, same "explicit beats
		 * implicit" convention every other flag in this project uses).
		 * A missing/malformed config is never fatal here -- membrane_
		 * server_config_load() itself already falls back to defaults
		 * for "file does not exist"; a genuinely malformed config
		 * (parse error/bad schema) is reported but does not block
		 * startup, since sensible defaults are always available. */
		membrane_server_config_t		cfg
				= membrane_server_config_defaults();
		membrane_server_config_error_t	cfg_err;
		std::string						config_path
				= membrane_server_config_resolve_path();

		if (!config_path.empty()
			&& !membrane_server_config_load(config_path, &cfg, &cfg_err))
			fprintf(stderr, "membrane serve: WARNING -- could not read "
				"server config, using defaults: %s\n",
				cfg_err.message.c_str());
		membrane_server_options_t	opts;

		opts.bind_address = cfg.listen_address;
		opts.port = cfg.port;
		opts.default_model = cfg.default_model;
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
	if (args[0] == "service")
	{
		std::vector<std::string>	service_args(args.begin() + 1,
				args.end());

		return (membrane_service_cmd_dispatch(service_args, want_json));
	}
	if (args[0] == "doctor")
	{
		std::vector<std::string>	doctor_args(args.begin() + 1,
				args.end());

		return (membrane_doctor_cmd_dispatch(doctor_args, want_json));
	}
	if (args[0] == "setup")
	{
		std::vector<std::string>	setup_args(args.begin() + 1,
				args.end());

		return (membrane_setup_cmd_dispatch(setup_args, want_json));
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
		json	j;

		if (!membrane_fetch_server_status(bind, port, &j))
		{
			if (want_json)
				printf("{\"running\":false}\n");
			else
				printf("MEMBRANE server\n  running: no (no response from "
					"http://%s:%d)\n", bind.c_str(), port);
			return (MEMBRANE_EXIT_SUCCESS);
		}
		if (want_json)
			printf("%s\n", j.dump().c_str());
		else
		{
			printf("MEMBRANE server\n");
			membrane_print_server_status_human(j);
		}
		return (MEMBRANE_EXIT_SUCCESS);
	}
	fprintf(stderr, "membrane: unknown command '%s'\n", args[0].c_str());
	print_usage(stderr);
	return (MEMBRANE_EXIT_CLI_ERROR);
}
