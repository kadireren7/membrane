#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model_cmd.h"
#include "product_cli.h"

/*
 * Mega Phase A, PR A2: `membrane`, the new product CONTROL CLI (Section
 * 10 of the task) -- distinct from `membrane-run`, which stays the
 * inference entry point (backwards compatible, unchanged by this PR).
 * `membrane model ...` is this phase's only subcommand; `membrane serve`/
 * `membrane status` are future Mega Phase A work (A3), not added here.
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
	fprintf(stderr, "membrane: unknown command '%s'\n", args[0].c_str());
	print_usage(stderr);
	return (MEMBRANE_EXIT_CLI_ERROR);
}
