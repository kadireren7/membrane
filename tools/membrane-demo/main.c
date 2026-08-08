#include <stdio.h>

#include "demo_core.h"

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-demo [options]\n"
		"\n"
		"Runs a deterministic, offline demonstration of the maintained\n"
		"MEMBRANE Q4_0/Q8_0 KV-block quantization path: no model\n"
		"download, no network access.\n"
		"\n"
		"  --blocks N   number of synthetic blocks to process "
		"(default %u)\n"
		"  --seed N     PRNG seed for the synthetic workload "
		"(default %u)\n"
		"  --json       print machine-readable JSON instead of the\n"
		"               human-readable report\n"
		"  --help       print this message and exit\n",
		MEMBRANE_DEMO_DEFAULT_BLOCKS, MEMBRANE_DEMO_DEFAULT_SEED);
}

int	main(int argc, char **argv)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	result;
	char					err_buf[160];
	int						want_json;
	int						want_help;
	int						rc;

	rc = membrane_demo_parse_args(argc, argv, &cfg, &want_json, &want_help,
			err_buf, sizeof(err_buf));
	if (want_help)
		return (usage(stdout), 0);
	if (rc != 0)
	{
		fprintf(stderr, "membrane-demo: %s\n", err_buf);
		usage(stderr);
		return (rc);
	}
	if (membrane_demo_run(&cfg, &result) != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-demo: workload failed to run "
			"(allocation failure?)\n");
		return (1);
	}
	if (want_json)
		membrane_demo_print_json(&result, stdout);
	else
		membrane_demo_print_human(&result, stdout);
	return (result.validation_pass ? 0 : 1);
}
