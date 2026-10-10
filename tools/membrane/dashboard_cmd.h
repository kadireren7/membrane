#ifndef MEMBRANE_DASHBOARD_CMD_H
# define MEMBRANE_DASHBOARD_CMD_H

/*
 * Final productization (P1): bare `membrane` is a concise status screen, not
 * a wall of help text.
 *
 *   MEMBRANE
 *
 *   Model      Qwen2.5-1.5B-Instruct
 *   Runtime    Native
 *   GPU        NVIDIA GeForce GTX 1650 (Vulkan)
 *   Memory     Healthy
 *   Server     Running
 *   Adaptive   Off
 *
 *   Next:
 *     membrane chat
 *     membrane models
 *
 * When setup is incomplete it shows the one or two best next steps (and only
 * commands that would work). `membrane --json` gives the same facts for a
 * machine. `membrane --help` / `membrane help` is the full command list.
 */

typedef struct s_membrane_dashboard
{
	bool			model_known;
	char			model[160];
	int				installed_count;
	bool			server_running;
	char			endpoint[96];
	bool			service_installed;
	bool			service_active;
	char			gpu[160];			/* "" = none */
	bool			gpu_backend_compiled;
	char			memory_status[16];	/* healthy|notice|warning|critical|unknown */
	char			adaptive_mode[16];
	char			next[2][96];
	int				next_count;
}	membrane_dashboard_t;

/* Pure: the best one or two next steps for the given facts. Only commands
 * that would succeed in that state are ever suggested. */
void	membrane_dashboard_next_steps(membrane_dashboard_t *d);

int		membrane_dashboard_cmd_dispatch(bool want_json);

#endif
