#ifndef MEMBRANE_SERVICE_CMD_H
# define MEMBRANE_SERVICE_CMD_H

# include <string>
# include <vector>

/*
 * Mega Phase B, PR B1: `membrane service install|uninstall|start|stop|
 * restart|status|logs` -- manages membrane.service as a systemd --user
 * unit (server.h's foreground membrane_server_run() is completely
 * unchanged; `membrane serve` still works exactly as before, for direct/
 * debug use -- Section 4 of the task).
 */
int	membrane_service_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
