#include "systemd_unit.h"

#include <cstdlib>
#include <sstream>

std::string	membrane_generate_unit_file(const membrane_unit_options_t &opts)
{
	std::string	description = opts.description.empty()
			? "MEMBRANE local inference server" : opts.description;
	std::ostringstream	oss;

	oss << MEMBRANE_UNIT_MARKER << "\n";
	oss << "[Unit]\n";
	oss << "Description=" << description << "\n";
	oss << "\n";
	oss << "[Service]\n";
	oss << "Type=simple\n";
	/* Quoted (systemd's own ExecStart= quoting, not a shell) so a path
	 * containing spaces still works; `serve` takes no arguments here on
	 * purpose -- bind address/port/default model live in server_config.h's
	 * own config file, read at startup, so changing them never requires
	 * regenerating this unit. */
	oss << "ExecStart=\"" << opts.exec_path << "\" serve\n";
	/* Section 7: bounded restart policy, never an aggressive loop. */
	oss << "Restart=on-failure\n";
	oss << "RestartSec=2\n";
	oss << "\n";
	oss << "[Install]\n";
	oss << "WantedBy=default.target\n";
	return (oss.str());
}

std::string	membrane_unit_file_path(void)
{
	const char	*dir_override = getenv("MEMBRANE_SYSTEMD_USER_DIR");

	if (dir_override != NULL && dir_override[0] != '\0')
		return (std::string(dir_override) + "/" + MEMBRANE_UNIT_NAME);
	const char	*xdg_config_home = getenv("XDG_CONFIG_HOME");

	if (xdg_config_home != NULL && xdg_config_home[0] != '\0')
		return (std::string(xdg_config_home) + "/systemd/user/"
			+ MEMBRANE_UNIT_NAME);
	const char	*home = getenv("HOME");

	if (home != NULL && home[0] != '\0')
		return (std::string(home) + "/.config/systemd/user/"
			+ MEMBRANE_UNIT_NAME);
	return ("");
}

bool	membrane_unit_is_membrane_managed(const std::string &content)
{
	return (content.find(MEMBRANE_UNIT_MARKER) != std::string::npos);
}
