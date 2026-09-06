#include "launchd_unit.h"

#include <cstdlib>
#include <sstream>

std::string	membrane_generate_launchd_plist(
				const membrane_launchd_options_t &opts)
{
	std::string	log_path = opts.log_path.empty()
			? "/tmp/membrane-service.log" : opts.log_path;
	std::ostringstream	oss;

	oss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	oss << "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
		"\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n";
	/* XML comment: plist parsers ignore it exactly like systemd ignores
	 * a leading "#" line -- same marker mechanism, different comment
	 * syntax. */
	oss << "<!-- " << MEMBRANE_LAUNCHD_MARKER << " -->\n";
	oss << "<plist version=\"1.0\">\n";
	oss << "<dict>\n";
	oss << "\t<key>Label</key>\n";
	oss << "\t<string>" << MEMBRANE_LAUNCHD_LABEL << "</string>\n";
	oss << "\t<key>ProgramArguments</key>\n";
	oss << "\t<array>\n";
	/* Unlike systemd's ExecStart= (its own quoting, not a shell),
	 * ProgramArguments is a real XML array -- each argument is its own
	 * <string> element, so a path containing spaces needs no quoting at
	 * all (the array structure itself is the delimiter). `serve` takes
	 * no arguments here on purpose, same reasoning as systemd_unit.h's
	 * own ExecStart=: bind address/port/default model live in
	 * server_config.h's own config file, read at startup. */
	oss << "\t\t<string>" << opts.exec_path << "</string>\n";
	oss << "\t\t<string>serve</string>\n";
	oss << "\t</array>\n";
	oss << "\t<key>RunAtLoad</key>\n";
	oss << "\t<false/>\n";
	oss << "\t<key>KeepAlive</key>\n";
	oss << "\t<dict>\n";
	oss << "\t\t<key>SuccessfulExit</key>\n";
	oss << "\t\t<false/>\n";
	oss << "\t</dict>\n";
	oss << "\t<key>ThrottleInterval</key>\n";
	oss << "\t<integer>2</integer>\n";
	oss << "\t<key>StandardOutPath</key>\n";
	oss << "\t<string>" << log_path << "</string>\n";
	oss << "\t<key>StandardErrorPath</key>\n";
	oss << "\t<string>" << log_path << "</string>\n";
	oss << "</dict>\n";
	oss << "</plist>\n";
	return (oss.str());
}

std::string	membrane_launchd_plist_path(void)
{
	const char	*dir_override = getenv("MEMBRANE_LAUNCHD_USER_DIR");

	if (dir_override != NULL && dir_override[0] != '\0')
		return (std::string(dir_override) + "/"
			MEMBRANE_LAUNCHD_LABEL ".plist");
	const char	*home = getenv("HOME");

	if (home != NULL && home[0] != '\0')
		return (std::string(home) + "/Library/LaunchAgents/"
			MEMBRANE_LAUNCHD_LABEL ".plist");
	return ("");
}

bool	membrane_launchd_plist_is_membrane_managed(const std::string &content)
{
	return (content.find(MEMBRANE_LAUNCHD_MARKER) != std::string::npos);
}
