#include "windows_task.h"

#include <cstdlib>
#include <sstream>

/*
 * XML-escapes the four characters that matter inside an element's text
 * content or a double-quoted attribute value (& < > "). Task Scheduler
 * XML is otherwise plain UTF-8 -- no CDATA sections, no other entities
 * needed for the values this file's own callers ever pass (a real
 * filesystem path).
 */
static std::string	xml_escape(const std::string &s)
{
	std::string	out;

	out.reserve(s.size());
	for (char c : s)
	{
		if (c == '&')
			out += "&amp;";
		else if (c == '<')
			out += "&lt;";
		else if (c == '>')
			out += "&gt;";
		else if (c == '"')
			out += "&quot;";
		else
			out += c;
	}
	return (out);
}

std::string	membrane_generate_task_xml(const membrane_task_options_t &opts)
{
	std::string	log_path = opts.log_path.empty()
			? "C:\\Windows\\Temp\\membrane-service.log" : opts.log_path;
	std::ostringstream	oss;

	oss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	oss << "<Task version=\"1.2\" "
		"xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n";
	oss << "  <RegistrationInfo>\n";
	oss << "    <Description>" << MEMBRANE_TASK_MARKER << "</Description>\n";
	oss << "  </RegistrationInfo>\n";
	/* No <Triggers> element's own child triggers -- nothing auto-starts
	 * this task; `membrane service start` (a real `schtasks /run`) is
	 * the one, explicit way it ever actually runs, matching
	 * systemd_unit.h/launchd_unit.h's own no-auto-start policy. */
	oss << "  <Triggers />\n";
	oss << "  <Principals>\n";
	oss << "    <Principal id=\"Author\">\n";
	oss << "      <LogonType>InteractiveToken</LogonType>\n";
	oss << "      <RunLevel>LeastPrivilege</RunLevel>\n";
	oss << "    </Principal>\n";
	oss << "  </Principals>\n";
	oss << "  <Settings>\n";
	oss << "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n";
	oss << "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n";
	oss << "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n";
	oss << "    <StartWhenAvailable>false</StartWhenAvailable>\n";
	oss << "    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>\n";
	oss << "    <Enabled>true</Enabled>\n";
	oss << "    <Hidden>false</Hidden>\n";
	oss << "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n";
	oss << "  </Settings>\n";
	oss << "  <Actions Context=\"Author\">\n";
	oss << "    <Exec>\n";
	oss << "      <Command>cmd.exe</Command>\n";
	/* Runs via cmd.exe /c so real stdout/stderr redirection to a real
	 * log file is possible at all -- Task Scheduler's own Exec action
	 * has no output-capture of its own (see windows_task.h's own top
	 * comment). `serve` takes no other arguments here on purpose, same
	 * reasoning as systemd_unit.h's own ExecStart=/launchd_unit.h's own
	 * ProgramArguments: bind address/port/default model live in
	 * server_config.h's own config file, read at startup. */
	oss << "      <Arguments>/c \"\"" << xml_escape(opts.exec_path)
		<< "\" serve &gt;&gt; \"" << xml_escape(log_path)
		<< "\" 2&gt;&amp;1\"</Arguments>\n";
	oss << "    </Exec>\n";
	oss << "  </Actions>\n";
	oss << "</Task>\n";
	return (oss.str());
}

std::string	membrane_task_name(void)
{
	const char	*name_override = getenv("MEMBRANE_TASK_NAME_OVERRIDE");

	if (name_override != NULL && name_override[0] != '\0')
		return (name_override);
	return (MEMBRANE_TASK_NAME);
}

bool	membrane_task_xml_is_membrane_managed(const std::string &xml)
{
	return (xml.find(MEMBRANE_TASK_MARKER) != std::string::npos);
}
