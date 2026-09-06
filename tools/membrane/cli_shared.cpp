#include "cli_shared.h"

#include <cstdarg>
#include <cstdio>
#include <iostream>

#include "membrane/posix_compat.h"

bool	membrane_cli_is_interactive(void)
{
	return (isatty(fileno(stdin)) != 0);
}

bool	membrane_cli_ask_yes_no(const std::string &prompt, bool default_yes,
				bool assume_yes)
{
	if (assume_yes || !membrane_cli_is_interactive())
		return (default_yes);
	printf("%s [%s]: ", prompt.c_str(), default_yes ? "Y/n" : "y/N");
	fflush(stdout);
	std::string	line;

	if (!std::getline(std::cin, line) || line.empty())
		return (default_yes);
	return (line[0] == 'y' || line[0] == 'Y');
}

std::string	membrane_cli_ask_line(const std::string &prompt)
{
	if (!membrane_cli_is_interactive())
		return ("");
	printf("%s", prompt.c_str());
	fflush(stdout);
	std::string	line;

	if (!std::getline(std::cin, line))
		return ("");
	return (line);
}

void	membrane_cli_narrate(bool want_json, const char *fmt, ...)
{
	if (want_json)
		return ;

	va_list	ap;

	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}

stdout_silencer_t::stdout_silencer_t(bool active)
	: active_(active), saved_fd_(-1)
{
	if (!active_)
		return ;
	fflush(stdout);
	saved_fd_ = dup(STDOUT_FILENO);
	int	devnull = open(MEMBRANE_NULL_DEVICE, O_WRONLY);

	if (devnull >= 0)
	{
		dup2(devnull, STDOUT_FILENO);
		close(devnull);
	}
}

stdout_silencer_t::~stdout_silencer_t(void)
{
	if (!active_ || saved_fd_ < 0)
		return ;
	fflush(stdout);
	dup2(saved_fd_, STDOUT_FILENO);
	close(saved_fd_);
}

int	membrane_cli_dispatch_silently_if_json(bool want_json,
				const std::vector<std::string> &args,
				int (*fn)(const std::vector<std::string> &, bool))
{
	stdout_silencer_t	silencer(want_json);

	return (fn(args, false));
}
