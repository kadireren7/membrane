#include "ui.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
# include <io.h>
# define MEMBRANE_UI_ISATTY(fd)	_isatty(fd)
#else
# include <unistd.h>
# define MEMBRANE_UI_ISATTY(fd)	isatty(fd)
#endif

/* See ui.h's top comment. */

static bool	g_color = false;
static bool	g_unicode = false;
static bool	g_forced = false;

static bool	env_set(const char *name)
{
	const char	*v = getenv(name);

	return (v != NULL && v[0] != '\0');
}

static bool	contains_utf8(const char *v)
{
	std::string	s;

	if (v == NULL)
		return (false);
	for (; *v; ++v)
		s.push_back((char)std::tolower((unsigned char)*v));
	return (s.find("utf-8") != std::string::npos
		|| s.find("utf8") != std::string::npos);
}

void	membrane_ui_configure(bool no_color_flag)
{
	bool		tty;
	const char	*term = getenv("TERM");

	if (g_forced)
		return ;
	tty = MEMBRANE_UI_ISATTY(1) != 0;
	g_color = tty && !no_color_flag && !env_set("NO_COLOR")
		&& !(term != NULL && strcmp(term, "dumb") == 0);
	/* unicode symbols only on a terminal that is known to render UTF-8 */
	g_unicode = tty && !env_set("MEMBRANE_ASCII")
		&& !(term != NULL && (strcmp(term, "dumb") == 0
			|| strcmp(term, "linux") == 0))
		&& (contains_utf8(getenv("LC_ALL")) || contains_utf8(getenv("LC_CTYPE"))
			|| contains_utf8(getenv("LANG")));
}

void	membrane_ui_force(bool color, bool unicode)
{
	g_forced = true;
	g_color = color;
	g_unicode = unicode;
}

void	membrane_ui_unforce(void)
{
	g_forced = false;
	membrane_ui_configure(false);
}

bool	membrane_ui_is_tty(void) { return (MEMBRANE_UI_ISATTY(1) != 0); }
bool	membrane_ui_color(void) { return (g_color); }
bool	membrane_ui_unicode(void) { return (g_unicode); }

std::string	membrane_ui_paint(membrane_ui_style_t s, const std::string &text)
{
	const char	*code;

	if (!g_color || text.empty())
		return (text);
	switch (s)
	{
		case MEMBRANE_UI_OK: code = "32"; break ;
		case MEMBRANE_UI_WARN: code = "33"; break ;
		case MEMBRANE_UI_ERR: code = "31"; break ;
		case MEMBRANE_UI_INFO: code = "36"; break ;
		case MEMBRANE_UI_DIM: code = "90"; break ;
		default: code = "1"; break ;
	}
	return (std::string("\033[") + code + "m" + text + "\033[0m");
}

std::string	membrane_ui_sym_ok(void)
{
	return (membrane_ui_paint(MEMBRANE_UI_OK, g_unicode ? "\xE2\x9C\x93"
		: "[ok]"));
}

std::string	membrane_ui_sym_warn(void)
{
	return (membrane_ui_paint(MEMBRANE_UI_WARN, "!"));
}

std::string	membrane_ui_sym_err(void)
{
	return (membrane_ui_paint(MEMBRANE_UI_ERR, g_unicode ? "\xC3\x97" : "[x]"));
}

std::string	membrane_ui_sym_dot(void)
{
	return (membrane_ui_paint(MEMBRANE_UI_DIM, g_unicode ? "\xE2\x97\x8F"
		: "*"));
}
