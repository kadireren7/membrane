#ifndef MEMBRANE_UI_H
# define MEMBRANE_UI_H

# include <string>

/*
 * Final productization (P12/P13): the one place that decides how human
 * output is styled.
 *
 *   HUMAN OUTPUT FOR HUMANS, JSON FOR MACHINES.
 *
 * Color is semantic and tasteful: green = healthy/success, yellow =
 * warning, red = error/critical, cyan = informational/selected, dim =
 * unavailable/unknown. It is ON only when stdout is a terminal, NO_COLOR is
 * not set (https://no-color.org), TERM is not "dumb", and `--no-color` was
 * not given. Redirected output never contains an ANSI escape, and JSON is
 * never colored (JSON paths do not call into this module).
 *
 * Symbols are the simple portable set  check / ! / x / dot, with an ASCII
 * fallback ([ok] [!] [x] *) used when the terminal is not known to render
 * UTF-8, when output is redirected, or when MEMBRANE_ASCII=1.
 */

typedef enum e_membrane_ui_style
{
	MEMBRANE_UI_OK = 0,		/* green  */
	MEMBRANE_UI_WARN,		/* yellow */
	MEMBRANE_UI_ERR,		/* red    */
	MEMBRANE_UI_INFO,		/* cyan   */
	MEMBRANE_UI_DIM,		/* gray   */
	MEMBRANE_UI_BOLD
}	membrane_ui_style_t;

/* Re-reads the environment and the terminal state. `no_color_flag` is the
 * global --no-color. Called once from main(); safe to call again. */
void		membrane_ui_configure(bool no_color_flag);

/* Test hooks: force the decision regardless of tty/env. */
void		membrane_ui_force(bool color, bool unicode);
void		membrane_ui_unforce(void);

bool		membrane_ui_color(void);
bool		membrane_ui_unicode(void);

/* text wrapped in the style's escape codes iff color is on */
std::string	membrane_ui_paint(membrane_ui_style_t s, const std::string &text);

/* check / ! / x / dot, already painted */
std::string	membrane_ui_sym_ok(void);
std::string	membrane_ui_sym_warn(void);
std::string	membrane_ui_sym_err(void);
std::string	membrane_ui_sym_dot(void);

#endif
