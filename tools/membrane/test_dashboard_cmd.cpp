#include <cstdio>
#include <cstdlib>
#include <string>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "dashboard_cmd.h"
#include "product_cli.h"
#include "test_helpers.h"

using json = nlohmann::json;

/* P1/P14: the dashboard's next-step logic (pure) and one end-to-end run on
 * an empty, isolated MEMBRANE home. */

static membrane_dashboard_t	base(void)
{
	membrane_dashboard_t	d = {};

	return (d);
}

static void	test_next_steps(void)
{
	membrane_dashboard_t	d = base();

	membrane_dashboard_next_steps(&d);
	TEST_ASSERT(d.next_count == 2 && std::string(d.next[0])
			== "membrane models recommend" && std::string(d.next[1])
			== "membrane use", "nothing installed => find a model first");
	d = base();
	d.installed_count = 2;
	membrane_dashboard_next_steps(&d);
	TEST_ASSERT(std::string(d.next[0]) == "membrane use",
		"installed but none selected => choose one");
	d = base();
	d.installed_count = 1;
	d.model_known = true;
	membrane_dashboard_next_steps(&d);
	TEST_ASSERT(std::string(d.next[0]) == "membrane serve",
		"model chosen, server stopped, no service => `serve`");
	d.service_installed = true;
	membrane_dashboard_next_steps(&d);
	TEST_ASSERT(std::string(d.next[0]) == "membrane service start",
		"service installed but stopped => `service start`");
	d.server_running = true;
	membrane_dashboard_next_steps(&d);
	TEST_ASSERT(std::string(d.next[0]) == "membrane chat"
		&& d.next_count == 2, "everything ready => chat");
}

static std::string	capture_json(int *rc)
{
	char	tmpl[] = "/tmp/membrane-dash-XXXXXX";
	int		fd = mkstemp(tmpl);

	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_dashboard_cmd_dispatch(true);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	FILE		*f = fopen(tmpl, "rb");
	std::string	out;
	char		buf[2048];
	size_t		n;

	while (f != NULL && (n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	if (f != NULL)
		fclose(f);
	unlink(tmpl);
	return (out);
}

static void	test_empty_home(void)
{
	char	tmpl[] = "/tmp/membrane-dash-home-XXXXXX";

	TEST_ASSERT(mkdtemp(tmpl) != NULL, "temp");
	std::string	dir = tmpl;

	setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
	setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(), 1);
	setenv("MEMBRANE_SYSTEMD_USER_DIR", dir.c_str(), 1);
	int		rc;
	json	j = json::parse(capture_json(&rc), nullptr, false);

	TEST_ASSERT(!j.is_discarded() && rc == MEMBRANE_EXIT_SUCCESS, "JSON");
	TEST_ASSERT(j["model"].is_null() && j["installed_models"] == 0
		&& j["server"]["running"] == false && j["adaptive"] == "off"
		&& j["runtime"] == "membrane-native", "an empty home");
	TEST_ASSERT(j["next"][0] == "membrane models recommend"
		&& j["next"][1] == "membrane use", "guided next steps");
	std::string	cmd = "rm -rf '" + dir + "'";

	if (system(cmd.c_str()) != 0)
		fprintf(stderr, "warning: cleanup failed\n");
}

int	main(void)
{
	test_next_steps();
	test_empty_home();
	printf("test_dashboard_cmd: all tests passed\n");
	return (0);
}
