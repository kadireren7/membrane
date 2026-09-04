#include "server_config.h"
#include "fs_util.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string	membrane_server_config_resolve_path(void)
{
	const char	*override_path = getenv("MEMBRANE_SERVER_CONFIG_PATH");

	if (override_path != NULL && override_path[0] != '\0')
		return (std::string(override_path));
	const char	*xdg_config_home = getenv("XDG_CONFIG_HOME");

	if (xdg_config_home != NULL && xdg_config_home[0] != '\0')
		return (std::string(xdg_config_home) + "/membrane/server.json");
	const char	*home = getenv("HOME");

	if (home != NULL && home[0] != '\0')
		return (std::string(home) + "/.config/membrane/server.json");
	return ("");
}

membrane_server_config_t	membrane_server_config_defaults(void)
{
	membrane_server_config_t	cfg;

	cfg.schema_version = MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION;
	cfg.listen_address = "127.0.0.1";
	cfg.port = 8642;
	cfg.default_model = "";
	cfg.log_level = "info";
	return (cfg);
}

bool	membrane_server_config_load(const std::string &path,
			membrane_server_config_t *out,
			membrane_server_config_error_t *err)
{
	*err = membrane_server_config_error_t();
	*out = membrane_server_config_defaults();
	FILE	*f = fopen(path.c_str(), "rb");

	if (f == NULL)
	{
		if (errno == ENOENT)
			return (true);	/* no config yet -- defaults, not an error */
		err->set = true;
		err->code = "IO_ERROR";
		err->message = std::string("could not open '") + path + "': "
			+ strerror(errno);
		return (false);
	}
	std::string	content;
	char		buf[4096];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		content.append(buf, n);
	fclose(f);
	json	root;

	try
	{
		root = json::parse(content);
	}
	catch (const json::parse_error &e)
	{
		err->set = true;
		err->code = "PARSE_ERROR";
		err->message = std::string("'") + path + "' is not valid JSON: "
			+ e.what();
		return (false);
	}
	if (!root.is_object() || !root.contains("schema_version"))
	{
		err->set = true;
		err->code = "PARSE_ERROR";
		err->message = std::string("'") + path + "' does not have the "
			"expected {schema_version, ...} shape";
		return (false);
	}
	int	schema = root.value("schema_version", 0);

	if (schema != MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION)
	{
		err->set = true;
		err->code = "UNSUPPORTED_SCHEMA";
		err->message = std::string("'") + path + "' has schema_version "
			+ std::to_string(schema) + ", this build only understands "
			+ std::to_string(MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION);
		return (false);
	}
	out->schema_version = schema;
	out->listen_address = root.value("listen_address",
		std::string("127.0.0.1"));
	out->port = root.value("port", 8642);
	out->default_model = root.value("default_model", std::string(""));
	out->log_level = root.value("log_level", std::string("info"));
	return (true);
}

bool	membrane_server_config_save(const std::string &path,
			const membrane_server_config_t &cfg,
			membrane_server_config_error_t *err)
{
	json	root;

	root["schema_version"] = cfg.schema_version;
	root["listen_address"] = cfg.listen_address;
	root["port"] = cfg.port;
	root["default_model"] = cfg.default_model;
	root["log_level"] = cfg.log_level;

	membrane_fs_error_t	fs_err;

	if (!membrane_atomic_write_file(path, root.dump(2), &fs_err))
	{
		*err = membrane_server_config_error_t();
		err->set = true;
		err->code = fs_err.code;
		err->message = fs_err.message;
		return (false);
	}
	*err = membrane_server_config_error_t();
	return (true);
}
