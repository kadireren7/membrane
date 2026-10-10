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
	std::string	home = membrane_resolve_home_dir();

	if (!home.empty())
		return (home + "/.config/membrane/server.json");
	return ("");
}

bool	membrane_server_overrides_empty(const membrane_server_overrides_t &o)
{
	return (!o.has_context && !o.has_kv_precision && !o.has_kv_placement
		&& !o.has_gpu_layers);
}

bool	membrane_server_overrides_equal(const membrane_server_overrides_t &a,
			const membrane_server_overrides_t &b)
{
	return (a.has_context == b.has_context
		&& (!a.has_context || a.context == b.context)
		&& a.has_kv_precision == b.has_kv_precision
		&& (!a.has_kv_precision || a.kv_precision == b.kv_precision)
		&& a.has_kv_placement == b.has_kv_placement
		&& (!a.has_kv_placement || a.kv_placement == b.kv_placement)
		&& a.has_gpu_layers == b.has_gpu_layers
		&& (!a.has_gpu_layers || a.gpu_layers == b.gpu_layers));
}

std::string	membrane_server_overrides_validate(
				const membrane_server_overrides_t &o)
{
	if (o.has_context && (o.context < 16 || o.context > (1u << 24)))
		return ("overrides.context must be between 16 and 16777216");
	if (o.has_kv_precision && o.kv_precision != "native"
		&& o.kv_precision != "q8" && o.kv_precision != "q5")
		return ("overrides.kv_precision must be native|q8|q5");
	if (o.has_kv_placement && o.kv_placement != "default"
		&& o.kv_placement != "cpu" && o.kv_placement != "gpu"
		&& o.kv_placement != "auto")
		return ("overrides.kv_placement must be default|cpu|gpu|auto");
	if (o.has_gpu_layers && o.gpu_layers < -2)
		return ("overrides.gpu_layers must be all, auto or N >= 0");
	return ("");
}

static json	overrides_to_json(const membrane_server_overrides_t &o)
{
	json	j = json::object();

	if (o.has_context)
		j["context"] = o.context;
	if (o.has_kv_precision)
		j["kv_precision"] = o.kv_precision;
	if (o.has_kv_placement)
		j["kv_placement"] = o.kv_placement;
	if (o.has_gpu_layers)
	{
		if (o.gpu_layers == -1)
			j["gpu_layers"] = "all";
		else if (o.gpu_layers == -2)
			j["gpu_layers"] = "auto";
		else
			j["gpu_layers"] = o.gpu_layers;
	}
	return (j);
}

/* Returns "" on success, else why the block is invalid (fail closed). */
static std::string	overrides_from_json(const json &j,
						membrane_server_overrides_t *o)
{
	*o = membrane_server_overrides_t();
	if (!j.is_object())
		return ("overrides must be an object");
	for (auto it = j.begin(); it != j.end(); ++it)
	{
		const std::string	&k = it.key();
		const json			&v = it.value();

		if (k == "context" && v.is_number_unsigned())
		{
			o->has_context = true;
			o->context = v.get<uint64_t>();
		}
		else if (k == "kv_precision" && v.is_string())
		{
			o->has_kv_precision = true;
			o->kv_precision = v.get<std::string>();
		}
		else if (k == "kv_placement" && v.is_string())
		{
			o->has_kv_placement = true;
			o->kv_placement = v.get<std::string>();
		}
		else if (k == "gpu_layers" && v.is_string()
			&& (v == "all" || v == "auto"))
		{
			o->has_gpu_layers = true;
			o->gpu_layers = (v == "all") ? -1 : -2;
		}
		else if (k == "gpu_layers" && v.is_number_unsigned()
			&& v.get<uint64_t>() < 100000)
		{
			o->has_gpu_layers = true;
			o->gpu_layers = (int32_t)v.get<uint64_t>();
		}
		else
			return ("overrides." + k + " is unknown or has a bad type");
	}
	return (membrane_server_overrides_validate(*o));
}

static json	adaptive_to_json(const membrane_server_adaptive_t &a)
{
	json	j;

	j["enabled"] = a.enabled;
	j["mode"] = a.mode;
	j["runtime"] = a.runtime;
	j["interval_s"] = a.interval_s;
	j["cooldown_s"] = a.cooldown_s;
	j["allow"] = a.allow;
	j["min_pressure"] = a.min_pressure;
	j["max_actions_per_window"] = a.max_actions_per_window;
	j["auto_restore"] = a.auto_restore;
	return (j);
}

/* Type-checks only; "" on success. */
static std::string	adaptive_from_json(const json &j,
						membrane_server_adaptive_t *a)
{
	*a = membrane_server_adaptive_t();
	if (!j.is_object())
		return ("adaptive must be an object");
	a->present = true;
	for (auto it = j.begin(); it != j.end(); ++it)
	{
		const std::string	&k = it.key();
		const json			&v = it.value();

		if (k == "enabled" && v.is_boolean())
			a->enabled = v.get<bool>();
		else if (k == "mode" && v.is_string())
			a->mode = v.get<std::string>();
		else if (k == "runtime" && v.is_string())
			a->runtime = v.get<std::string>();
		else if (k == "min_pressure" && v.is_string())
			a->min_pressure = v.get<std::string>();
		else if (k == "interval_s" && v.is_number_integer())
			a->interval_s = v.get<int32_t>();
		else if (k == "cooldown_s" && v.is_number_integer())
			a->cooldown_s = v.get<int32_t>();
		else if (k == "max_actions_per_window" && v.is_number_integer())
			a->max_actions_per_window = v.get<int32_t>();
		else if (k == "auto_restore" && v.is_boolean())
			a->auto_restore = v.get<bool>();
		else if (k == "allow" && v.is_array())
		{
			a->allow.clear();
			for (const auto &e : v)
			{
				if (!e.is_string())
					return ("adaptive.allow must be an array of strings");
				a->allow.push_back(e.get<std::string>());
			}
		}
		else
			return ("adaptive." + k + " is unknown or has a bad type");
	}
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
	if (root.contains("overrides"))
	{
		std::string	why = overrides_from_json(root["overrides"],
				&out->overrides);

		if (!why.empty())
		{
			err->set = true;
			err->code = "INVALID_OVERRIDES";
			err->message = std::string("'") + path + "': " + why;
			return (false);
		}
	}
	if (root.contains("adaptive"))
	{
		std::string	why = adaptive_from_json(root["adaptive"], &out->adaptive);

		if (!why.empty())
		{
			err->set = true;
			err->code = "INVALID_ADAPTIVE";
			err->message = std::string("'") + path + "': " + why;
			return (false);
		}
	}
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
	if (!membrane_server_overrides_empty(cfg.overrides))
	{
		std::string	why = membrane_server_overrides_validate(cfg.overrides);

		if (!why.empty())
		{
			*err = membrane_server_config_error_t();
			err->set = true;
			err->code = "INVALID_OVERRIDES";
			err->message = why;
			return (false);
		}
		root["overrides"] = overrides_to_json(cfg.overrides);
	}
	if (cfg.adaptive.present)
		root["adaptive"] = adaptive_to_json(cfg.adaptive);

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
