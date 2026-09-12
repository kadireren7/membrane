#include "chat_stream_parser.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static void	handle_line(const std::string &line,
				const std::function<void(const std::string &)> &on_content,
				const std::function<void(const std::string &,
					const std::string &)> &on_error,
				const std::function<void(const std::string &)>
					&on_finish_reason,
				const std::function<void(void)> &on_done)
{
	if (line.empty() || line.rfind("data: ", 0) != 0)
		return ;

	std::string	payload = line.substr(6);

	if (payload == "[DONE]")
	{
		if (on_done)
			on_done();
		return ;
	}

	json	ev;

	try
	{
		ev = json::parse(payload);
	}
	catch (const json::parse_error &)
	{
		return ;
	}
	if (!ev.is_object())
		return ;
	if (ev.contains("error") && ev["error"].is_object())
	{
		if (on_error)
			on_error(ev["error"].value("code", std::string("ERROR")),
				ev["error"].value("message",
					std::string("request failed")));
		return ;
	}
	if (!ev.contains("choices") || !ev["choices"].is_array()
		|| ev["choices"].empty())
		return ;

	const json	&choice = ev["choices"][0];

	if (choice.contains("delta") && choice["delta"].is_object()
		&& choice["delta"].contains("content")
		&& choice["delta"]["content"].is_string())
	{
		std::string	text = choice["delta"]["content"].get<std::string>();

		if (!text.empty() && on_content)
			on_content(text);
	}
	if (choice.contains("finish_reason") && choice["finish_reason"].is_string()
		&& on_finish_reason)
		on_finish_reason(choice["finish_reason"].get<std::string>());
}

void	membrane_chat_stream_feed(membrane_chat_stream_parser_t *parser,
				const char *data, size_t len,
				const std::function<void(const std::string &)> &on_content,
				const std::function<void(const std::string &,
					const std::string &)> &on_error,
				const std::function<void(const std::string &)>
					&on_finish_reason,
				const std::function<void(void)> &on_done)
{
	parser->pending.append(data, len);

	size_t	pos;

	while ((pos = parser->pending.find('\n')) != std::string::npos)
	{
		std::string	line = parser->pending.substr(0, pos);

		parser->pending.erase(0, pos + 1);
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		handle_line(line, on_content, on_error, on_finish_reason, on_done);
	}
}
