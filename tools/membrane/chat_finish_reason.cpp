#include "chat_finish_reason.h"

const char	*membrane_chat_finish_reason(bool stop_matched, bool stopped_eog)
{
	if (stop_matched || stopped_eog)
		return ("stop");
	return ("length");
}
