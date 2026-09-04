#include "utf8_stream.h"

size_t	membrane_utf8_incomplete_suffix_len(const std::string &s)
{
	size_t	n = s.size();
	/* Scans up to 4 bytes back -- NOT 3 -- because recognizing a
	 * COMPLETE 4-byte sequence (the longest valid UTF-8 form) requires
	 * seeing its own lead byte, which sits 4 positions from the end when
	 * the string's last character is one; a 3-byte-deep window would
	 * misidentify e.g. a complete 4-byte emoji's trailing 3 continuation
	 * bytes as an incomplete tail with "no lead byte found" (a real bug
	 * caught by test_utf8_stream.cpp's own 4-byte-emoji test). The
	 * INCOMPLETE tail this function actually reports is still bounded to
	 * at most 3 bytes (an incomplete sequence can never include its own
	 * complete self) -- the 4th byte of lookback exists purely to
	 * disambiguate "complete" from "incomplete", never to be counted
	 * into a returned incomplete length itself (back < expected_len can
	 * only be true for back in [1, expected_len-1], and expected_len is
	 * at most 4, so the returned value is always <= 3). */
	size_t	max_back = n < 4 ? n : 4;

	for (size_t back = 1; back <= max_back; ++back)
	{
		unsigned char	c = (unsigned char)s[n - back];

		if ((c & 0xC0) == 0x80)
			continue ;	/* a continuation byte -- keep looking further back
						 * for this sequence's own lead byte */
		size_t	expected_len;

		if ((c & 0x80) == 0x00)
			expected_len = 1;
		else if ((c & 0xE0) == 0xC0)
			expected_len = 2;
		else if ((c & 0xF0) == 0xE0)
			expected_len = 3;
		else if ((c & 0xF8) == 0xF0)
			expected_len = 4;
		else
			expected_len = 1;	/* not a valid UTF-8 lead byte -- treat as
								 * already "complete" rather than risk
								 * holding it back forever on malformed
								 * input */
		return (back < expected_len ? back : 0);
	}
	/* No lead byte found anywhere in the scanned window -- either the
	 * whole (short) string is stray continuation bytes (max_back < 4,
	 * including the empty-string case where max_back == 0 and nothing
	 * should be held back at all), or 4 continuation bytes in a row with
	 * no lead byte found (genuinely malformed: no valid UTF-8 lead byte
	 * requires more than 3 continuation bytes) -- either way, bounded to
	 * the same structural maximum as any other unresolved tail, never
	 * unbounded. */
	return (max_back < 3 ? max_back : 3);
}
