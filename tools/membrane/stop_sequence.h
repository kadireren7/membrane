#ifndef MEMBRANE_STOP_SEQUENCE_H
# define MEMBRANE_STOP_SEQUENCE_H

# include <cstddef>
# include <string>
# include <vector>

/*
 * OpenAI-compatible "stop" string matching, as a pure module with no
 * httplib/llama dependency -- same reasoning as utf8_stream.h, and for
 * the same reason: server.cpp's streaming path can only be exercised
 * end-to-end with a real model and a real HTTP round trip, so the
 * decision logic lives here instead, where it is unit-testable under
 * the default llama-free build.
 *
 * Two functions, both operating on the FULL accumulated response text
 * (never one token's own piece in isolation -- a stop string routinely
 * spans several BPE tokens).
 */

/*
 * Finds the earliest occurrence of any string in `stops` within `acc`.
 * Returns true and sets *out_pos to that offset; returns false and
 * leaves *out_pos untouched when nothing matches. Empty stop strings
 * are ignored (they would otherwise match at offset 0 and terminate
 * every response immediately).
 */
bool	membrane_stop_find_match(const std::string &acc,
			const std::vector<std::string> &stops, size_t *out_pos);

/*
 * Returns how many of `acc`'s own TRAILING bytes are a proper prefix of
 * some stop string -- bytes that must NOT be emitted yet, because the
 * next token could complete a stop match that begins inside them.
 * Returns 0 when no such overlap exists.
 *
 * This is the stop-sequence analogue of
 * membrane_utf8_incomplete_suffix_len(): both answer "how much of what
 * I am holding is not safe to send yet". Bounded by the longest stop
 * string's length minus one, so a caller can never be held back
 * indefinitely. Only a PROPER prefix counts -- a full match is
 * membrane_stop_find_match()'s job, not this one's.
 */
size_t	membrane_stop_pending_suffix_len(const std::string &acc,
			const std::vector<std::string> &stops);

#endif
