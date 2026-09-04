#ifndef MEMBRANE_UTF8_STREAM_H
# define MEMBRANE_UTF8_STREAM_H

# include <cstddef>
# include <string>

/*
 * Mega Phase B, PR B2, Section 21 of the task: a single BPE token's own
 * bytes do not always align to a full UTF-8 character (common with
 * multi-byte scripts -- CJK, emoji, accented Latin) -- streaming a piece
 * the instant it arrives can emit a truncated/invalid byte sequence over
 * the wire. Pulled out of server.cpp into its own pure module (no
 * httplib/llama dependency) specifically so this logic is unit-testable
 * without a real model or a real HTTP round trip.
 */

/* Returns how many of `s`'s own TRAILING bytes are part of a UTF-8
 * sequence that is not yet complete (0 if `s` already ends on a clean
 * boundary, including the empty string). Structurally bounded to at
 * most 3 (the longest possible incomplete tail of a 4-byte UTF-8
 * sequence) -- never an unbounded scan, never a way to get permanently
 * stuck accumulating on malformed input (an invalid lead byte is
 * treated as "already complete" rather than held back forever). */
size_t	membrane_utf8_incomplete_suffix_len(const std::string &s);

#endif
