#include <cstdio>
#include <string>

#include "sha256.h"
#include "test_helpers.h"

/* FIPS 180-4 / NIST known-answer vectors. */

static void	test_vectors(void)
{
	TEST_ASSERT(membrane_sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427"
		"ae41e4649b934ca495991b7852b855", "empty");
	TEST_ASSERT(membrane_sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223"
		"b00361a396177a9cb410ff61f20015ad", "abc");
	const char	*two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

	TEST_ASSERT(membrane_sha256_hex(two, 56) == "248d6a61d20638b8e5c026930c3e6039"
		"a33ce45964ff2167f6ecedd419db06c1", "two-block message");
}

static void	test_streaming_matches_oneshot(void)
{
	std::string			big(1000003, 'a');
	membrane_sha256_t	c;

	membrane_sha256_init(&c);
	for (size_t i = 0; i < big.size(); i += 777)
		membrane_sha256_update(&c, big.data() + i,
			big.size() - i < 777 ? big.size() - i : 777);
	TEST_ASSERT(membrane_sha256_final_hex(&c) == membrane_sha256_hex(big.data(),
		big.size()), "chunking never changes the digest");
	std::string	million(1000000, 'a');

	TEST_ASSERT(membrane_sha256_hex(million.data(), million.size())
		== "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
		"one million 'a'");
}

static void	test_file(void)
{
	std::string	path = std::string("test_sha256_tmp_") + std::to_string(
		(long)sizeof(void *)) + ".bin";
	FILE		*f = fopen(path.c_str(), "wb");

	TEST_ASSERT(f != NULL, "temp file");
	fwrite("abc", 1, 3, f);
	fclose(f);
	std::string	hex;

	TEST_ASSERT(membrane_sha256_file(path, &hex) && hex == "ba7816bf8f01cfea4141"
		"40de5dae2223b00361a396177a9cb410ff61f20015ad", "file digest");
	remove(path.c_str());
	TEST_ASSERT(!membrane_sha256_file("definitely/not/here.bin", &hex),
		"unreadable file is an error, not an empty digest");
}

int	main(void)
{
	test_vectors();
	test_streaming_matches_oneshot();
	test_file();
	printf("test_sha256: all tests passed\n");
	return (0);
}
