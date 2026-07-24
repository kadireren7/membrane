#include <string.h>

#include "membrane/f16convert.h"

float	membrane_f16_to_f32(uint16_t h)
{
	uint32_t	sign;
	uint32_t	exp;
	uint32_t	mant;
	uint32_t	bits;
	float		out;

	sign = (uint32_t)(h & 0x8000u) << 16;
	exp = (h >> 10) & 0x1Fu;
	mant = h & 0x3FFu;
	if (exp == 0)
	{
		if (mant == 0)
			bits = sign;
		else
		{
			exp = 1;
			while ((mant & 0x400u) == 0)
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x3FFu;
			bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	}
	else if (exp == 0x1Fu)
		bits = sign | 0x7F800000u | (mant << 13);
	else
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	memcpy(&out, &bits, sizeof(out));
	return (out);
}

/* Rounds a 24-bit significand (implicit leading bit already ORed in)
 * right by `shift` bits, round-to-nearest-even, returning the rounded
 * quotient. `shift` must be in [1, 24]. */
static uint32_t	round_shift(uint32_t full_mant, unsigned shift)
{
	uint32_t	q;
	uint32_t	remainder;
	uint32_t	halfway;

	q = full_mant >> shift;
	remainder = full_mant & ((1u << shift) - 1u);
	halfway = 1u << (shift - 1);
	if (remainder > halfway || (remainder == halfway && (q & 1u)))
		q++;
	return (q);
}

static uint16_t	f16_subnormal(uint32_t sign, uint32_t full_mant, int h_exp)
{
	unsigned	shift;
	uint32_t	mant_h;

	shift = (unsigned)(14 - h_exp);
	if (shift > 24)
		return ((uint16_t)sign);
	mant_h = round_shift(full_mant, shift);
	if (mant_h >= 1024)
		return ((uint16_t)(sign | (1u << 10)));
	return ((uint16_t)(sign | mant_h));
}

static uint16_t	f16_normal(uint32_t sign, uint32_t mant_f, int h_exp)
{
	uint32_t	mant_h;
	uint32_t	remainder;

	mant_h = mant_f >> 13;
	remainder = mant_f & 0x1FFFu;
	if (remainder > 0x1000u || (remainder == 0x1000u && (mant_h & 1u)))
		mant_h++;
	if (mant_h == 0x400u)
	{
		mant_h = 0;
		h_exp++;
	}
	if (h_exp >= 31)
		return ((uint16_t)(sign | 0x7C00u));
	return ((uint16_t)(sign | ((uint32_t)h_exp << 10) | mant_h));
}

uint16_t	membrane_f32_to_f16(float f)
{
	uint32_t	x;
	uint32_t	sign;
	uint32_t	exp_f;
	uint32_t	mant_f;
	int			h_exp;

	memcpy(&x, &f, sizeof(x));
	sign = (x >> 16) & 0x8000u;
	exp_f = (x >> 23) & 0xFFu;
	mant_f = x & 0x7FFFFFu;
	if (exp_f == 0xFFu)
	{
		if (mant_f == 0)
			return ((uint16_t)(sign | 0x7C00u));
		return ((uint16_t)(sign | 0x7C00u | (mant_f >> 13 | 1u)));
	}
	if (exp_f == 0)
		return ((uint16_t)sign);
	h_exp = (int)exp_f - 127 + 15;
	if (h_exp >= 31)
		return ((uint16_t)(sign | 0x7C00u));
	if (h_exp <= 0)
		return (f16_subnormal(sign, mant_f | 0x800000u, h_exp));
	return (f16_normal(sign, mant_f, h_exp));
}
