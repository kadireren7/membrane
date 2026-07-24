/*
 * Length-limited canonical Huffman coder (Phase 2.4). See
 * include/membrane/huffman.h for the stream format. The code-length limit
 * (15 bits) is enforced with the classic zlib-style Kraft repair on the
 * bit-length histogram, after which lengths are reassigned to symbols in
 * decreasing frequency order (most frequent gets the shortest code) and
 * canonical codes are handed out in (length, symbol) order.
 */

#include <stdlib.h>
#include <string.h>

#include "membrane/huffman.h"

# define HUF_MAXLEN MEMBRANE_HUFFMAN_MAXLEN
# define HUF_NSYM 256
# define HUF_NODES 512

typedef struct s_heap
{
	int				size;
	int				a[HUF_NODES];
	const uint64_t	*freq;
}	t_heap;

static void	heap_swap(t_heap *h, int i, int j)
{
	int	t;

	t = h->a[i];
	h->a[i] = h->a[j];
	h->a[j] = t;
}

static void	heap_up(t_heap *h, int i)
{
	int	p;

	while (i > 0)
	{
		p = (i - 1) / 2;
		if (h->freq[h->a[i]] >= h->freq[h->a[p]])
			break ;
		heap_swap(h, i, p);
		i = p;
	}
}

static void	heap_down(t_heap *h, int i)
{
	int	l;
	int	m;

	m = i;
	while (1)
	{
		l = 2 * i + 1;
		if (l < h->size && h->freq[h->a[l]] < h->freq[h->a[m]])
			m = l;
		if (l + 1 < h->size && h->freq[h->a[l + 1]] < h->freq[h->a[m]])
			m = l + 1;
		if (m == i)
			return ;
		heap_swap(h, i, m);
		i = m;
	}
}

static void	heap_push(t_heap *h, int id)
{
	h->a[h->size] = id;
	h->size += 1;
	heap_up(h, h->size - 1);
}

static int	heap_pop(t_heap *h)
{
	int	top;

	top = h->a[0];
	h->size -= 1;
	h->a[0] = h->a[h->size];
	heap_down(h, 0);
	return (top);
}

/* Builds the Huffman tree over used leaves and returns the raw (unlimited)
 * code length of every symbol via `depth`; unused symbols get 0. */
static int	huf_tree_depths(const uint64_t *freq, uint64_t *nf, int *parent,
				uint8_t *depth)
{
	t_heap	h;
	int		s;
	int		next;
	int		a;
	int		d;

	h.size = 0;
	h.freq = nf;
	s = 0;
	while (s < HUF_NSYM)
	{
		nf[s] = freq[s];
		if (freq[s] > 0)
			heap_push(&h, s);
		s++;
	}
	next = HUF_NSYM;
	while (h.size > 1)
	{
		a = heap_pop(&h);
		s = heap_pop(&h);
		nf[next] = nf[a] + nf[s];
		parent[a] = next;
		parent[s] = next;
		heap_push(&h, next++);
	}
	s = 0;
	while (s < HUF_NSYM)
	{
		d = 0;
		a = parent[s];
		while (freq[s] > 0 && a >= 0)
		{
			d++;
			a = parent[a];
		}
		depth[s] = (uint8_t)d;
		s++;
	}
	return (0);
}

/* Repairs a bit-length histogram (indexed 0..255) so no code exceeds
 * HUF_MAXLEN while the Kraft sum stays exactly full, preserving the number
 * of leaves. */
static void	huf_limit(uint32_t *bl_count, int maxdepth)
{
	int			l;
	uint64_t	k;
	uint64_t	target;

	l = HUF_MAXLEN + 1;
	while (l <= maxdepth)
	{
		bl_count[HUF_MAXLEN] += bl_count[l];
		bl_count[l] = 0;
		l++;
	}
	k = 0;
	l = 1;
	while (l <= HUF_MAXLEN)
	{
		k += (uint64_t)bl_count[l] << (HUF_MAXLEN - l);
		l++;
	}
	target = (uint64_t)1 << HUF_MAXLEN;
	while (k > target)
	{
		l = HUF_MAXLEN - 1;
		while (l >= 1 && bl_count[l] == 0)
			l--;
		if (l < 1)
			break ;
		bl_count[l]--;
		bl_count[l + 1] += 2;
		bl_count[HUF_MAXLEN]--;
		k--;
	}
}

/* Insertion-sorts the `n` used symbols by frequency descending (ties by
 * symbol value ascending), so the shortest codes go to the hottest bytes. */
static void	huf_sort_by_freq(int *sym, int n, const uint64_t *freq)
{
	int	i;
	int	j;
	int	key;

	i = 1;
	while (i < n)
	{
		key = sym[i];
		j = i - 1;
		while (j >= 0 && freq[sym[j]] < freq[key])
		{
			sym[j + 1] = sym[j];
			j--;
		}
		sym[j + 1] = key;
		i++;
	}
}

/* Fills len[256] with a valid canonical, length-limited code length per
 * symbol (0 = unused). Handles the empty and single-symbol degenerate
 * cases explicitly. */
static void	build_lengths(const uint64_t *freq, uint8_t *len)
{
	uint64_t	nf[HUF_NODES];
	int			parent[HUF_NODES];
	uint8_t		depth[HUF_NSYM];
	uint32_t	bl_count[HUF_NSYM];
	int			sym[HUF_NSYM];
	int			s;
	int			n;
	int			maxd;
	int			l;
	int			idx;

	memset(len, 0, HUF_NSYM);
	memset(bl_count, 0, sizeof(bl_count));
	n = 0;
	s = 0;
	while (s < HUF_NSYM)
	{
		if (freq[s] > 0)
			sym[n++] = s;
		s++;
	}
	if (n == 0)
		return ;
	if (n == 1)
	{
		len[sym[0]] = 1;
		return ;
	}
	l = 0;
	while (l < HUF_NODES)
		parent[l++] = -1;
	huf_tree_depths(freq, nf, parent, depth);
	maxd = 0;
	s = 0;
	while (s < HUF_NSYM)
	{
		if (freq[s] > 0)
		{
			bl_count[depth[s]] += 1;
			if (depth[s] > maxd)
				maxd = depth[s];
		}
		s++;
	}
	huf_limit(bl_count, maxd);
	huf_sort_by_freq(sym, n, freq);
	idx = 0;
	l = 1;
	while (l <= HUF_MAXLEN)
	{
		s = 0;
		while ((uint32_t)s < bl_count[l])
		{
			len[sym[idx++]] = (uint8_t)l;
			s++;
		}
		l++;
	}
}

/* Assigns canonical codes from the code lengths (DEFLATE convention). */
static void	canonical_codes(const uint8_t *len, uint32_t *code)
{
	uint32_t	bl_count[HUF_MAXLEN + 1];
	uint32_t	next[HUF_MAXLEN + 1];
	uint32_t	c;
	int			l;
	int			s;

	memset(bl_count, 0, sizeof(bl_count));
	s = 0;
	while (s < HUF_NSYM)
		bl_count[len[s++]] += 1;
	bl_count[0] = 0;
	c = 0;
	l = 1;
	while (l <= HUF_MAXLEN)
	{
		c = (c + bl_count[l - 1]) << 1;
		next[l] = c;
		l++;
	}
	s = 0;
	while (s < HUF_NSYM)
	{
		if (len[s] != 0)
			code[s] = next[len[s]]++;
		s++;
	}
}

static void	put_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static uint32_t	get_le32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static void	pack_lengths(const uint8_t *len, uint8_t *out)
{
	int	i;

	i = 0;
	while (i < HUF_NSYM / 2)
	{
		out[i] = (uint8_t)((len[2 * i] & 0x0F) | ((len[2 * i + 1] & 0x0F) << 4));
		i++;
	}
}

static void	unpack_lengths(const uint8_t *in, uint8_t *len)
{
	int	i;

	i = 0;
	while (i < HUF_NSYM / 2)
	{
		len[2 * i] = (uint8_t)(in[i] & 0x0F);
		len[2 * i + 1] = (uint8_t)(in[i] >> 4);
		i++;
	}
}

typedef struct s_bitw
{
	uint8_t	*buf;
	size_t	cap;
	size_t	pos;
	uint32_t	acc;
	int		nb;
}	t_bitw;

static int	bw_put(t_bitw *w, uint32_t code, int nbits)
{
	int	i;

	i = nbits - 1;
	while (i >= 0)
	{
		w->acc = (w->acc << 1) | ((code >> i) & 1u);
		w->nb += 1;
		if (w->nb == 8)
		{
			if (w->pos >= w->cap)
				return (-1);
			w->buf[w->pos++] = (uint8_t)w->acc;
			w->acc = 0;
			w->nb = 0;
		}
		i--;
	}
	return (0);
}

static int	bw_flush(t_bitw *w)
{
	if (w->nb == 0)
		return (0);
	if (w->pos >= w->cap)
		return (-1);
	w->buf[w->pos++] = (uint8_t)(w->acc << (8 - w->nb));
	return (0);
}

size_t	membrane_huffman_bound(size_t len)
{
	if (len > (SIZE_MAX - MEMBRANE_HUFFMAN_HEADER - 1) / 2)
		return (SIZE_MAX);
	return (MEMBRANE_HUFFMAN_HEADER + 2 * len + 1);
}

static membrane_status_t	huf_emit(const uint8_t *src, size_t len,
								const uint8_t *lentab, const uint32_t *code,
								t_bitw *w)
{
	size_t	i;

	i = 0;
	while (i < len)
	{
		if (bw_put(w, code[src[i]], lentab[src[i]]) != 0)
			return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
		i++;
	}
	if (bw_flush(w) != 0)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_huffman_compress(const uint8_t *src, size_t len,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	uint64_t			freq[HUF_NSYM];
	uint8_t				lentab[HUF_NSYM];
	uint32_t			code[HUF_NSYM];
	t_bitw				w;
	membrane_status_t	st;
	size_t				i;

	if ((src == NULL && len > 0) || out == NULL || out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (out_cap < 4)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	put_le32(out, (uint32_t)len);
	if (len == 0)
		return (*out_len = 4, MEMBRANE_OK);
	if (len > 0xFFFFFFFFu || out_cap < MEMBRANE_HUFFMAN_HEADER)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	memset(freq, 0, sizeof(freq));
	i = 0;
	while (i < len)
		freq[src[i++]] += 1;
	build_lengths(freq, lentab);
	canonical_codes(lentab, code);
	pack_lengths(lentab, out + 4);
	w.buf = out;
	w.cap = out_cap;
	w.pos = MEMBRANE_HUFFMAN_HEADER;
	w.acc = 0;
	w.nb = 0;
	st = huf_emit(src, len, lentab, code, &w);
	if (st != MEMBRANE_OK)
		return (st);
	*out_len = w.pos;
	return (MEMBRANE_OK);
}

typedef struct s_bitr
{
	const uint8_t	*buf;
	size_t			len;
	size_t			pos;
	int				bit;
}	t_bitr;

static int	br_get(t_bitr *r)
{
	int	v;

	if (r->pos >= r->len)
		return (-1);
	v = (r->buf[r->pos] >> (7 - r->bit)) & 1;
	r->bit += 1;
	if (r->bit == 8)
	{
		r->bit = 0;
		r->pos += 1;
	}
	return (v);
}

/* Validates the code-length table is not over-subscribed and builds the
 * canonical decode order. Returns the symbol count, or -1 if malformed. */
static int	huf_prepare(const uint8_t *lentab, uint32_t *bl_count,
				uint8_t *sorted)
{
	uint64_t	k;
	int			l;
	int			s;
	int			total;

	memset(bl_count, 0, sizeof(uint32_t) * (HUF_MAXLEN + 1));
	s = 0;
	while (s < HUF_NSYM)
	{
		if (lentab[s] > HUF_MAXLEN)
			return (-1);
		bl_count[lentab[s]] += 1;
		s++;
	}
	bl_count[0] = 0;
	k = 0;
	l = 1;
	while (l <= HUF_MAXLEN)
	{
		k += (uint64_t)bl_count[l] << (HUF_MAXLEN - l);
		l++;
	}
	if (k > ((uint64_t)1 << HUF_MAXLEN))
		return (-1);
	total = 0;
	l = 1;
	while (l <= HUF_MAXLEN)
	{
		s = 0;
		while (s < HUF_NSYM)
		{
			if (lentab[s] == l)
				sorted[total++] = (uint8_t)s;
			s++;
		}
		l++;
	}
	return (total);
}

/* Decodes one symbol via the puff-style canonical walk. Returns the symbol,
 * -1 on an invalid (over-long) code, -2 on a truncated bitstream. */
static int	huf_decode_sym(t_bitr *r, const uint32_t *bl_count,
				const uint8_t *sorted)
{
	int		len;
	int		bit;
	long	code;
	long	first;
	int		index;

	code = 0;
	first = 0;
	index = 0;
	len = 1;
	while (len <= HUF_MAXLEN)
	{
		bit = br_get(r);
		if (bit < 0)
			return (-2);
		code |= bit;
		if ((unsigned long)(code - first) < bl_count[len])
			return (sorted[index + (code - first)]);
		index += bl_count[len];
		first += bl_count[len];
		first <<= 1;
		code <<= 1;
		len++;
	}
	return (-1);
}

membrane_status_t	membrane_huffman_decompress(const uint8_t *src, size_t len,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	uint8_t		lentab[HUF_NSYM];
	uint32_t	bl_count[HUF_MAXLEN + 1];
	uint8_t		sorted[HUF_NSYM];
	t_bitr		r;
	uint32_t	n;
	uint32_t	i;
	int			sym;

	if ((src == NULL && len > 0) || (out == NULL && out_cap > 0)
		|| out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (len < 4)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	n = get_le32(src);
	if (n == 0)
		return (*out_len = 0, MEMBRANE_OK);
	if (out_cap < n)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (len < MEMBRANE_HUFFMAN_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	unpack_lengths(src + 4, lentab);
	if (huf_prepare(lentab, bl_count, sorted) <= 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	r.buf = src + MEMBRANE_HUFFMAN_HEADER;
	r.len = len - MEMBRANE_HUFFMAN_HEADER;
	r.pos = 0;
	r.bit = 0;
	i = 0;
	while (i < n)
	{
		sym = huf_decode_sym(&r, bl_count, sorted);
		if (sym < 0)
			return (MEMBRANE_ERR_CORRUPT_DATA);
		out[i++] = (uint8_t)sym;
	}
	*out_len = n;
	return (MEMBRANE_OK);
}
