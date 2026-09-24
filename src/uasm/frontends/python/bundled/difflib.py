"""`difflib`, as ordinary Python this compiler compiles.

COVERAGE: `SequenceMatcher` -- construction with `isjunk`, `a`, `b` and
`autojunk`, `set_seqs`/`set_seq1`/`set_seq2`, `find_longest_match` (with and
without bounds), `get_matching_blocks`, `get_opcodes`, `get_grouped_opcodes`,
`ratio`, `quick_ratio`, `real_quick_ratio`, and the documented attributes
`b2j`, `bjunk` and `bpopular`; `Match`; `get_close_matches`; `Differ` and
`ndiff`, with intraline `?` marking; `restore`; `unified_diff`,
`context_diff` and `diff_bytes`; `IS_LINE_JUNK` and `IS_CHARACTER_JUNK`.

NOT COVERED: `HtmlDiff` -- two thirds of a module's worth of table layout for
a side-by-side HTML view -- and the private `_mdiff` generator that exists
only to feed it. `SequenceMatcher[str]` (`__class_getitem__`) is not
provided either.

## The algorithm is the specification

What a program sees -- a ratio, an opcode list, where an `ndiff` puts its
`?` marks -- is decided by which matching block is found FIRST, so the
search below is CPython's search and not merely an equivalent one:

  * The longest junk-free block wins; among equally long ones, the one that
    starts earliest in `a`, and among those, earliest in `b`. That is what
    the strict `>` in `find_longest_match` enforces, and a `>=` would find
    the LAST one instead.
  * A block is then grown by equal non-junk elements on both sides -- the
    POPULAR ones, which the index leaves out -- and after that by equal junk.
  * The two unknown regions either side of it are searched the same way,
    and blocks that turn out adjacent are merged.
  * A second sequence of 200 or more elements treats as junk anything that
    makes up more than 1% of it (`autojunk`), counted as `n // 100 + 1`.

`Differ` synchronises two replaced blocks of lines on their most similar
pair -- a ratio above 0.74999, searched within ten lines of the diagonal --
which is CPython 3.14's search, not the recursive one of earlier releases.

A QUIRK KEPT ON PURPOSE: `get_grouped_opcodes` trims the first and last
opcode of the list `get_opcodes` caches, IN PLACE, so `get_opcodes()` asked
afterwards answers the trimmed list. That is CPython's behaviour, measured,
and a program that asks in that order sees it.
"""

import heapq as _heapq
from collections import namedtuple as _namedtuple

__all__ = ['get_close_matches', 'ndiff', 'restore', 'SequenceMatcher',
           'Differ', 'IS_CHARACTER_JUNK', 'IS_LINE_JUNK', 'context_diff',
           'unified_diff', 'diff_bytes', 'Match']

Match = _namedtuple('Match', 'a b size')


def _calculate_ratio(matches, length):
    """2M/T, and 1.0 for two empty sequences, which are identical."""
    if length:
        return 2.0 * matches / length
    return 1.0


class SequenceMatcher:
    """A flexible class for comparing pairs of sequences of any type, so long
    as the sequence elements are hashable.

    See the module docstring for the order the search is made in, which is
    the whole of what makes two implementations agree.
    """

    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a, b)

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        # THE SAME OBJECT IS NOT A CHANGE: what was computed for it stands.
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b):
        # `b` IS THE ONE THAT IS INDEXED, which is why comparing one
        # sequence against many is `set_seq2` once and `set_seq1` per
        # candidate -- `get_close_matches` below is written that way.
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self._index_b()

    def _index_b(self):
        """`b2j`: every element of `b` and where it occurs, less junk and
        less what `autojunk` calls popular.

        `isjunk` IS ASKED ONCE PER DISTINCT ELEMENT, in the order each first
        appears, and never for an element that is absent -- a program that
        counts its calls sees exactly that many."""
        b = self.b
        where = {}
        for i in range(len(b)):
            elt = b[i]
            found = where.get(elt)
            if found is None:
                where[elt] = [i]
            else:
                found.append(i)
        junk = set()
        if self.isjunk:
            for elt in list(where.keys()):
                if self.isjunk(elt):
                    junk.add(elt)
            for elt in junk:
                del where[elt]
        popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            most = n // 100 + 1
            for elt, idxs in where.items():
                if len(idxs) > most:
                    popular.add(elt)
            for elt in popular:
                del where[elt]
        self.b2j = where
        self.bjunk = junk
        self.bpopular = popular

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        """The longest matching block in a[alo:ahi] and b[blo:bhi], as a
        `Match`; `Match(alo, blo, 0)` when nothing matches."""
        a, b, index, junk = self.a, self.b, self.b2j, self.bjunk
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0
        # `run[j]` IS THE LENGTH OF THE JUNK-FREE MATCH ending at a[i - 1]
        # and b[j], so extending it by a[i] == b[j + 1] is one lookup. Junk
        # and popular elements are not in the index at all, so a block can
        # neither start nor continue through one here.
        run = {}
        for i in range(alo, ahi):
            nxt = {}
            for j in index.get(a[i], ()):
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = run.get(j - 1, 0) + 1
                nxt[j] = k
                if k > bestsize:
                    besti, bestj, bestsize = i - k + 1, j - k + 1, k
            run = nxt
        # GROWN BY WHAT THE INDEX LEFT OUT: first equal elements that are
        # not junk -- the popular ones -- then equal junk, on both sides. In
        # an empty match that is the only kind of match there can be.
        while besti > alo and bestj > blo \
                and b[bestj - 1] not in junk \
                and a[besti - 1] == b[bestj - 1]:
            besti, bestj, bestsize = besti - 1, bestj - 1, bestsize + 1
        while besti + bestsize < ahi and bestj + bestsize < bhi \
                and b[bestj + bestsize] not in junk \
                and a[besti + bestsize] == b[bestj + bestsize]:
            bestsize += 1
        while besti > alo and bestj > blo \
                and b[bestj - 1] in junk \
                and a[besti - 1] == b[bestj - 1]:
            besti, bestj, bestsize = besti - 1, bestj - 1, bestsize + 1
        while besti + bestsize < ahi and bestj + bestsize < bhi \
                and b[bestj + bestsize] in junk \
                and a[besti + bestsize] == b[bestj + bestsize]:
            bestsize += 1
        return Match(besti, bestj, bestsize)

    def get_matching_blocks(self):
        """Every matching block, ascending, adjacent ones merged, ending in
        the sentinel `Match(len(a), len(b), 0)`."""
        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)
        # A WORK LIST AND NOT RECURSION: two long sequences with little in
        # common split into many small regions, and the depth would follow.
        # The order regions are taken in does not matter -- which blocks
        # exist is fixed by where each split falls -- so they are sorted at
        # the end.
        pending = [(0, la, 0, lb)]
        found = []
        while pending:
            alo, ahi, blo, bhi = pending.pop()
            i, j, k = x = self.find_longest_match(alo, ahi, blo, bhi)
            if k:
                found.append(x)
                if alo < i and blo < j:
                    pending.append((alo, i, blo, j))
                if i + k < ahi and j + k < bhi:
                    pending.append((i + k, ahi, j + k, bhi))
        found.sort()
        merged = []
        i1 = j1 = k1 = 0
        for i2, j2, k2 in found:
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    merged.append(Match(i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            merged.append(Match(i1, j1, k1))
        merged.append(Match(la, lb, 0))
        self.matching_blocks = merged
        return merged

    def get_opcodes(self):
        """How to turn `a` into `b`, as (tag, i1, i2, j1, j2) tuples whose
        ranges tile both sequences."""
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        answer = []
        self.opcodes = answer
        for ai, bj, size in self.get_matching_blocks():
            if i < ai and j < bj:
                answer.append(('replace', i, ai, j, bj))
            elif i < ai:
                answer.append(('delete', i, ai, j, bj))
            elif j < bj:
                answer.append(('insert', i, ai, j, bj))
            i, j = ai + size, bj + size
            if size:
                answer.append(('equal', ai, i, bj, j))
        return answer

    def get_grouped_opcodes(self, n=3):
        """The opcodes in clusters, each with up to `n` lines of unchanged
        context either side -- the hunks of a unified diff."""
        codes = self.get_opcodes()
        if not codes:
            codes = [('equal', 0, 1, 0, 1)]
        # IN PLACE, on the cached list: see the module docstring.
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2 - n), i2, max(j1, j2 - n), j2
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)
        # A RUN OF MORE THAN 2n UNCHANGED LINES SPLITS THE GROUP: n of them
        # close the one before and n open the one after.
        wide = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            if tag == 'equal' and i2 - i1 > wide:
                group.append((tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)))
                yield group
                group = []
                i1, j1 = max(i1, i2 - n), max(j1, j2 - n)
            group.append((tag, i1, i2, j1, j2))
        if group and not (len(group) == 1 and group[0][0] == 'equal'):
            yield group

    def ratio(self):
        """2M/T, over the matching blocks."""
        matches = 0
        for block in self.get_matching_blocks():
            matches += block[-1]
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self):
        """An upper bound on `ratio`: the two sequences as MULTISETS, so how
        many of `a`'s elements `b` has, regardless of order."""
        if self.fullbcount is None:
            counts = {}
            for elt in self.b:
                counts[elt] = counts.get(elt, 0) + 1
            self.fullbcount = counts
        counts = self.fullbcount
        left = {}
        matches = 0
        for elt in self.a:
            if elt in left:
                have = left[elt]
            else:
                have = counts.get(elt, 0)
            left[elt] = have - 1
            if have > 0:
                matches += 1
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def real_quick_ratio(self):
        """An upper bound on `quick_ratio`: no more matches than the shorter
        sequence has elements."""
        la, lb = len(self.a), len(self.b)
        return _calculate_ratio(min(la, lb), la + lb)


def get_close_matches(word, possibilities, n=3, cutoff=0.6):
    """The best "good enough" matches for `word`, most similar first.

    THE CHEAP BOUNDS ARE ASKED FIRST, and a candidate only reaches `ratio`
    when neither rules it out -- which changes nothing about the answer and
    everything about the cost. Ties in score are broken by the candidate
    itself, larger first, because the ranking is `heapq.nlargest` over
    (score, candidate) pairs."""
    if not n > 0:
        raise ValueError('n must be > 0: %r' % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError('cutoff must be in [0.0, 1.0]: %r' % (cutoff,))
    scored = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for x in possibilities:
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff \
                and s.ratio() >= cutoff:
            scored.append((s.ratio(), x))
    return [x for score, x in _heapq.nlargest(n, scored)]


def _keep_original_ws(s, tag_s):
    """The tag line with the ORIGINAL whitespace under each untagged space,
    so a `?` line lines up under a line that holds tabs."""
    out = []
    for c, tag_c in zip(s, tag_s):
        out.append(c if tag_c == ' ' and c.isspace() else tag_c)
    return ''.join(out)


class Differ:
    """Human-readable deltas between sequences of lines: `- ` for a line
    only in the first, `+ ` only in the second, two spaces for both, and
    `? ` guide lines marking what changed inside a pair of similar lines."""

    def __init__(self, linejunk=None, charjunk=None):
        self.linejunk = linejunk
        self.charjunk = charjunk

    def compare(self, a, b):
        cruncher = SequenceMatcher(self.linejunk, a, b)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if tag == 'replace':
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            elif tag == 'delete':
                g = self._dump('-', a, alo, ahi)
            elif tag == 'insert':
                g = self._dump('+', b, blo, bhi)
            elif tag == 'equal':
                g = self._dump(' ', a, alo, ahi)
            else:
                raise ValueError('unknown tag %r' % (tag,))
            yield from g

    def _dump(self, tag, x, lo, hi):
        # `%s` AND NOT `+`: a sequence of numbers compares as well as one of
        # lines does, and CPython prints the number.
        for i in range(lo, hi):
            yield '%s %s' % (tag, x[i])

    def _plain_replace(self, a, alo, ahi, b, blo, bhi):
        # THE SHORTER BLOCK FIRST, so a one-line change beside a long one
        # is read before the long one scrolls it away.
        if bhi - blo < ahi - alo:
            yield from self._dump('+', b, blo, bhi)
            yield from self._dump('-', a, alo, ahi)
        else:
            yield from self._dump('-', a, alo, ahi)
            yield from self._dump('+', b, blo, bhi)

    def _fancy_replace(self, a, alo, ahi, b, blo, bhi):
        """A replaced block, synchronised on its most similar pairs of lines.

        FOR EACH LINE OF `b`, the lines of `a` within ten of the diagonal
        that have not been passed yet are scored, cheapest bound first, and
        the best one above 0.74999 becomes a synch point: what lies before
        it is a plain replace, and the pair itself gets `?` lines."""
        cutoff = 0.74999
        cruncher = SequenceMatcher(self.charjunk)
        window = 10
        best_i = best_j = None
        dump_i, dump_j = alo, blo
        for j in range(blo, bhi):
            cruncher.set_seq2(b[j])
            aequiv = alo + (j - blo)
            lo = max(aequiv - window, dump_i)
            hi = min(aequiv + window + 1, ahi)
            if lo >= hi:
                break
            best_ratio = cutoff
            for i in range(lo, hi):
                cruncher.set_seq1(a[i])
                if cruncher.real_quick_ratio() > best_ratio \
                        and cruncher.quick_ratio() > best_ratio \
                        and cruncher.ratio() > best_ratio:
                    best_i, best_j, best_ratio = i, j, cruncher.ratio()
            if best_i is None:
                continue
            yield from self._fancy_helper(a, dump_i, best_i, b, dump_j, best_j)
            aelt, belt = a[best_i], b[best_j]
            if aelt != belt:
                atags = btags = ''
                cruncher.set_seqs(aelt, belt)
                for tag, ai1, ai2, bj1, bj2 in cruncher.get_opcodes():
                    la, lb = ai2 - ai1, bj2 - bj1
                    if tag == 'replace':
                        atags += '^' * la
                        btags += '^' * lb
                    elif tag == 'delete':
                        atags += '-' * la
                    elif tag == 'insert':
                        btags += '+' * lb
                    elif tag == 'equal':
                        atags += ' ' * la
                        btags += ' ' * lb
                    else:
                        raise ValueError('unknown tag %r' % (tag,))
                yield from self._qformat(aelt, belt, atags, btags)
            else:
                yield '  ' + aelt
            dump_i, dump_j = best_i + 1, best_j + 1
            best_i = best_j = None
        yield from self._fancy_helper(a, dump_i, ahi, b, dump_j, bhi)

    def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
        if alo < ahi:
            if blo < bhi:
                yield from self._plain_replace(a, alo, ahi, b, blo, bhi)
            else:
                yield from self._dump('-', a, alo, ahi)
        elif blo < bhi:
            yield from self._dump('+', b, blo, bhi)

    def _qformat(self, aline, bline, atags, btags):
        atags = _keep_original_ws(aline, atags).rstrip()
        btags = _keep_original_ws(bline, btags).rstrip()
        yield '- ' + aline
        if atags:
            yield '? ' + atags + '\n'
        yield '+ ' + bline
        if btags:
            yield '? ' + btags + '\n'


def IS_LINE_JUNK(line, pat=None):
    """True for a line that is blank or holds one `#`.

    `line.strip() in '#'` IS THE WHOLE TEST, and it is a substring test: the
    empty string and `#` are the only strings that are in `'#'`."""
    if pat is None:
        return line.strip() in '#'
    return pat(line) is not None


def IS_CHARACTER_JUNK(ch, ws=' \t'):
    """True for a space or a tab -- never a newline."""
    return ch in ws


def _format_range_unified(start, stop):
    """`start,length` as a unified diff numbers lines: one-based, the length
    omitted when it is 1, and an empty range named by the line BEFORE it."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return str(beginning)
    if not length:
        beginning -= 1
    return str(beginning) + ',' + str(length)


def _format_range_context(start, stop):
    """`first,last` as a context diff numbers lines, with the same rule for
    an empty range and a single line written once."""
    beginning = start + 1
    length = stop - start
    if not length:
        beginning -= 1
    if length <= 1:
        return str(beginning)
    return str(beginning) + ',' + str(beginning + length - 1)


def _check_types(a, b, *args):
    """Refuse mixed str and bytes before anything is written: formatting a
    bytes filename into a str header would print `b'name'`."""
    if a and not isinstance(a[0], str):
        raise TypeError('lines to compare must be str, not %s (%r)' %
                        (type(a[0]).__name__, a[0]))
    if b and not isinstance(b[0], str):
        raise TypeError('lines to compare must be str, not %s (%r)' %
                        (type(b[0]).__name__, b[0]))
    if isinstance(a, str):
        raise TypeError('input must be a sequence of strings, not %s' %
                        type(a).__name__)
    if isinstance(b, str):
        raise TypeError('input must be a sequence of strings, not %s' %
                        type(b).__name__)
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError('all arguments must be str, not: %r' % (arg,))


def _header_date(date):
    return '\t' + date if date else ''


def unified_diff(a, b, fromfile='', tofile='', fromfiledate='',
                 tofiledate='', n=3, lineterm='\n'):
    """The delta between two sequences of lines as a unified diff."""
    _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
    started = False
    for group in SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        # THE HEADER ONLY WHEN THERE IS A DIFFERENCE: identical inputs give
        # no output at all, not two lines of filenames.
        if not started:
            started = True
            yield '--- ' + fromfile + _header_date(fromfiledate) + lineterm
            yield '+++ ' + tofile + _header_date(tofiledate) + lineterm
        first, last = group[0], group[-1]
        yield ('@@ -' + _format_range_unified(first[1], last[2]) + ' +'
               + _format_range_unified(first[3], last[4]) + ' @@' + lineterm)
        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                for line in a[i1:i2]:
                    yield ' ' + line
                continue
            if tag == 'replace' or tag == 'delete':
                for line in a[i1:i2]:
                    yield '-' + line
            if tag == 'replace' or tag == 'insert':
                for line in b[j1:j2]:
                    yield '+' + line


def context_diff(a, b, fromfile='', tofile='', fromfiledate='',
                 tofiledate='', n=3, lineterm='\n'):
    """The delta between two sequences of lines as a context diff."""
    _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
    prefix = {'insert': '+ ', 'delete': '- ', 'replace': '! ',
              'equal': '  '}
    started = False
    for group in SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            yield '*** ' + fromfile + _header_date(fromfiledate) + lineterm
            yield '--- ' + tofile + _header_date(tofiledate) + lineterm
        first, last = group[0], group[-1]
        yield '***************' + lineterm
        yield ('*** ' + _format_range_context(first[1], last[2]) + ' ****'
               + lineterm)
        # EACH HALF IS LISTED ONLY WHEN IT HAS A CHANGE OF ITS OWN: a hunk
        # that only inserts shows no `a` lines, just the range.
        if any(tag == 'replace' or tag == 'delete'
               for tag, _i1, _i2, _j1, _j2 in group):
            for tag, i1, i2, _j1, _j2 in group:
                if tag != 'insert':
                    for line in a[i1:i2]:
                        yield prefix[tag] + line
        yield ('--- ' + _format_range_context(first[3], last[4]) + ' ----'
               + lineterm)
        if any(tag == 'replace' or tag == 'insert'
               for tag, _i1, _i2, _j1, _j2 in group):
            for tag, _i1, _i2, j1, j2 in group:
                if tag != 'delete':
                    for line in b[j1:j2]:
                        yield prefix[tag] + line


def diff_bytes(dfunc, a, b, fromfile=b'', tofile=b'', fromfiledate=b'',
               tofiledate=b'', n=3, lineterm=b'\n'):
    """`dfunc` -- `unified_diff` or `context_diff` -- over lines of BYTES.

    Decoded as ASCII with `surrogateescape`, which maps every byte to a
    character and back again exactly, so the lines come back as the bytes
    they went in as whatever their encoding was."""
    def decode(s):
        try:
            return s.decode('ascii', 'surrogateescape')
        except AttributeError as err:
            raise TypeError('all arguments must be bytes, not %s (%r)' %
                            (type(s).__name__, s)) from err
    a = [decode(one) for one in a]
    b = [decode(one) for one in b]
    lines = dfunc(a, b, decode(fromfile), decode(tofile),
                  decode(fromfiledate), decode(tofiledate), n,
                  decode(lineterm))
    for line in lines:
        yield line.encode('ascii', 'surrogateescape')


def ndiff(a, b, linejunk=None, charjunk=IS_CHARACTER_JUNK):
    """A `Differ` delta between two lists of lines."""
    return Differ(linejunk, charjunk).compare(a, b)


def restore(delta, which):
    """One of the two sequences an `ndiff` delta was made from: 1 for the
    first, 2 for the second."""
    try:
        tag = {1: '- ', 2: '+ '}[int(which)]
    except KeyError:
        raise ValueError('unknown delta choice (must be 1 or 2): %r'
                         % which) from None
    for line in delta:
        if line[:2] == '  ' or line[:2] == tag:
            yield line[2:]
