"""`shlex`, as ordinary Python this compiler compiles.

COVERAGE: `split` (`comments=`, `posix=`), `quote`, `join`, and the `shlex`
lexer itself -- over a string or any object with `read`/`readline`, in POSIX
and non-POSIX mode, with every attribute a program sets to reshape it
(`commenters`, `wordchars`, `whitespace`, `whitespace_split`, `quotes`,
`escape`, `escapedquotes`, `eof`, `source`, `debug`), `punctuation_chars`,
`get_token`/`push_token`/`read_token`, `push_source`/`pop_source`,
`sourcehook`, `error_leader`, `lineno`, and iteration.

NOT COVERED: a lexer with no stream, which CPython reads from `sys.stdin`:
standard input is not bundled, so `shlex()` alone is refused BY NAME.

## One character at a time, and why that is the specification

The lexer is a state machine over single characters, and what a program
gets from it is decided by exactly when a token is EMITTED -- which is why
the states are CPython's own spellings rather than names of this file's
choosing: ' ' between tokens, 'a' inside a word, 'c' inside a run of
punctuation, a quote character inside that kind of quotes, an escape
character just after one, and None past the end. `lex.state` is an
attribute a program can read, and `debug` prints it.

POSIX MODE differs in four ways, all of them CPython's: quotes are removed
rather than kept, a quote may start in the middle of a word and does not
end it, a backslash escapes -- anywhere outside quotes, and inside double
quotes only before a double quote or a backslash -- and the end of input is
None rather than ''. An empty quoted string is a token in POSIX mode
(`''` gives one empty argument) and nothing in the other.
"""

import os as _os
from collections import deque as _deque
from io import StringIO as _StringIO

__all__ = ["shlex", "split", "quote", "join"]


class shlex:
    "A lexical analyzer class for simple shell-like syntaxes."

    def __init__(self, instream=None, infile=None, posix=False,
                 punctuation_chars=False):
        if isinstance(instream, str):
            instream = _StringIO(instream)
        if instream is None:
            # REFUSED BY NAME rather than failing at the first read: CPython
            # reads `sys.stdin` here, and standard input is not bundled.
            raise TypeError("shlex() with no instream reads sys.stdin, "
                            "which is not supported -- pass a string or a "
                            "stream")
        self.instream = instream
        self.infile = infile
        self.posix = posix
        self.eof = None if posix else ''
        self.commenters = '#'
        self.wordchars = ('abcdfeghijklmnopqrstuvwxyz'
                          'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        # POSIX MODE ALSO COUNTS LATIN-1 LETTERS as word characters -- the
        # accented ones, the same sixty-odd CPython lists, and no others.
        if self.posix:
            self.wordchars += ('ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
                               'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ')
        self.whitespace = ' \t\r\n'
        self.whitespace_split = False
        self.quotes = '\'"'
        self.escape = '\\'
        self.escapedquotes = '"'
        self.state = ' '
        self.pushback = _deque()
        self.lineno = 1
        self.debug = 0
        self.token = ''
        self.filestack = _deque()
        self.source = None
        if not punctuation_chars:
            punctuation_chars = ''
        elif punctuation_chars is True:
            punctuation_chars = '();<>|&'
        self._punctuation_chars = punctuation_chars
        if punctuation_chars:
            # A LOOKAHEAD OF ITS OWN, for the one character that ends a run
            # of punctuation and belongs to whatever comes next.
            self._pushback_chars = _deque()
            # AND THE CHARACTERS A FILE NAME OR A GLOB IS MADE OF become word
            # characters -- less any that were named as punctuation.
            kept = []
            for c in self.wordchars + '~-./*?=':
                if c not in punctuation_chars:
                    kept.append(c)
            self.wordchars = ''.join(kept)

    @property
    def punctuation_chars(self):
        return self._punctuation_chars

    def push_token(self, tok):
        "Push a token onto the stack popped by the get_token method"
        if self.debug >= 1:
            print("shlex: pushing token " + repr(tok))
        self.pushback.appendleft(tok)

    def push_source(self, newstream, newfile=None):
        "Push an input source onto the lexer's input source stack."
        if isinstance(newstream, str):
            newstream = _StringIO(newstream)
        self.filestack.appendleft((self.infile, self.instream, self.lineno))
        self.infile = newfile
        self.instream = newstream
        self.lineno = 1
        if self.debug:
            if newfile is not None:
                print('shlex: pushing to file %s' % (self.infile,))
            else:
                print('shlex: pushing to stream %s' % (self.instream,))

    def pop_source(self):
        "Pop the input source stack."
        self.instream.close()
        (self.infile, self.instream, self.lineno) = self.filestack.popleft()
        if self.debug:
            print('shlex: popping to %s, line %d' % (self.instream,
                                                     self.lineno))
        self.state = ' '

    def get_token(self):
        "Get a token from the input stream (or from stack if it's nonempty)"
        if self.pushback:
            tok = self.pushback.popleft()
            if self.debug >= 1:
                print("shlex: popping token " + repr(tok))
            return tok
        raw = self.read_token()
        # `source`: A TOKEN THAT NAMES A FILE TO READ NEXT, the way `source`
        # does in a shell. What follows it is handed to `sourcehook`, and a
        # file ending returns to the one that included it.
        if self.source is not None:
            while raw == self.source:
                spec = self.sourcehook(self.read_token())
                if spec:
                    (newfile, newstream) = spec
                    self.push_source(newstream, newfile)
                raw = self.get_token()
        while raw == self.eof:
            if not self.filestack:
                return self.eof
            self.pop_source()
            raw = self.get_token()
        if self.debug >= 1:
            if raw != self.eof:
                print("shlex: token=" + repr(raw))
            else:
                print("shlex: token=EOF")
        return raw

    def read_token(self):
        """The next token, read a character at a time.

        `quoted` IS WHETHER ANY QUOTES WERE SEEN in this token, which is what
        lets POSIX mode emit an empty one: `''` is an argument, and an empty
        token that was never quoted is the end of input. `escapedstate` is
        the state an escape returns to once its character has been taken.
        """
        quoted = False
        escapedstate = ' '
        while True:
            if self.punctuation_chars and self._pushback_chars:
                nextchar = self._pushback_chars.pop()
            else:
                nextchar = self.instream.read(1)
            if nextchar == '\n':
                self.lineno += 1
            if self.debug >= 3:
                print("shlex: in state %r I see character: %r"
                      % (self.state, nextchar))
            state = self.state
            if state is None:
                # PAST THE END: every later call answers the end again.
                self.token = ''
                break
            if state == ' ':
                if not nextchar:
                    self.state = None
                    break
                if nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in whitespace state")
                    if self.token or (self.posix and quoted):
                        break
                    continue
                if nextchar in self.commenters:
                    # A COMMENT RUNS TO THE END OF THE LINE, and the line it
                    # ends is counted here because `readline` swallowed the
                    # newline the counter above would have seen.
                    self.instream.readline()
                    self.lineno += 1
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif nextchar in self.wordchars:
                    self.token = nextchar
                    self.state = 'a'
                elif nextchar in self.punctuation_chars:
                    self.token = nextchar
                    self.state = 'c'
                elif nextchar in self.quotes:
                    if not self.posix:
                        self.token = nextchar
                    self.state = nextchar
                elif self.whitespace_split:
                    self.token = nextchar
                    self.state = 'a'
                else:
                    # ANY OTHER CHARACTER IS A TOKEN BY ITSELF, which is how
                    # `a=b` lexes as three tokens without `whitespace_split`.
                    self.token = nextchar
                    if self.token or (self.posix and quoted):
                        break
                    continue
            elif state in self.quotes:
                quoted = True
                if not nextchar:
                    if self.debug >= 2:
                        print("shlex: I see EOF in quotes state")
                    raise ValueError("No closing quotation")
                if nextchar == state:
                    # THE CLOSING QUOTE: the token ends here outside POSIX
                    # mode, and in it the word simply goes on.
                    if not self.posix:
                        self.token += nextchar
                        self.state = ' '
                        break
                    self.state = 'a'
                elif (self.posix and nextchar in self.escape
                      and state in self.escapedquotes):
                    escapedstate = state
                    self.state = nextchar
                else:
                    self.token += nextchar
            elif state in self.escape:
                if not nextchar:
                    if self.debug >= 2:
                        print("shlex: I see EOF in escape state")
                    raise ValueError("No escaped character")
                # INSIDE QUOTES ONLY THE QUOTE AND THE ESCAPE ITSELF CAN BE
                # ESCAPED: before anything else the backslash is kept, as a
                # POSIX shell keeps it.
                if (escapedstate in self.quotes and nextchar != state
                        and nextchar != escapedstate):
                    self.token += state
                self.token += nextchar
                self.state = escapedstate
            elif state == 'a' or state == 'c':
                if not nextchar:
                    self.state = None
                    break
                if nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break
                    continue
                if nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                    if self.posix:
                        self.state = ' '
                        if self.token or (self.posix and quoted):
                            break
                        continue
                elif state == 'c':
                    # A RUN OF PUNCTUATION is one token -- `&&`, `>>` -- and
                    # the character that ends it is kept for the next one.
                    if nextchar in self.punctuation_chars:
                        self.token += nextchar
                    else:
                        if nextchar not in self.whitespace:
                            self._pushback_chars.append(nextchar)
                        self.state = ' '
                        break
                elif self.posix and nextchar in self.quotes:
                    self.state = nextchar
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif (nextchar in self.wordchars or nextchar in self.quotes
                      or (self.whitespace_split
                          and nextchar not in self.punctuation_chars)):
                    self.token += nextchar
                else:
                    # A CHARACTER THAT CANNOT CONTINUE THE WORD ends it and is
                    # read again as the start of the next token.
                    if self.punctuation_chars:
                        self._pushback_chars.append(nextchar)
                    else:
                        self.pushback.appendleft(nextchar)
                    if self.debug >= 2:
                        print("shlex: I see punctuation in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break
                    continue
        result = self.token
        self.token = ''
        if self.posix and not quoted and result == '':
            result = None
        if self.debug > 1:
            if result:
                print("shlex: raw token=" + repr(result))
            else:
                print("shlex: raw token=EOF")
        return result

    def sourcehook(self, newfile):
        "Hook called on a filename to be sourced."
        if newfile[0] == '"':
            newfile = newfile[1:-1]
        # RELATIVE TO THE FILE THAT NAMED IT, as `#include` is.
        if isinstance(self.infile, str) and not _os.path.isabs(newfile):
            newfile = _os.path.join(_os.path.dirname(self.infile), newfile)
        return (newfile, open(newfile, "r"))

    def error_leader(self, infile=None, lineno=None):
        "Emit a C-compiler-like, Emacs-friendly error-message leader."
        if infile is None:
            infile = self.infile
        if lineno is None:
            lineno = self.lineno
        return "\"%s\", line %d: " % (infile, lineno)

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token == self.eof:
            raise StopIteration
        return token


def split(s, comments=False, posix=True):
    """Split the string *s* using shell-like syntax."""
    if s is None:
        raise ValueError("s argument must not be None")
    lex = shlex(s, posix=posix)
    lex.whitespace_split = True
    if not comments:
        lex.commenters = ''
    return list(lex)


def join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return ' '.join([quote(arg) for arg in split_command])


#: What a shell never needs quoted. ASCII only: a non-ASCII string is always
#: quoted, whatever its characters are.
_SAFE = frozenset('%+,-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
                  'abcdefghijklmnopqrstuvwxyz')


def quote(s):
    """Return a shell-escaped version of the string *s*.

    AN EMPTY VALUE IS `''` BEFORE ITS TYPE IS ASKED -- CPython's order, so
    `quote(None)` and `quote(0)` answer `''` rather than refusing -- and a
    single quote inside is closed, double-quoted and reopened: `'$'"'"'b'`.
    """
    if not s:
        return "''"
    if not isinstance(s, str):
        raise TypeError("expected string object, got "
                        + repr(type(s).__name__))
    if s.isascii():
        for c in s:
            if c not in _SAFE:
                break
        else:
            return s
    return "'" + s.replace("'", "'\"'\"'") + "'"
