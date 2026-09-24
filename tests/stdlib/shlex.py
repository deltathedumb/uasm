# COVERAGE: split -- words, both quotes, a quote opened mid-word, the
# backslash outside quotes, inside double quotes (before a quote, a
# backslash, and anything else) and inside single quotes, an empty quoted
# argument, comments=True/False, posix=False, and both ValueErrors; quote --
# safe strings, spaces, a single quote inside, non-ASCII, the empty value
# before the type check, the TypeError; join; the shlex lexer in POSIX and
# non-POSIX mode -- whitespace_split off, punctuation_chars (True and a
# string, runs of punctuation, the wordchars it adds and removes),
# commenters, wordchars, quotes and escape reassigned, lineno across lines
# and comments, get_token/push_token/read_token, push_source/pop_source, the
# eof value, iteration, error_leader, the state attribute, and debug output.
#
# Run under CPython and under uasm; the outputs must be identical. So
# the assertions below are written against what the module IS SPECIFIED to
# do, not against what uasm currently does.
import io
import shlex


def attempt(label, f):
    try:
        print(label, repr(f()))
    except (TypeError, ValueError) as e:
        print(label, type(e).__name__ + ":", e)


# ---- split -------------------------------------------------------------------
for text in ("a b c", "  spaced   out  ", "'single quoted' \"double quoted\"",
             "mid'dle quo'ted", "back\\ slash", "a\\\\b", "\"dq \\\" and \\\\\"",
             "\"dq \\n kept\"", "'sq \\ stays'", "''", "x '' y", "\"\"",
             "a#b # comment", "tab\there\nnewline", "é ß", "a=b c=d",
             "--flag=\"value with spaces\"", "cp *.txt ~/dir/", "a;b|c&&d",
             "'it'\\''s'", "", "   "):
    attempt(repr(text), lambda: shlex.split(text))
attempt("comments", lambda: shlex.split("a b # the rest", comments=True))
attempt("comment only", lambda: shlex.split("# nothing", comments=True))
attempt("non-posix", lambda: shlex.split("'a b' \"c\" d\\e", posix=False))
attempt("non-posix empty", lambda: shlex.split("x '' y", posix=False))
attempt("unclosed", lambda: shlex.split("a 'b c"))
attempt("unclosed dq", lambda: shlex.split('a "b c'))
attempt("trailing escape", lambda: shlex.split("a b\\"))
attempt("None", lambda: shlex.split(None))

# ---- quote and join ----------------------------------------------------------
for value in ("", "safe-name_1.txt", "a b", "it's", "$HOME", "café", "a\nb",
              "%+,-./:=@", "*", "'"):
    attempt("quote " + repr(value), lambda: shlex.quote(value))
attempt("quote None", lambda: shlex.quote(None))
attempt("quote 0", lambda: shlex.quote(0))
attempt("quote 5", lambda: shlex.quote(5))
attempt("quote bytes", lambda: shlex.quote(b"x"))
print(shlex.join(["ls", "-l", "my file", "it's"]))
print(shlex.split(shlex.join(["a b", "c'd", "", "e"])))

# ---- the lexer ---------------------------------------------------------------
lex = shlex.shlex("a=b 'c d' e_f(g) # comment\nnext")
print(list(lex), lex.lineno)
lex = shlex.shlex("a=b 'c d' e_f(g)", posix=True)
print(list(lex))
lex = shlex.shlex("a && b || c; (d) > e >> f", punctuation_chars=True)
print(list(lex), repr(lex.punctuation_chars))
lex = shlex.shlex("ls -l ~/x.txt|wc -l", posix=True, punctuation_chars="|")
print(list(lex), repr(lex.punctuation_chars))
lex = shlex.shlex("a:b:c d", posix=True)
lex.whitespace_split = True
lex.whitespace = ":"
print(list(lex))
lex = shlex.shlex("path/to-file.txt next", posix=True)
lex.wordchars += "/-."
print(list(lex))
lex = shlex.shlex("x ;; comment\ny", posix=True)
lex.commenters = ";"
print(list(lex))
lex = shlex.shlex("|a b| c 'd e'", posix=True)
lex.quotes = "|"
print(list(lex))
lex = shlex.shlex("say %\"hi%\" to you", posix=True)
lex.escape = "%"
lex.whitespace_split = True
print(list(lex))

lex = shlex.shlex(io.StringIO("one two\nthree\n\nfour"), infile="f.sh",
                  posix=True)
print(lex.get_token(), lex.lineno, lex.state)
lex.push_token("pushed")
print(lex.get_token(), lex.get_token(), lex.lineno)
print(repr(lex.error_leader()), repr(lex.error_leader("other", 9)))
print(lex.read_token(), lex.get_token(), repr(lex.get_token()), lex.state)
print(repr(lex.get_token()), repr(lex.eof))

lex = shlex.shlex("outer1 outer2")
lex.push_source("inner1 inner2", "inner.txt")
print(lex.infile, lex.get_token(), lex.get_token(), lex.get_token(),
      lex.infile, lex.get_token(), repr(lex.get_token()))
lex = shlex.shlex("x")
print(repr(lex.eof), repr(lex.get_token()), repr(lex.get_token()))

for level in (1, 2, 3):
    lex = shlex.shlex("a 'b'", posix=True)
    lex.debug = level
    print("debug", level, list(lex))
print("done")
