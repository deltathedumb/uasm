"""`argparse`, as ordinary Python this compiler compiles.

COVERAGE: `ArgumentParser` with every constructor parameter -- `prog`,
`usage`, `description`, `epilog`, `parents`, `formatter_class`,
`prefix_chars`, `fromfile_prefix_chars`, `argument_default`,
`conflict_handler` (`error` and `resolve`), `add_help`, `allow_abbrev`,
`exit_on_error`, `color`; `add_argument` with every keyword -- `action`,
`nargs` (None, an int, `?`, `*`, `+`, `REMAINDER`, `PARSER`, `SUPPRESS`),
`const`, `default`, `type`, `choices`, `required`, `help` (with its
`%(...)s` expansion), `metavar` (tuples too), `dest`, `deprecated`; every
action -- `store`, `store_const`, `store_true`, `store_false`, `append`,
`append_const`, `extend`, `count`, `help`, `version`, `BooleanOptionalAction`
and a program's own `Action` subclass; argument groups, mutually exclusive
groups (required or not), subparsers (`add_subparsers` with `title`,
`description`, `dest`, `required`, `help`, `metavar`, `prog`, and
`add_parser` with `aliases`, `help`, `deprecated`, `parents`);
`set_defaults`/`get_default`; `parse_args`, `parse_known_args`,
`parse_intermixed_args`, `parse_known_intermixed_args`; unambiguous
abbreviations, `-xyz` clusters, `--opt=value`, `-ovalue`, negative numbers,
`--`; `fromfile_prefix_chars` with `convert_arg_line_to_args`; `error`,
`exit`, `ArgumentError`, `ArgumentTypeError`, `exit_on_error=False`;
`format_usage`, `format_help`, `print_usage`, `print_help` and the five
formatter classes, laid out exactly as CPython lays them out, including
the usage wrap; `FileType`; `Namespace` (`repr`, `==`, `in`); and colour,
decided by the same `PYTHON_COLORS`/`NO_COLOR`/`FORCE_COLOR`/`TERM` rules.

`suggest_on_error=True` suggests from `difflib.get_close_matches`, as
CPython's does, for a mistyped choice and a mistyped subcommand alike.

NOT COVERED: TRANSLATION. Messages are never translated -- CPython asks
`gettext` for the `messages` domain, which no CPython install ships a
catalog for, so its answer is the English text too, and `_` and
`ngettext` below are that answer rather than an approximation of it. An
interactive terminal's WIDTH: help is laid out for `$COLUMNS`
columns, or 80, where CPython asks the terminal itself -- the two agree
whenever output goes to a pipe or a file, and `$COLUMNS` is read the same
way. A program run as `python -m pkg` or from a zip is named after its
`argv[0]` like any script, because a compiled program is always one.
`fromfile_prefix_chars` reads UTF-8 with strict errors rather than the
filesystem encoding's own error handler. `FileType` handed `-` for reading,
or in a binary mode: the first is `sys.stdin` and the second a stream's
`.buffer`, and neither is bundled. Colour needs the environment to ask for
it: there is no terminal query, so a stream is never a TTY here.

## Why the positionals are not matched with `re`

CPython decides how many strings each positional gets by writing every
positional's `nargs` as a small regular expression over a string of `A`s
(arguments), `O`s (options) and `-` (the first `--`), concatenating them,
and taking the first match a BACKTRACKING engine with greedy quantifiers
finds. That rule is the specification, and it is not a simple one: `nargs=
'*'` followed by `nargs=1` gives the last string to the second, because the
engine backs the star off one character at a time until what follows fits.

The patterns only ever contain one shape, though -- a character class under
a greedy quantifier, in sequence, grouped per positional -- so `_match_runs`
below is exactly that engine for exactly that shape: each run takes as many
characters as it may, and on failure gives one back and lets the rest try
again. The first success in that order is the one the regex engine reports,
because it is the same search. Doing it here keeps `re` out of every program
that parses a command line; `bundled/re.py` is the largest module in the
library, and a program that imports `argparse` did not ask for it.

## Why the hook methods keep CPython's names

`HelpFormatter._split_lines`, `_fill_text`, `_get_help_string`,
`_format_action_invocation`, `_get_default_metavar_for_optional` and their
siblings are private by name and public in practice: `RawTextHelpFormatter`
IS an override of `_split_lines`, and programs subclass the formatter the
same way. `ArgumentParser.error`, `exit`, `_get_formatter`, `_get_values`,
`_get_value`, `_check_value`, `_parse_optional` and
`convert_arg_line_to_args` are overridden in the wild too. So those names and
signatures are CPython's, and so are the attributes programs read off a
parser -- `_actions`, `_option_string_actions`, `_action_groups`,
`_mutually_exclusive_groups`, `_group_actions`, `_positionals`,
`_optionals`, `_subparsers`, `_defaults`. What runs behind them is this
file's own.
"""

import copy as _copy
import os as _os
import sys as _sys
import textwrap as _textwrap
import warnings as _warnings

__version__ = '1.1'
__all__ = [
    'ArgumentParser',
    'ArgumentError',
    'ArgumentTypeError',
    'BooleanOptionalAction',
    'FileType',
    'HelpFormatter',
    'ArgumentDefaultsHelpFormatter',
    'RawDescriptionHelpFormatter',
    'RawTextHelpFormatter',
    'MetavarTypeHelpFormatter',
    'Namespace',
    'Action',
    'ONE_OR_MORE',
    'OPTIONAL',
    'PARSER',
    'REMAINDER',
    'SUPPRESS',
    'ZERO_OR_MORE',
]

SUPPRESS = '==SUPPRESS=='

OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'
_UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'


# ── messages ────────────────────────────────────────────────────────────────

def _(message):
    """What `gettext.gettext` answers with no catalog: the message itself.
    See the module docstring for why that is CPython's answer as well."""
    return message


def ngettext(singular, plural, n):
    """`gettext.ngettext` with no catalog. `n == 1` AND NOTHING LOOSER, so
    `nargs='A...'` -- which reaches one of these with a string for `n` --
    takes the plural, as it does in CPython."""
    if n == 1:
        return singular
    return plural


# ── text helpers ────────────────────────────────────────────────────────────

#: The six characters `\s` matches under `re.ASCII`, which is what CPython's
#: formatter collapses. NOT `str.split()`'s set: that one is Unicode's, and a
#: no-break space in a help string survives CPython's formatter as itself.
_ASCII_SPACE = ' \t\n\r\x0b\x0c'


def _squash_space(text):
    """Every run of ASCII whitespace as one space."""
    out = []
    in_space = False
    for c in text:
        if c in _ASCII_SPACE:
            if not in_space:
                out.append(' ')
            in_space = True
        else:
            out.append(c)
            in_space = False
    return ''.join(out)


def _collapse_breaks(text):
    """Three or more newlines in a row as two: a section that ends in a
    blank line followed by one that starts with a blank line is ONE blank
    line in the help, not two."""
    out = []
    run = 0
    for c in text:
        if c == '\n':
            run += 1
            if run <= 2:
                out.append(c)
        else:
            run = 0
            out.append(c)
    return ''.join(out)


def _looks_negative(text):
    """Does `text` START like a negative number -- a dash, an optional dot,
    then a digit? That is the whole test, so `-1x` looks like one too.

    `isdecimal` IS WHAT `\\d` MEANS for a `str` pattern: any character of
    Unicode category Nd, not only the ten ASCII ones."""
    if len(text) < 2 or text[0] != '-':
        return False
    if text[1].isdecimal():
        return True
    return text[1] == '.' and len(text) > 2 and text[2].isdecimal()


def _terminal_columns():
    """`shutil.get_terminal_size().columns` for output that is not a
    terminal: `$COLUMNS` when it is a positive integer, 80 otherwise.

    `int()` IS THE PARSER, so ` 120 ` counts and `wide` does not, exactly as
    in `shutil`; and zero or less is not a width, it is a request to ask the
    terminal -- which, with nothing to ask, is the fallback."""
    try:
        columns = int(_os.environ['COLUMNS'])
    except (KeyError, ValueError):
        columns = 0
    if columns <= 0:
        columns = 80
    return columns


# ── colour ──────────────────────────────────────────────────────────────────

class _Theme:
    """The escape sequences the help is painted with, or twelve empty strings.

    CPython's are `_colorize`'s `Argparse` section. EMPTY STRINGS AND NOT A
    FLAG, because that is how the formatter is written on both sides: every
    piece of output is `code + text + reset`, and a theme with no colour is
    one where both halves are nothing."""

    def __init__(self, coloured):
        self.usage = '\x1b[1;34m' if coloured else ''
        self.prog = '\x1b[1;35m' if coloured else ''
        self.prog_extra = '\x1b[35m' if coloured else ''
        self.heading = '\x1b[1;34m' if coloured else ''
        self.summary_long_option = '\x1b[36m' if coloured else ''
        self.summary_short_option = '\x1b[32m' if coloured else ''
        self.summary_label = '\x1b[33m' if coloured else ''
        self.summary_action = '\x1b[32m' if coloured else ''
        self.long_option = '\x1b[1;36m' if coloured else ''
        self.short_option = '\x1b[1;32m' if coloured else ''
        self.label = '\x1b[1;33m' if coloured else ''
        self.action = '\x1b[1;32m' if coloured else ''
        self.reset = '\x1b[0m' if coloured else ''


_PLAIN = _Theme(False)
_COLOURED = _Theme(True)

#: EVERY CODE `_colorize.ANSIColors` DEFINES, not only the twelve above:
#: `decolor` strips all of them, so a help string carrying its own
#: `\x1b[31m` is measured without it too.
_ANSI_CODES = (
    '\x1b[0m', '\x1b[1m', '\x1b[30m', '\x1b[31m', '\x1b[32m', '\x1b[33m',
    '\x1b[34m', '\x1b[35m', '\x1b[36m', '\x1b[37m', '\x1b[40m', '\x1b[41m',
    '\x1b[42m', '\x1b[43m', '\x1b[44m', '\x1b[45m', '\x1b[46m', '\x1b[47m',
    '\x1b[90m', '\x1b[91m', '\x1b[92m', '\x1b[93m', '\x1b[94m', '\x1b[95m',
    '\x1b[96m', '\x1b[97m', '\x1b[100m', '\x1b[101m', '\x1b[102m',
    '\x1b[103m', '\x1b[104m', '\x1b[105m', '\x1b[106m', '\x1b[107m',
    '\x1b[1;30m', '\x1b[1;31m', '\x1b[1;32m', '\x1b[1;33m', '\x1b[1;34m',
    '\x1b[1;35m', '\x1b[1;36m', '\x1b[1;37m',
)


def _decolor(text):
    for code in _ANSI_CODES:
        text = text.replace(code, '')
    return text


def _can_colour():
    """`_colorize.can_colorize()`, in CPython's order: `PYTHON_COLORS` says
    yes or no outright, then `NO_COLOR` refuses, then `FORCE_COLOR` insists,
    then `TERM=dumb` refuses, and only then is the stream asked whether it is
    a terminal -- which, here, it never is."""
    environ = _os.environ
    wanted = environ.get('PYTHON_COLORS')
    if wanted == '0':
        return False
    if wanted == '1':
        return True
    if environ.get('NO_COLOR'):
        return False
    if environ.get('FORCE_COLOR'):
        return True
    if environ.get('TERM') == 'dumb':
        return False
    out = _sys.stdout
    if not hasattr(out, 'fileno'):
        return False
    isatty = getattr(out, 'isatty', None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


# ── small pieces ────────────────────────────────────────────────────────────

class _AttributeHolder:
    """`Name(attr=value, ...)` as a repr, from `_get_kwargs`.

    A NAME THAT IS NOT AN IDENTIFIER is gathered into a trailing
    `**{...}`, so `Namespace(**{'a-b': 1})` reads back as something a
    program could have written."""

    def __repr__(self):
        shown = []
        for arg in self._get_args():
            shown.append(repr(arg))
        odd = {}
        for name, value in self._get_kwargs():
            if name.isidentifier():
                shown.append(name + '=' + repr(value))
            else:
                odd[name] = value
        if odd:
            shown.append('**' + repr(odd))
        return type(self).__name__ + '(' + ', '.join(shown) + ')'

    def _get_kwargs(self):
        return list(self.__dict__.items())

    def _get_args(self):
        return []


def _copy_items(items):
    """The list an `append` adds to, as a NEW list: the default a parser was
    given must not grow every time a command line appends to it."""
    if items is None:
        return []
    if type(items) is list:
        return items[:]
    return _copy.copy(items)


def _identity(value):
    return value


# ── formatting help ─────────────────────────────────────────────────────────

class HelpFormatter:
    """Formatter for generating usage messages and argument help strings.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def __init__(self, prog, indent_increment=2, max_help_position=24,
                 width=None, color=True):
        if width is None:
            width = _terminal_columns() - 2
        self._set_color(color)
        self._prog = prog
        self._indent_increment = indent_increment
        # THE HELP COLUMN NEVER STARTS PAST `width - 20`, so a narrow
        # terminal still leaves the help text twenty columns -- and never
        # before two indents, so a very narrow one still has somewhere to put
        # the invocation.
        self._max_help_position = min(max_help_position,
                                      max(width - 20, indent_increment * 2))
        self._width = width
        self._current_indent = 0
        self._level = 0
        self._action_max_length = 0
        self._root_section = self._Section(self, None)
        self._current_section = self._root_section

    def _set_color(self, color):
        if color and _can_colour():
            self._theme = _COLOURED
            self._decolor = _decolor
        else:
            self._theme = _PLAIN
            self._decolor = _identity

    # -- sections and indentation -------------------------------------------

    def _indent(self):
        self._current_indent += self._indent_increment
        self._level += 1

    def _dedent(self):
        self._current_indent -= self._indent_increment
        assert self._current_indent >= 0, 'Indent decreased below 0.'
        self._level -= 1

    class _Section:
        """A heading and the items under it, rendered only when asked.

        RENDERED LATE, and that is not laziness: where the help column sits
        depends on the widest invocation in the WHOLE parser, which is not
        known until every argument has been added. So an item is a function
        and its arguments, called when the text is finally wanted."""

        def __init__(self, formatter, parent, heading=None):
            self.formatter = formatter
            self.parent = parent
            self.heading = heading
            self.items = []

        def format_help(self):
            formatter = self.formatter
            if self.parent is not None:
                formatter._indent()
            body = formatter._join_parts([func(*args)
                                          for func, args in self.items])
            if self.parent is not None:
                formatter._dedent()
            # AN EMPTY SECTION HAS NO HEADING EITHER: a group whose every
            # argument is suppressed leaves no trace in the help.
            if not body:
                return ''
            heading = ''
            if self.heading is not SUPPRESS and self.heading is not None:
                t = formatter._theme
                heading = (' ' * formatter._current_indent + t.heading
                           + _('%(heading)s:') % dict(heading=self.heading)
                           + t.reset + '\n')
            return formatter._join_parts(['\n', heading, body, '\n'])

    def _add_item(self, func, args):
        self._current_section.items.append((func, args))

    # -- building a message --------------------------------------------------

    def start_section(self, heading):
        self._indent()
        section = self._Section(self, self._current_section, heading)
        self._add_item(section.format_help, [])
        self._current_section = section

    def end_section(self):
        self._current_section = self._current_section.parent
        self._dedent()

    def add_text(self, text):
        if text is not SUPPRESS and text is not None:
            self._add_item(self._format_text, [text])

    def add_usage(self, usage, actions, groups, prefix=None):
        if usage is not SUPPRESS:
            self._add_item(self._format_usage, [usage, actions, groups,
                                                prefix])

    def add_argument(self, action):
        if action.help is SUPPRESS:
            return
        # THE WIDEST INVOCATION DECIDES THE HELP COLUMN, measured at the
        # indent it will be printed at and without its colour codes -- a
        # subcommand's line is indented one step further than its parent's,
        # which is why the subactions are measured inside the generator that
        # indents them.
        longest = (len(self._decolor(self._format_action_invocation(action)))
                   + self._current_indent)
        for subaction in self._iter_indented_subactions(action):
            size = (len(self._decolor(
                self._format_action_invocation(subaction)))
                + self._current_indent)
            if size > longest:
                longest = size
        if longest > self._action_max_length:
            self._action_max_length = longest
        self._add_item(self._format_action, [action])

    def add_arguments(self, actions):
        for action in actions:
            self.add_argument(action)

    # -- producing it --------------------------------------------------------

    def format_help(self):
        text = self._root_section.format_help()
        if text:
            text = _collapse_breaks(text)
            text = text.strip('\n') + '\n'
        return text

    def _join_parts(self, part_strings):
        return ''.join([part for part in part_strings
                        if part and part is not SUPPRESS])

    def _format_usage(self, usage, actions, groups, prefix):
        t = self._theme
        if prefix is None:
            prefix = _('usage: ')

        if usage is not None:
            # A WRITTEN USAGE IS USED AS WRITTEN, `%(prog)s` aside -- no
            # wrapping, and no second look at the arguments.
            usage = (t.prog_extra
                     + usage % {'prog': t.prog + self._prog + t.reset
                                + t.prog_extra}
                     + t.reset)
        elif not actions:
            usage = t.prog + self._prog + t.reset
        else:
            prog = self._prog
            parts, pos_start = self._get_actions_usage_parts(actions, groups)
            usage = ' '.join([one for one in [prog] + parts if one])
            text_width = self._width - self._current_indent
            if len(prefix) + len(self._decolor(usage)) > text_width:
                usage = '\n'.join(self._wrap_usage(prog, prefix,
                                                   parts[:pos_start],
                                                   parts[pos_start:],
                                                   text_width))
            # THE PROG IS PAINTED LAST, once the lines are decided: it is
            # measured without its colour everywhere above.
            usage = t.prog + prog + t.reset + usage.removeprefix(prog)
        return t.usage + prefix + t.reset + usage + '\n\n'

    def _wrap_usage(self, prog, prefix, opt_parts, pos_parts, text_width):
        """A usage too long for one line, as lines.

        TWO LAYOUTS, chosen by how long the program's name is. A short one
        is followed by the options on its own line and every continuation is
        indented to line up after it; the positionals start a fresh line of
        their own. A name taking more than three quarters of the width goes
        on a line by itself instead, and the rest is indented only as far as
        `usage: ` -- and still split into options and positionals whenever it
        does not fit on one line."""
        if len(prefix) + len(self._decolor(prog)) <= 0.75 * text_width:
            indent = ' ' * (len(prefix) + len(self._decolor(prog)) + 1)
            if opt_parts:
                lines = self._usage_lines([prog] + opt_parts, indent,
                                          text_width, prefix)
                lines.extend(self._usage_lines(pos_parts, indent,
                                               text_width))
            elif pos_parts:
                lines = self._usage_lines([prog] + pos_parts, indent,
                                          text_width, prefix)
            else:
                lines = [prog]
            return lines
        indent = ' ' * len(prefix)
        lines = self._usage_lines(opt_parts + pos_parts, indent, text_width)
        if len(lines) > 1:
            lines = (self._usage_lines(opt_parts, indent, text_width)
                     + self._usage_lines(pos_parts, indent, text_width))
        return [prog] + lines

    def _usage_lines(self, parts, indent, text_width, prefix=None):
        """`parts` filled into lines no wider than `text_width`.

        WITH A PREFIX, the first line is measured as if `usage: ` already
        sat in front of it, and comes back WITHOUT the indent the others
        carry -- the prefix is what goes there. A part longer than the width
        is never split: it gets a line of its own and overhangs."""
        lines = []
        line = []
        if prefix is not None:
            used = len(prefix) - 1
        else:
            used = len(indent) - 1
        for part in parts:
            size = len(self._decolor(part))
            if used + 1 + size > text_width and line:
                lines.append(indent + ' '.join(line))
                line = []
                used = len(indent) - 1
            line.append(part)
            used += size + 1
        if line:
            lines.append(indent + ' '.join(line))
        if prefix is not None:
            lines[0] = lines[0][len(indent):]
        return lines

    def _is_long_option(self, string):
        return len(string) > 2

    def _get_actions_usage_parts(self, actions, groups):
        """The pieces of the usage line, and where the positionals start.

        A MUTUALLY EXCLUSIVE GROUP IS ONE ENTRY, printed `[a | b]` or
        `(a | b)` when it is required, wherever its first member would have
        stood -- and a group holding a positional stands where the
        POSITIONAL does, since a positional's place on the line is its place
        on the command line. The split point is what lets a wrapped usage
        start the positionals on a line of their own without breaking such a
        group in half."""
        t = self._theme
        visible = [one for one in actions if one.help is not SUPPRESS]
        owner = dict.fromkeys(visible)
        for group in groups:
            for one in group._group_actions:
                if one in owner:
                    owner[one] = group

        def claim(group):
            # THE GROUP'S OTHER OPTIONALS, each claimed exactly once: the
            # first member reached takes the rest with it.
            claimed = []
            for other in group._group_actions:
                if other.option_strings and owner.pop(other, None):
                    claimed.append(other)
            return claimed

        positionals = []
        for one in visible:
            if not one.option_strings:
                group = owner.pop(one)
                if group:
                    positionals.append((group.required, claim(group) + [one]))
                else:
                    positionals.append((None, [one]))
        optionals = []
        for one in visible:
            if one.option_strings and one in owner:
                group = owner.pop(one)
                if group:
                    optionals.append((group.required, [one] + claim(group)))
                else:
                    optionals.append((None, [one]))

        parts = []
        pos_start = None
        entries = optionals + positionals
        for index in range(len(entries)):
            required, members = entries[index]
            start = len(parts)
            if index == len(optionals):
                pos_start = start
            in_group = len(members) > 1
            for one in members:
                if not one.option_strings:
                    part = self._format_args(
                        one, self._get_default_metavar_for_positional(one))
                    # INSIDE A GROUP the group's own brackets say it is
                    # optional, so the positional's `[...]` comes off.
                    if in_group and part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]
                    part = t.summary_action + part + t.reset
                else:
                    option_string = one.option_strings[0]
                    if self._is_long_option(option_string):
                        colour = t.summary_long_option
                    else:
                        colour = t.summary_short_option
                    if one.nargs == 0:
                        part = colour + one.format_usage() + t.reset
                    else:
                        args_string = self._format_args(
                            one, self._get_default_metavar_for_optional(one))
                        part = (colour + option_string + ' '
                                + t.summary_label + args_string + t.reset)
                    if not (one.required or required or in_group):
                        part = '[' + part + ']'
                parts.append(part)
            if in_group:
                parts[start] = ('(' if required else '[') + parts[start]
                for i in range(start, len(parts) - 1):
                    parts[i] += ' |'
                parts[-1] += ')' if required else ']'
        if pos_start is None:
            pos_start = len(parts)
        return parts, pos_start

    def _format_text(self, text):
        if '%(prog)' in text:
            text = text % dict(prog=self._prog)
        text_width = max(self._width - self._current_indent, 11)
        indent = ' ' * self._current_indent
        return self._fill_text(text, text_width, indent) + '\n\n'

    def _format_action(self, action):
        """One argument's entry: its invocation, then its help beside it or,
        when the invocation is too wide for the column, below it."""
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        header = self._format_action_invocation(action)
        plain = self._decolor(header)
        indent_first = 0
        if not action.help:
            header = ' ' * self._current_indent + header + '\n'
        elif len(plain) <= action_width:
            # PADDED WITHOUT ITS COLOUR, then the colour put back: the codes
            # take no columns on a terminal and must not take any here.
            padded = ' ' * self._current_indent + plain.ljust(action_width)
            header = padded.replace(plain, header) + '  '
        else:
            header = ' ' * self._current_indent + header + '\n'
            indent_first = help_position
        parts = [header]
        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                lines = self._split_lines(help_text, help_width)
                parts.append(' ' * indent_first + lines[0] + '\n')
                for line in lines[1:]:
                    parts.append(' ' * help_position + line + '\n')
        elif not header.endswith('\n'):
            parts.append('\n')
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))
        return self._join_parts(parts)

    def _format_action_invocation(self, action):
        t = self._theme
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            return (t.action
                    + ' '.join(self._metavar_formatter(action, default)(1))
                    + t.reset)
        painted = []
        for one in action.option_strings:
            if self._is_long_option(one):
                painted.append(t.long_option + one + t.reset)
            else:
                painted.append(t.short_option + one + t.reset)
        # `-s, --long` for a flag; `-s, --long ARGS` -- the arguments ONCE,
        # after the last spelling -- for anything that takes a value.
        if action.nargs == 0:
            return ', '.join(painted)
        default = self._get_default_metavar_for_optional(action)
        return (', '.join(painted) + ' ' + t.label
                + self._format_args(action, default) + t.reset)

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            result = '{' + ','.join(map(str, action.choices)) + '}'
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            return (result,) * tuple_size
        return format

    def _format_args(self, action, default_metavar):
        """The `ARG [ARG ...]` that stands for an action's values.

        WRITTEN WITH `%` ON PURPOSE: a metavar TUPLE of the wrong length for
        its `nargs` must fail here with a `TypeError`, because that is how
        `add_argument` finds out the tuple and the count disagree."""
        get_metavar = self._metavar_formatter(action, default_metavar)
        nargs = action.nargs
        if nargs is None:
            return '%s' % get_metavar(1)
        if nargs == OPTIONAL:
            return '[%s]' % get_metavar(1)
        if nargs == ZERO_OR_MORE:
            metavar = get_metavar(1)
            if len(metavar) == 2:
                return '[%s [%s ...]]' % metavar
            return '[%s ...]' % metavar
        if nargs == ONE_OR_MORE:
            return '%s [%s ...]' % get_metavar(2)
        if nargs == REMAINDER:
            return '...'
        if nargs == PARSER:
            return '%s ...' % get_metavar(1)
        if nargs == SUPPRESS:
            return ''
        try:
            formats = ['%s' for _unused in range(nargs)]
        except TypeError:
            raise ValueError('invalid nargs value') from None
        return ' '.join(formats) % get_metavar(nargs)

    def _expand_help(self, action):
        """`%(default)s` and friends: every attribute of the action, plus
        `prog`. A value with a `__name__` -- a type, a function -- is shown by
        that name, and `choices` as a comma-separated list; a SUPPRESSed
        value is not offered at all, so naming it is a `KeyError`."""
        help_string = self._get_help_string(action)
        if '%' not in help_string:
            return help_string
        params = dict(vars(action), prog=self._prog)
        for name in list(params):
            value = params[name]
            if value is SUPPRESS:
                del params[name]
            elif hasattr(value, '__name__'):
                params[name] = value.__name__
        if params.get('choices') is not None:
            params['choices'] = ', '.join(map(str, params['choices']))
        return help_string % params

    def _iter_indented_subactions(self, action):
        get_subactions = getattr(action, '_get_subactions', None)
        if get_subactions is None:
            return
        self._indent()
        for subaction in get_subactions():
            yield subaction
        self._dedent()

    def _split_lines(self, text, width):
        return _textwrap.wrap(_squash_space(text).strip(), width)

    def _fill_text(self, text, width, indent):
        return _textwrap.fill(_squash_space(text).strip(), width,
                              initial_indent=indent,
                              subsequent_indent=indent)

    def _get_help_string(self, action):
        return action.help

    def _get_default_metavar_for_optional(self, action):
        return action.dest.upper()

    def _get_default_metavar_for_positional(self, action):
        return action.dest


class RawDescriptionHelpFormatter(HelpFormatter):
    """Help message formatter which retains any formatting in descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _fill_text(self, text, width, indent):
        return ''.join([indent + line
                        for line in text.splitlines(keepends=True)])


class RawTextHelpFormatter(RawDescriptionHelpFormatter):
    """Help message formatter which retains formatting of all help text.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _split_lines(self, text, width):
        return text.splitlines()


class ArgumentDefaultsHelpFormatter(HelpFormatter):
    """Help message formatter which adds default values to argument help.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_help_string(self, action):
        # ONLY WHERE A DEFAULT CAN HAPPEN: an optional, or a positional that
        # may be left out. A required argument never shows its default,
        # because it never gets it.
        help = action.help
        if help is None:
            help = ''
        if ('%(default)' not in help and action.default is not SUPPRESS
                and not action.required):
            if action.option_strings or action.nargs in (OPTIONAL,
                                                         ZERO_OR_MORE):
                help += _(' (default: %(default)s)')
        return help


class MetavarTypeHelpFormatter(HelpFormatter):
    """Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__


# ── errors ──────────────────────────────────────────────────────────────────

def _get_action_name(argument):
    """How an error message names an argument: its option strings joined by
    `/`, else its metavar, else its dest, else its choices."""
    if argument is None:
        return None
    if argument.option_strings:
        return '/'.join(argument.option_strings)
    if argument.metavar not in (None, SUPPRESS):
        metavar = argument.metavar
        if not isinstance(metavar, tuple):
            return metavar
        if argument.nargs == ZERO_OR_MORE and len(metavar) == 2:
            return '%s[, %s]' % metavar
        if argument.nargs == ONE_OR_MORE:
            return '%s[, %s]' % metavar
        return ', '.join(metavar)
    if argument.dest not in (None, SUPPRESS):
        return argument.dest
    if argument.choices:
        return '{' + ','.join(map(str, argument.choices)) + '}'
    return None


class ArgumentError(Exception):
    """An error from creating or using an argument (optional or positional).

    The string value of this exception is the message, augmented with
    information about the argument that caused it.
    """

    def __init__(self, argument, message):
        # NO `super().__init__`: `args` is already what the constructor was
        # called with, recorded before this runs -- `BaseException.__new__`
        # does it in CPython and the raise does it here.
        self.argument_name = _get_action_name(argument)
        self.message = message

    def __str__(self):
        if self.argument_name is None:
            return self.message
        return (_('argument %(argument_name)s: %(message)s')
                % dict(message=self.message,
                       argument_name=self.argument_name))


class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass


# ── actions ─────────────────────────────────────────────────────────────────

class Action(_AttributeHolder):
    """Information about how to convert command line strings to Python objects.

    Action objects are used by an ArgumentParser to represent the information
    needed to parse a single argument from one or more strings from the
    command line. The keyword arguments to the Action constructor are also
    all attributes of Action instances.
    """

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None, deprecated=False):
        self.option_strings = option_strings
        self.dest = dest
        self.nargs = nargs
        self.const = const
        self.default = default
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        self.metavar = metavar
        self.deprecated = deprecated

    def _get_kwargs(self):
        names = ['option_strings', 'dest', 'nargs', 'const', 'default',
                 'type', 'choices', 'required', 'help', 'metavar',
                 'deprecated']
        return [(name, getattr(self, name)) for name in names]

    def format_usage(self):
        return self.option_strings[0]

    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError('.__call__() not defined')


class BooleanOptionalAction(Action):
    """`--flag` and `--no-flag`, one action: every long spelling gains a
    `--no-` twin that stores False."""

    def __init__(self, option_strings, dest, default=None, required=False,
                 help=None, deprecated=False):
        spellings = []
        for one in option_strings:
            spellings.append(one)
            if one.startswith('--'):
                if one.startswith('--no-'):
                    raise ValueError('invalid option name ' + repr(one)
                                     + ' for BooleanOptionalAction')
                spellings.append('--no-' + one[2:])
        super().__init__(option_strings=spellings, dest=dest, nargs=0,
                         default=default, required=required, help=help,
                         deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest,
                    not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


class _StoreAction(Action):

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None, deprecated=False):
        if nargs == 0:
            raise ValueError('nargs for store actions must be != 0; if you '
                             'have nothing to store, actions such as store '
                             'true or store const may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super().__init__(option_strings=option_strings, dest=dest,
                         nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required,
                         help=help, metavar=metavar, deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _StoreConstAction(Action):

    def __init__(self, option_strings, dest, const=None, default=None,
                 required=False, help=None, metavar=None, deprecated=False):
        # `metavar` IS ACCEPTED AND DROPPED, as in CPython: a flag takes no
        # value, so there is nothing for it to name.
        super().__init__(option_strings=option_strings, dest=dest, nargs=0,
                         const=const, default=default, required=required,
                         help=help, deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


class _StoreTrueAction(_StoreConstAction):

    def __init__(self, option_strings, dest, default=False, required=False,
                 help=None, deprecated=False):
        super().__init__(option_strings=option_strings, dest=dest,
                         const=True, deprecated=deprecated,
                         required=required, help=help, default=default)


class _StoreFalseAction(_StoreConstAction):

    def __init__(self, option_strings, dest, default=True, required=False,
                 help=None, deprecated=False):
        super().__init__(option_strings=option_strings, dest=dest,
                         const=False, default=default, required=required,
                         help=help, deprecated=deprecated)


class _AppendAction(Action):

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None, deprecated=False):
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super().__init__(option_strings=option_strings, dest=dest,
                         nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required,
                         help=help, metavar=metavar, deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        items = _copy_items(getattr(namespace, self.dest, None))
        items.append(values)
        setattr(namespace, self.dest, items)


class _AppendConstAction(Action):

    def __init__(self, option_strings, dest, const=None, default=None,
                 required=False, help=None, metavar=None, deprecated=False):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0,
                         const=const, default=default, required=required,
                         help=help, metavar=metavar, deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        items = _copy_items(getattr(namespace, self.dest, None))
        items.append(self.const)
        setattr(namespace, self.dest, items)


class _CountAction(Action):

    def __init__(self, option_strings, dest, default=None, required=False,
                 help=None, deprecated=False):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0,
                         default=default, required=required, help=help,
                         deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        count = getattr(namespace, self.dest, None)
        if count is None:
            count = 0
        setattr(namespace, self.dest, count + 1)


class _HelpAction(Action):

    def __init__(self, option_strings, dest=SUPPRESS, default=SUPPRESS,
                 help=None, deprecated=False):
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help,
                         deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


class _VersionAction(Action):

    def __init__(self, option_strings, version=None, dest=SUPPRESS,
                 default=SUPPRESS, help=None, deprecated=False):
        if help is None:
            help = _("show program's version number and exit")
        # `deprecated` IS NOT PASSED ON, which is CPython's behaviour rather
        # than a slip copied from it: `--version` is never reported as
        # deprecated, whatever it was declared with.
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help)
        self.version = version

    def __call__(self, parser, namespace, values, option_string=None):
        version = self.version
        if version is None:
            version = parser.version
        formatter = parser._get_formatter()
        formatter.add_text(version)
        parser._print_message(formatter.format_help(), _sys.stdout)
        parser.exit()


class _SubParsersAction(Action):

    class _ChoicesPseudoAction(Action):
        """A subcommand's line in its parent's help: the name, its aliases in
        parentheses, and the `help=` it was added with."""

        def __init__(self, name, aliases, help):
            metavar = dest = name
            if aliases:
                metavar += ' (%s)' % ', '.join(aliases)
            super().__init__(option_strings=[], dest=dest, help=help,
                             metavar=metavar)

    def __init__(self, option_strings, prog, parser_class, dest=SUPPRESS,
                 required=False, help=None, metavar=None):
        self._prog_prefix = prog
        self._parser_class = parser_class
        self._name_parser_map = {}
        self._choices_actions = []
        self._deprecated = set()
        self._color = True
        # THE MAP IS THE CHOICES, the same object: a parser added later is a
        # valid choice without anything being told.
        super().__init__(option_strings=option_strings, dest=dest,
                         nargs=PARSER, choices=self._name_parser_map,
                         required=required, help=help, metavar=metavar)

    def add_parser(self, name, *, deprecated=False, **kwargs):
        if kwargs.get('prog') is None:
            kwargs['prog'] = '%s %s' % (self._prog_prefix, name)
        if kwargs.get('color') is None:
            kwargs['color'] = self._color
        aliases = kwargs.pop('aliases', ())
        if name in self._name_parser_map:
            raise ValueError('conflicting subparser: ' + name)
        for alias in aliases:
            if alias in self._name_parser_map:
                raise ValueError('conflicting subparser alias: ' + alias)
        choice_action = None
        if 'help' in kwargs:
            choice_action = self._ChoicesPseudoAction(name, aliases,
                                                      kwargs.pop('help'))
            self._choices_actions.append(choice_action)
        parser = self._parser_class(**kwargs)
        if choice_action is not None:
            parser._check_help(choice_action)
        self._name_parser_map[name] = parser
        for alias in aliases:
            self._name_parser_map[alias] = parser
        if deprecated:
            self._deprecated.add(name)
            self._deprecated.update(aliases)
        return parser

    def _get_subactions(self):
        return self._choices_actions

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        arg_strings = values[1:]
        if self.dest is not SUPPRESS:
            setattr(namespace, self.dest, parser_name)
        try:
            subparser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') \
                % args
            raise ArgumentError(self, msg)
        if parser_name in self._deprecated:
            parser._warning(_("command '%(parser_name)s' is deprecated")
                            % {'parser_name': parser_name})
        # A NAMESPACE OF ITS OWN, copied across afterwards: the subparser's
        # defaults must win over the parent's for the names they share, and
        # parsing straight into the parent's namespace would let the parent's
        # already-set defaults stand. What the subparser did not recognise
        # is left on the namespace for the TOP parser to report.
        subnamespace, arg_strings = subparser.parse_known_args(arg_strings,
                                                               None)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)
        if arg_strings:
            if not hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                setattr(namespace, _UNRECOGNIZED_ARGS_ATTR, [])
            getattr(namespace, _UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)


class _ExtendAction(_AppendAction):

    def __call__(self, parser, namespace, values, option_string=None):
        items = _copy_items(getattr(namespace, self.dest, None))
        items.extend(values)
        setattr(namespace, self.dest, items)


# ── type classes ────────────────────────────────────────────────────────────

class FileType:
    """Deprecated factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.
    """

    def __init__(self, mode='r', bufsize=-1, encoding=None, errors=None):
        _warnings.warn(
            'FileType is deprecated. Simply open files after parsing '
            'arguments.', category=PendingDeprecationWarning, stacklevel=2)
        self._mode = mode
        self._bufsize = bufsize
        self._encoding = encoding
        self._errors = errors

    def __call__(self, string):
        # `-` IS THE STANDARD STREAM the mode points at.
        if string == '-':
            if 'r' in self._mode:
                return _sys.stdin.buffer if 'b' in self._mode else _sys.stdin
            if 'w' in self._mode or 'a' in self._mode or 'x' in self._mode:
                if 'b' in self._mode:
                    return _sys.stdout.buffer
                return _sys.stdout
            raise ValueError(_('argument "-" with mode %r') % self._mode)
        try:
            return open(string, self._mode, self._bufsize, self._encoding,
                        self._errors)
        except OSError as e:
            args = {'filename': string, 'error': e}
            message = _("can't open '%(filename)s': %(error)s")
            raise ArgumentTypeError(message % args)

    def __repr__(self):
        shown = [repr(one) for one in (self._mode, self._bufsize)
                 if one != -1]
        for name, value in (('encoding', self._encoding),
                            ('errors', self._errors)):
            if value is not None:
                shown.append(name + '=' + repr(value))
        return type(self).__name__ + '(' + ', '.join(shown) + ')'


# ── the namespace ───────────────────────────────────────────────────────────

class Namespace(_AttributeHolder):
    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__


# ── matching strings to arguments ───────────────────────────────────────────
#
# A command line is first reduced to one character per string: `A` for an
# argument, `O` for something that looks like an option, `-` for the first
# `--`. What an action may take is then a sequence of RUNS -- a set of those
# characters and how many of them, at least and at most -- and a run takes
# as many as it may. See the module docstring for why this is the regex
# engine's own rule.

def _runs_for(nargs, is_optional):
    """The runs one action's `nargs` stands for.

    AN OPTION NEVER TAKES THE `--`, so its runs never contain `-`; a
    positional may have any number of them on either side, and they are
    removed from what it is given afterwards."""
    if is_optional:
        if nargs is None:
            return [('A', 1, 1)]
        if nargs == OPTIONAL:
            return [('A', 0, 1)]
        if nargs == ZERO_OR_MORE:
            return [('A', 0, None)]
        if nargs == ONE_OR_MORE:
            return [('A', 1, None)]
        if nargs == REMAINDER:
            return [('AO', 0, None)]
        if nargs == PARSER:
            return [('A', 1, 1), ('AO', 0, None)]
        if nargs == SUPPRESS:
            return []
        # AN EXACT COUNT MAY TAKE THINGS THAT LOOK LIKE OPTIONS: `--pair -a
        # -b` gives `--pair` both, because it asked for two.
        return [('AO', nargs, nargs)]
    dashes = ('-', 0, None)
    if nargs is None:
        return [dashes, ('A', 1, 1), dashes]
    if nargs == OPTIONAL:
        return [dashes, ('A', 0, 1), dashes]
    if nargs == ZERO_OR_MORE:
        return [dashes, ('A-', 0, None)]
    if nargs == ONE_OR_MORE:
        return [dashes, ('A', 1, 1), ('A-', 0, None)]
    if nargs == REMAINDER:
        return [('AO-', 0, None)]
    if nargs == PARSER:
        return [dashes, ('A', 1, 1), ('AO-', 0, None)]
    if nargs == SUPPRESS:
        return [dashes]
    runs = []
    for _unused in range(nargs):
        runs.append(dashes)
        runs.append(('A', 1, 1))
    runs.append(dashes)
    return runs


def _match_runs(groups, pattern):
    """How many characters of `pattern` each group of runs takes, or None.

    Anchored at the start and NOT at the end: what is left over belongs to
    whatever comes next. The first allocation found by taking greedily and
    giving back one character at a time is the answer, which is precisely
    the order a backtracking regex engine searches in."""
    flat = []
    for index in range(len(groups)):
        for chars, low, high in groups[index]:
            flat.append((chars, low, high, index))
    taken = [0] * len(groups)
    if _walk_runs(flat, 0, 0, pattern, taken):
        return taken
    return None


def _walk_runs(flat, at, pos, pattern, taken):
    if at == len(flat):
        return True
    chars, low, high, group = flat[at]
    # A NEGATIVE COUNT MATCHES NOTHING, which is what `{-1}` does in a
    # pattern too -- it is not a quantifier there at all.
    if low < 0 or (high is not None and high < 0):
        return False
    limit = len(pattern) - pos
    if high is not None and high < limit:
        limit = high
    count = 0
    while count < limit and pattern[pos + count] in chars:
        count += 1
    while count >= low:
        taken[group] += count
        if _walk_runs(flat, at + 1, pos + count, pattern, taken):
            return True
        taken[group] -= count
        count -= 1
    return False


# ── containers of actions ───────────────────────────────────────────────────

class _ActionsContainer:

    def __init__(self, description, prefix_chars, argument_default,
                 conflict_handler):
        super().__init__()
        self.description = description
        self.argument_default = argument_default
        self.prefix_chars = prefix_chars
        self.conflict_handler = conflict_handler

        self._registries = {}
        self.register('action', None, _StoreAction)
        self.register('action', 'store', _StoreAction)
        self.register('action', 'store_const', _StoreConstAction)
        self.register('action', 'store_true', _StoreTrueAction)
        self.register('action', 'store_false', _StoreFalseAction)
        self.register('action', 'append', _AppendAction)
        self.register('action', 'append_const', _AppendConstAction)
        self.register('action', 'count', _CountAction)
        self.register('action', 'help', _HelpAction)
        self.register('action', 'version', _VersionAction)
        self.register('action', 'parsers', _SubParsersAction)
        self.register('action', 'extend', _ExtendAction)

        # AN UNKNOWN CONFLICT HANDLER IS REFUSED NOW, not at the first
        # conflict -- which might never come.
        self._get_handler()

        self._actions = []
        self._option_string_actions = {}
        self._action_groups = []
        self._mutually_exclusive_groups = []
        self._defaults = {}
        # A LIST SO IT CAN BE SHARED: a group adding `-1` has to change what
        # its parser thinks `-1` on a command line means.
        self._has_negative_number_optionals = []

    # -- registries ----------------------------------------------------------

    def register(self, registry_name, value, object):
        registry = self._registries.setdefault(registry_name, {})
        registry[value] = object

    def _registry_get(self, registry_name, value, default=None):
        return self._registries[registry_name].get(value, default)

    # -- defaults ------------------------------------------------------------

    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]

    def get_default(self, dest):
        for action in self._actions:
            if action.dest == dest and action.default is not None:
                return action.default
        return self._defaults.get(dest, None)

    # -- adding arguments ----------------------------------------------------

    def add_argument(self, *args, **kwargs):
        """
        add_argument(dest, ..., name=value, ...)
        add_argument(option_string, option_string, ..., name=value, ...)
        """
        chars = self.prefix_chars
        # ONE STRING THAT DOES NOT START WITH A PREFIX CHARACTER, or none at
        # all, is a positional; anything else is a set of option strings.
        if not args or len(args) == 1 and args[0][0] not in chars:
            if args and 'dest' in kwargs:
                raise TypeError('dest supplied twice for positional argument,'
                                ' did you mean metavar?')
            kwargs = self._get_positional_kwargs(*args, **kwargs)
        else:
            kwargs = self._get_optional_kwargs(*args, **kwargs)

        if 'default' not in kwargs:
            dest = kwargs['dest']
            if dest in self._defaults:
                kwargs['default'] = self._defaults[dest]
            elif self.argument_default is not None:
                kwargs['default'] = self.argument_default

        action_name = kwargs.get('action')
        action_class = self._pop_action_class(kwargs)
        if not callable(action_class):
            raise ValueError('unknown action ' + repr(action_class))
        action = action_class(**kwargs)

        if not action.option_strings and action.nargs == 0:
            raise ValueError('action ' + repr(action_name)
                             + ' is not valid for positional arguments')

        type_func = self._registry_get('type', action.type, action.type)
        if not callable(type_func):
            raise TypeError(repr(type_func) + ' is not callable')
        if type_func is FileType:
            raise TypeError(repr(type_func) + ' is a FileType class object, '
                            'instance of it must be passed')

        # THE METAVAR IS CHECKED AGAINST NARGS NOW, by laying the arguments
        # out once, so a mismatched tuple is an error where it was written
        # rather than at the first `--help`.
        if hasattr(self, '_get_validation_formatter'):
            formatter = self._get_validation_formatter()
            try:
                formatter._format_args(action, None)
            except TypeError:
                raise ValueError('length of metavar tuple does not match '
                                 'nargs')
        self._check_help(action)
        return self._add_action(action)

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = _MutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def _add_action(self, action):
        self._check_conflict(action)
        self._actions.append(action)
        action.container = self
        for option_string in action.option_strings:
            self._option_string_actions[option_string] = action
        for option_string in action.option_strings:
            if _looks_negative(option_string):
                if not self._has_negative_number_optionals:
                    self._has_negative_number_optionals.append(True)
        return action

    def _remove_action(self, action):
        self._actions.remove(action)

    def _add_container_actions(self, container):
        """A parent parser's arguments, groups and all, added to this one.

        GROUPS ARE MATCHED BY TITLE, so a parent's `options` land in this
        parser's `options` rather than in a second group of the same name."""
        title_group_map = {}
        for group in self._action_groups:
            if group.title in title_group_map:
                raise ValueError('cannot merge actions - two groups are '
                                 'named ' + repr(group.title))
            title_group_map[group.title] = group

        group_map = {}
        for group in container._action_groups:
            if group.title not in title_group_map:
                title_group_map[group.title] = self.add_argument_group(
                    title=group.title, description=group.description,
                    conflict_handler=group.conflict_handler)
            for action in group._group_actions:
                group_map[action] = title_group_map[group.title]

        for group in container._mutually_exclusive_groups:
            if group._container is container:
                cont = self
            else:
                cont = title_group_map[group._container.title]
            mutex_group = cont.add_mutually_exclusive_group(
                required=group.required)
            for action in group._group_actions:
                group_map[action] = mutex_group

        for action in container._actions:
            group_map.get(action, self)._add_action(action)

    def _get_positional_kwargs(self, dest, **kwargs):
        if 'required' in kwargs:
            raise TypeError("'required' is an invalid argument for "
                            "positionals")
        # A POSITIONAL IS REQUIRED UNLESS ITS NARGS LETS IT BE ABSENT.
        nargs = kwargs.get('nargs')
        if nargs == 0:
            raise ValueError('nargs for positionals must be != 0')
        if nargs not in [OPTIONAL, ZERO_OR_MORE, REMAINDER, SUPPRESS]:
            kwargs['required'] = True
        return dict(kwargs, dest=dest, option_strings=[])

    def _get_optional_kwargs(self, *args, **kwargs):
        option_strings = []
        long_option_strings = []
        option_string = None
        for option_string in args:
            if not option_string[0] in self.prefix_chars:
                raise ValueError(
                    'invalid option string ' + repr(option_string)
                    + ': must start with a character '
                    + repr(self.prefix_chars))
            option_strings.append(option_string)
            if len(option_string) > 1 \
                    and option_string[1] in self.prefix_chars:
                long_option_strings.append(option_string)

        # THE DEST COMES FROM THE FIRST LONG SPELLING, or the first of any
        # kind: `--foo-bar` is `foo_bar` and `-x` is `x`.
        dest = kwargs.pop('dest', None)
        if dest is None:
            if long_option_strings:
                dest_option_string = long_option_strings[0]
            else:
                dest_option_string = option_strings[0]
            dest = dest_option_string.lstrip(self.prefix_chars)
            if not dest:
                raise TypeError('dest= is required for options like '
                                + repr(option_string))
            dest = dest.replace('-', '_')
        return dict(kwargs, dest=dest, option_strings=option_strings)

    def _pop_action_class(self, kwargs, default=None):
        action = kwargs.pop('action', default)
        return self._registry_get('action', action, action)

    def _get_handler(self):
        handler_func_name = '_handle_conflict_%s' % self.conflict_handler
        try:
            return getattr(self, handler_func_name)
        except AttributeError:
            raise ValueError('invalid conflict_resolution value: '
                             + repr(self.conflict_handler))

    def _check_conflict(self, action):
        confl_optionals = []
        for option_string in action.option_strings:
            if option_string in self._option_string_actions:
                confl_optionals.append(
                    (option_string,
                     self._option_string_actions[option_string]))
        if confl_optionals:
            self._get_handler()(action, confl_optionals)

    def _handle_conflict_error(self, action, conflicting_actions):
        message = ngettext('conflicting option string: %s',
                           'conflicting option strings: %s',
                           len(conflicting_actions))
        conflict_string = ', '.join([option_string for option_string, _unused
                                     in conflicting_actions])
        raise ArgumentError(action, message % conflict_string)

    def _handle_conflict_resolve(self, action, conflicting_actions):
        # THE NEWER ACTION WINS the spelling; an older one left with no
        # spelling at all is removed from its container.
        for option_string, older in conflicting_actions:
            older.option_strings.remove(option_string)
            self._option_string_actions.pop(option_string, None)
            if not older.option_strings:
                older.container._remove_action(older)

    def _check_help(self, action):
        if action.help and hasattr(self, '_get_validation_formatter'):
            formatter = self._get_validation_formatter()
            try:
                formatter._expand_help(action)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('badly formed help string') from exc


class _ArgumentGroup(_ActionsContainer):

    def __init__(self, container, title=None, description=None, **kwargs):
        if 'prefix_chars' in kwargs:
            _warnings.warn(
                "The use of the undocumented 'prefix_chars' parameter in "
                "ArgumentParser.add_argument_group() is deprecated.",
                DeprecationWarning, stacklevel=3)
        kwargs.setdefault('conflict_handler', container.conflict_handler)
        kwargs.setdefault('prefix_chars', container.prefix_chars)
        kwargs.setdefault('argument_default', container.argument_default)
        super().__init__(description=description, **kwargs)

        self.title = title
        self._group_actions = []

        # A GROUP IS A VIEW OF ITS PARSER, not a parser of its own: the
        # actions, the option strings and the defaults are the container's,
        # the same objects, so adding to the group adds to the parser.
        self._registries = container._registries
        self._actions = container._actions
        self._option_string_actions = container._option_string_actions
        self._defaults = container._defaults
        self._has_negative_number_optionals = \
            container._has_negative_number_optionals
        self._mutually_exclusive_groups = container._mutually_exclusive_groups

    def _add_action(self, action):
        action = super()._add_action(action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action):
        super()._remove_action(action)
        self._group_actions.remove(action)

    def add_argument_group(self, *args, **kwargs):
        raise ValueError('argument groups cannot be nested')


class _MutuallyExclusiveGroup(_ArgumentGroup):

    def __init__(self, container, required=False):
        super().__init__(container)
        self.required = required
        self._container = container

    def _add_action(self, action):
        if action.required:
            raise ValueError('mutually exclusive arguments must be optional')
        action = self._container._add_action(action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action):
        self._container._remove_action(action)
        self._group_actions.remove(action)

    def add_mutually_exclusive_group(self, **kwargs):
        raise ValueError('mutually exclusive groups cannot be nested')


def _prog_name(prog=None):
    if prog is not None:
        return prog
    return _os.path.basename(_sys.argv[0])


# ── one parse ───────────────────────────────────────────────────────────────

class _Parse:
    """The state of ONE call of `_parse_known_args`.

    Which strings are options and which actions they name is decided once,
    up front; then positionals and options are consumed alternately, each
    consumer advancing a shared position. `extras` gathers what nothing
    claimed, with its A/O shape alongside for the intermixed second pass."""

    def __init__(self, parser, arg_strings, namespace):
        self.parser = parser
        self.namespace = namespace
        self.args = arg_strings
        # WHO EXCLUDES WHOM: every member of a mutually exclusive group
        # conflicts with every other member.
        self.conflicts = {}
        for group in parser._mutually_exclusive_groups:
            members = group._group_actions
            for i in range(len(members)):
                against = self.conflicts.setdefault(members[i], [])
                against.extend(members[:i])
                against.extend(members[i + 1:])
        self.options = {}
        marks = []
        count = len(arg_strings)
        i = 0
        while i < count:
            one = arg_strings[i]
            if one == '--':
                # EVERYTHING AFTER THE FIRST `--` IS AN ARGUMENT, however it
                # looks -- and is not even asked.
                marks.append('-')
                marks.extend(['A'] * (count - i - 1))
                break
            found = parser._parse_optional(one)
            if found is None:
                marks.append('A')
            else:
                self.options[i] = found
                marks.append('O')
            i += 1
        self.pattern = ''.join(marks)
        self.seen = set()
        self.seen_non_default = set()
        self.warned = set()
        self.extras = []
        self.extras_pattern = []
        self.positionals = parser._get_positional_actions()

    def take(self, action, argument_strings, option_string=None):
        self.seen.add(action)
        values = self.parser._get_values(action, argument_strings)
        # A POSITIONAL THAT GOT NOTHING HAS NOT BEEN USED, as far as a
        # mutually exclusive group is concerned: it was only given its
        # default.
        if action.option_strings or argument_strings:
            self.seen_non_default.add(action)
            for other in self.conflicts.get(action, []):
                if other in self.seen_non_default:
                    raise ArgumentError(
                        action, _('not allowed with argument %s')
                        % _get_action_name(other))
        if values is not SUPPRESS:
            action(self.parser, self.namespace, values, option_string)

    def consume_optional(self, start):
        parser = self.parser
        found = self.options[start]
        if len(found) > 1:
            options = ', '.join([option_string
                                 for _a, option_string, _s, _e in found])
            raise ArgumentError(None, _('ambiguous option: %(option)s could '
                                        'match %(matches)s')
                                % {'option': self.args[start],
                                   'matches': options})
        action, option_string, sep, explicit_arg = found[0]
        chars = parser.prefix_chars
        taken = []
        while True:
            if action is None:
                # NOT OURS -- perhaps a subparser's. It goes to the extras
                # whole.
                self.extras.append(self.args[start])
                self.extras_pattern.append('O')
                return start + 1
            if explicit_arg is not None:
                arg_count = parser._match_argument(action, 'A')
                if (arg_count == 0 and option_string[1] not in chars
                        and explicit_arg != ''):
                    # `-xyz` WITH `-x` A FLAG: the rest is more single-dash
                    # options, `-y` and then `-z`, unless it was written with
                    # an `=` or starts with a prefix character.
                    if sep or explicit_arg[0] in chars:
                        raise ArgumentError(
                            action, _('ignored explicit argument %r')
                            % explicit_arg)
                    taken.append((action, [], option_string))
                    char = option_string[0]
                    option_string = char + explicit_arg[0]
                    if option_string in parser._option_string_actions:
                        action = parser._option_string_actions[option_string]
                        explicit_arg = explicit_arg[1:]
                        if not explicit_arg:
                            sep = explicit_arg = None
                        elif explicit_arg[0] == '=':
                            sep = '='
                            explicit_arg = explicit_arg[1:]
                        else:
                            sep = ''
                    else:
                        self.extras.append(char + explicit_arg)
                        self.extras_pattern.append('O')
                        stop = start + 1
                        break
                elif arg_count == 1:
                    stop = start + 1
                    taken.append((action, [explicit_arg], option_string))
                    break
                else:
                    raise ArgumentError(
                        action, _('ignored explicit argument %r')
                        % explicit_arg)
            else:
                first = start + 1
                arg_count = parser._match_argument(action,
                                                   self.pattern[first:])
                stop = first + arg_count
                taken.append((action, self.args[first:stop], option_string))
                break
        for action, args, option_string in taken:
            if action.deprecated and option_string not in self.warned:
                parser._warning(_("option '%(option)s' is deprecated")
                                % {'option': option_string})
                self.warned.add(option_string)
            self.take(action, args, option_string)
        return stop

    def consume_positionals(self, start):
        parser = self.parser
        counts = parser._match_arguments_partial(self.positionals,
                                                 self.pattern[start:])
        for action, arg_count in zip(self.positionals, counts):
            args = self.args[start:start + arg_count]
            # THE `--` IS NOT AN ARGUMENT, and comes out of what a positional
            # is given -- except under REMAINDER, which takes the rest of the
            # line exactly as written, and PARSER, where only a leading one
            # is the subcommand's to lose.
            if action.nargs == PARSER:
                if self.pattern[start] == '-':
                    args.remove('--')
            elif action.nargs != REMAINDER:
                if self.pattern.find('-', start, start + arg_count) >= 0:
                    args.remove('--')
            start += arg_count
            if args and action.deprecated and action.dest not in self.warned:
                parser._warning(_("argument '%(argument_name)s' is "
                                  "deprecated")
                                % {'argument_name': action.dest})
                self.warned.add(action.dest)
            self.take(action, args)
        self.positionals[:] = self.positionals[len(counts):]
        return start

    def run(self, intermixed):
        start = 0
        last = max(self.options) if self.options else -1
        while start <= last:
            next_option = start
            while next_option not in self.options:
                next_option += 1
            if not intermixed and start != next_option:
                end = self.consume_positionals(start)
                # ONLY WHEN THE POSITIONALS TOOK SOMETHING does the loop go
                # round again: they may have swallowed the option too.
                if end > start:
                    start = end
                    continue
                start = end
            if start not in self.options:
                self.extras.extend(self.args[start:next_option])
                self.extras_pattern.extend(self.pattern[start:next_option])
                start = next_option
            start = self.consume_optional(start)

        if not intermixed:
            stop = self.consume_positionals(start)
            self.extras.extend(self.args[stop:])
            return self.extras

        # INTERMIXED: the options have all been taken with the positionals
        # held back; now the positionals get everything that is left that is
        # not an option, as if it had been written contiguously.
        self.extras.extend(self.args[start:])
        self.extras_pattern.extend(self.pattern[start:])
        marks = ''.join(self.extras_pattern)
        self.args = [one for one, mark in zip(self.extras, marks)
                     if mark != 'O']
        self.pattern = marks.replace('O', '')
        stop = self.consume_positionals(0)
        for i in range(len(marks)):
            if not stop:
                break
            if marks[i] != 'O':
                stop -= 1
                self.extras[i] = None
        return [one for one in self.extras if one is not None]


# ── the parser ──────────────────────────────────────────────────────────────

class ArgumentParser(_AttributeHolder, _ActionsContainer):
    """Object for parsing command line strings into Python objects.

    Keyword Arguments:
        - prog -- The name of the program (default:
            ``os.path.basename(sys.argv[0])``)
        - usage -- A usage message (default: auto-generated from arguments)
        - description -- A description of what the program does
        - epilog -- Text following the argument descriptions
        - parents -- Parsers whose arguments should be copied into this one
        - formatter_class -- HelpFormatter class for printing help messages
        - prefix_chars -- Characters that prefix optional arguments
        - fromfile_prefix_chars -- Characters that prefix files containing
            additional arguments
        - argument_default -- The default value for all arguments
        - conflict_handler -- String indicating how to handle conflicts
        - add_help -- Add a -h/-help option
        - allow_abbrev -- Allow long options to be abbreviated unambiguously
        - exit_on_error -- Determines whether or not ArgumentParser exits with
            error info when an error occurs
        - suggest_on_error - Enables suggestions for mistyped argument choices
            and subparser names (default: ``False``)
        - color - Allow color output in help messages (default: ``False``)
    """

    def __init__(self, prog=None, usage=None, description=None, epilog=None,
                 parents=[], formatter_class=HelpFormatter, prefix_chars='-',
                 fromfile_prefix_chars=None, argument_default=None,
                 conflict_handler='error', add_help=True, allow_abbrev=True,
                 exit_on_error=True, *, suggest_on_error=False, color=True):
        super().__init__(description=description, prefix_chars=prefix_chars,
                         argument_default=argument_default,
                         conflict_handler=conflict_handler)
        self.prog = _prog_name(prog)
        self.usage = usage
        self.epilog = epilog
        self.formatter_class = formatter_class
        self.fromfile_prefix_chars = fromfile_prefix_chars
        self.add_help = add_help
        self.allow_abbrev = allow_abbrev
        self.exit_on_error = exit_on_error
        self.suggest_on_error = suggest_on_error
        self.color = color
        self._cached_formatter = None

        add_group = self.add_argument_group
        self._positionals = add_group(_('positional arguments'))
        self._optionals = add_group(_('options'))
        self._subparsers = None

        self.register('type', None, _identity)

        # `-h`/`--help` SPELLED WITH THE PARSER'S OWN PREFIX: `+h`/`++help`
        # for a parser whose prefix characters do not include `-`. The
        # explicit default keeps `argument_default` off it.
        default_prefix = '-' if '-' in prefix_chars else prefix_chars[0]
        if self.add_help:
            self.add_argument(default_prefix + 'h', default_prefix * 2 + 'help',
                              action='help', default=SUPPRESS,
                              help=_('show this help message and exit'))

        for parent in parents:
            if not isinstance(parent, ArgumentParser):
                raise TypeError('parents must be a list of ArgumentParser')
            self._add_container_actions(parent)
            self._defaults.update(parent._defaults)

    def _get_kwargs(self):
        names = ['prog', 'usage', 'description', 'formatter_class',
                 'conflict_handler', 'add_help']
        return [(name, getattr(self, name)) for name in names]

    # -- positionals and subcommands -----------------------------------------

    def add_subparsers(self, **kwargs):
        if self._subparsers is not None:
            raise ValueError('cannot have multiple subparser arguments')
        kwargs.setdefault('parser_class', type(self))
        if 'title' in kwargs or 'description' in kwargs:
            title = kwargs.pop('title', _('subcommands'))
            description = kwargs.pop('description', None)
            self._subparsers = self.add_argument_group(title, description)
        else:
            self._subparsers = self._positionals

        # A SUBCOMMAND'S PROG IS THIS PARSER'S USAGE WITHOUT THE OPTIONS --
        # `prog pos1 pos2` -- because those are what has to have been typed
        # before the subcommand's name. Laid out without colour, since it is
        # stored rather than printed.
        if kwargs.get('prog') is None:
            formatter = self.formatter_class(prog=self.prog)
            formatter._set_color(False)
            positionals = self._get_positional_actions()
            groups = self._mutually_exclusive_groups
            formatter.add_usage(None, positionals, groups, '')
            kwargs['prog'] = formatter.format_help().strip()

        parsers_class = self._pop_action_class(kwargs, 'parsers')
        action = parsers_class(option_strings=[], **kwargs)
        action._color = self.color
        self._check_help(action)
        self._subparsers._add_action(action)
        return action

    def _add_action(self, action):
        if action.option_strings:
            self._optionals._add_action(action)
        else:
            self._positionals._add_action(action)
        return action

    def _get_optional_actions(self):
        return [action for action in self._actions if action.option_strings]

    def _get_positional_actions(self):
        return [action for action in self._actions
                if not action.option_strings]

    # -- parsing -------------------------------------------------------------

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s') % ' '.join(argv)
            if self.exit_on_error:
                self.error(msg)
            else:
                raise ArgumentError(None, msg)
        return args

    def parse_known_args(self, args=None, namespace=None):
        return self._parse_known_args2(args, namespace, intermixed=False)

    def _parse_known_args2(self, args, namespace, intermixed):
        if args is None:
            args = _sys.argv[1:]
        else:
            args = list(args)
        if namespace is None:
            namespace = Namespace()

        # DEFAULTS FIRST, and only where the namespace has nothing: a
        # namespace passed in keeps what it already holds.
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        setattr(namespace, action.dest, action.default)
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        if self.exit_on_error:
            try:
                namespace, args = self._parse_known_args(args, namespace,
                                                         intermixed)
            except ArgumentError as err:
                self.error(str(err))
        else:
            namespace, args = self._parse_known_args(args, namespace,
                                                     intermixed)

        if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
            args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
        return namespace, args

    def _parse_known_args(self, arg_strings, namespace, intermixed):
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)
        one = _Parse(self, arg_strings, namespace)
        extras = one.run(intermixed)

        # WHAT WAS NEVER SEEN is either missing, if it was required, or gets
        # its default CONVERTED -- a string default goes through `type` like
        # anything typed, but only now, so that a default that would not
        # convert is harmless when the argument was given.
        required_actions = []
        for action in self._actions:
            if action not in one.seen:
                if action.required:
                    required_actions.append(_get_action_name(action))
                elif (action.default is not None
                      and isinstance(action.default, str)
                      and hasattr(namespace, action.dest)
                      and action.default is getattr(namespace, action.dest)):
                    setattr(namespace, action.dest,
                            self._get_value(action, action.default))
        if required_actions:
            raise ArgumentError(None, _('the following arguments are '
                                        'required: %s')
                                % ', '.join(required_actions))

        for group in self._mutually_exclusive_groups:
            if not group.required:
                continue
            used = False
            for action in group._group_actions:
                if action in one.seen_non_default:
                    used = True
                    break
            if not used:
                names = [_get_action_name(action)
                         for action in group._group_actions
                         if action.help is not SUPPRESS]
                raise ArgumentError(None, _('one of the arguments %s is '
                                            'required') % ' '.join(names))
        return namespace, extras

    def _read_args_from_files(self, arg_strings):
        new_arg_strings = []
        for arg_string in arg_strings:
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)
                continue
            # ONE ARGUMENT PER LINE by default, and a file may name further
            # files: the lines are expanded again before they are used.
            try:
                with open(arg_string[1:], encoding='utf-8') as args_file:
                    found = []
                    for arg_line in args_file.read().splitlines():
                        for arg in self.convert_arg_line_to_args(arg_line):
                            found.append(arg)
                    new_arg_strings.extend(self._read_args_from_files(found))
            except OSError as err:
                raise ArgumentError(None, str(err))
        return new_arg_strings

    def convert_arg_line_to_args(self, arg_line):
        return [arg_line]

    def _match_argument(self, action, arg_strings_pattern):
        got = _match_runs([_runs_for(action.nargs,
                                     bool(action.option_strings))],
                          arg_strings_pattern)
        if got is None:
            nargs_errors = {
                None: _('expected one argument'),
                OPTIONAL: _('expected at most one argument'),
                ONE_OR_MORE: _('expected at least one argument'),
            }
            msg = nargs_errors.get(action.nargs)
            if msg is None:
                msg = ngettext('expected %s argument',
                               'expected %s arguments',
                               action.nargs) % action.nargs
            raise ArgumentError(action, msg)
        return got[0]

    def _match_arguments_partial(self, actions, arg_strings_pattern):
        """As many of `actions` as can be satisfied from the front of the
        pattern, and how much each takes.

        FEWER AND FEWER POSITIONALS are tried until the pattern fits. When
        what stops the match is an OPTION, the positionals at the end that
        got nothing are handed back: they may yet get something after it."""
        for i in range(len(actions), 0, -1):
            groups = [_runs_for(action.nargs, bool(action.option_strings))
                      for action in actions[:i]]
            got = _match_runs(groups, arg_strings_pattern)
            if got is not None:
                end = sum(got)
                if (end < len(arg_strings_pattern)
                        and arg_strings_pattern[end] == 'O'):
                    while got and not got[-1]:
                        del got[-1]
                return got
        return []

    def _parse_optional(self, arg_string):
        """None when `arg_string` is an argument; otherwise every way of
        reading it as an option, as (action, option_string, sep,
        explicit_arg) -- more than one is an ambiguous abbreviation, and an
        action of None an option this parser does not have."""
        if not arg_string:
            return None
        if not arg_string[0] in self.prefix_chars:
            return None
        if arg_string in self._option_string_actions:
            return [(self._option_string_actions[arg_string], arg_string,
                     None, None)]
        # A LONE PREFIX CHARACTER -- `-` is the usual spelling of stdin -- is
        # an argument.
        if len(arg_string) == 1:
            return None
        option_string, sep, explicit_arg = arg_string.partition('=')
        if sep and option_string in self._option_string_actions:
            return [(self._option_string_actions[option_string],
                     option_string, sep, explicit_arg)]
        option_tuples = self._get_option_tuples(arg_string)
        if option_tuples:
            return option_tuples
        # A NEGATIVE NUMBER is an argument -- unless this parser has options
        # that look like one, in which case it cannot be.
        if _looks_negative(arg_string):
            if not self._has_negative_number_optionals:
                return None
        # WITH A SPACE IN IT, it was never typed as an option.
        if ' ' in arg_string:
            return None
        return [(None, arg_string, None, None)]

    def _get_option_tuples(self, option_string):
        result = []
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            # A LONG OPTION is split only at `=`, and matches every option
            # it abbreviates.
            if self.allow_abbrev:
                option_prefix, sep, explicit_arg = option_string.partition('=')
                if not sep:
                    sep = explicit_arg = None
                for candidate in self._option_string_actions:
                    if candidate.startswith(option_prefix):
                        result.append((self._option_string_actions[candidate],
                                       candidate, sep, explicit_arg))
        elif option_string[0] in chars and option_string[1] not in chars:
            # A SHORT OPTION may carry its argument glued on -- `-n5` -- or
            # abbreviate a longer single-dash one.
            option_prefix, sep, explicit_arg = option_string.partition('=')
            if not sep:
                sep = explicit_arg = None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]
            for candidate in self._option_string_actions:
                if candidate == short_option_prefix:
                    result.append((self._option_string_actions[candidate],
                                   candidate, '', short_explicit_arg))
                elif self.allow_abbrev \
                        and candidate.startswith(option_prefix):
                    result.append((self._option_string_actions[candidate],
                                   candidate, sep, explicit_arg))
        else:
            raise ArgumentError(None, _('unexpected option string: %s')
                                % option_string)
        return result

    def parse_intermixed_args(self, args=None, namespace=None):
        args, argv = self.parse_known_intermixed_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s') % ' '.join(argv)
            if self.exit_on_error:
                self.error(msg)
            else:
                raise ArgumentError(None, msg)
        return args

    def parse_known_intermixed_args(self, args=None, namespace=None):
        # A SUBCOMMAND OR A REMAINDER cannot be intermixed: both take the
        # rest of the line as it stands, options included.
        for action in self._get_positional_actions():
            if action.nargs in [PARSER, REMAINDER]:
                raise TypeError('parse_intermixed_args: positional arg'
                                ' with nargs=%s' % action.nargs)
        return self._parse_known_args2(args, namespace, intermixed=True)

    # -- converting values ---------------------------------------------------

    def _get_values(self, action, arg_strings):
        if not arg_strings and action.nargs == OPTIONAL:
            # `--opt` GIVEN BARE takes `const`; a positional left out takes
            # its default -- and either, when it is a string, is converted.
            if action.option_strings:
                value = action.const
            else:
                value = action.default
            if isinstance(value, str) and value is not SUPPRESS:
                value = self._get_value(action, value)
        elif (not arg_strings and action.nargs == ZERO_OR_MORE
              and not action.option_strings):
            if action.default is not None:
                value = action.default
            else:
                value = []
        elif len(arg_strings) == 1 and action.nargs in [None, OPTIONAL]:
            value = self._get_value(action, arg_strings[0])
            self._check_value(action, value)
        elif action.nargs == REMAINDER:
            value = [self._get_value(action, v) for v in arg_strings]
        elif action.nargs == PARSER:
            # ONLY THE SUBCOMMAND'S NAME is a choice; the rest is its own.
            value = [self._get_value(action, v) for v in arg_strings]
            self._check_value(action, value[0])
        elif action.nargs == SUPPRESS:
            value = SUPPRESS
        else:
            value = [self._get_value(action, v) for v in arg_strings]
            for v in value:
                self._check_value(action, v)
        return value

    def _get_value(self, action, arg_string):
        type_func = self._registry_get('type', action.type, action.type)
        if not callable(type_func):
            raise TypeError(repr(type_func) + ' is not callable')
        try:
            return type_func(arg_string)
        except ArgumentTypeError as err:
            raise ArgumentError(action, str(err))
        except (TypeError, ValueError):
            # THE TYPE'S NAME, and the string exactly as it was typed.
            name = getattr(action.type, '__name__', repr(action.type))
            raise ArgumentError(action, _('invalid %(type)s value: %(value)r')
                                % {'type': name, 'value': arg_string})

    def _check_value(self, action, value):
        choices = action.choices
        if choices is None:
            return
        # A STRING OF CHOICES IS ITS CHARACTERS, not its substrings: `in` on
        # a str would accept `ab` from `abc`.
        if isinstance(choices, str):
            choices = iter(choices)
        if value not in choices:
            args = {'value': str(value),
                    'choices': ', '.join([repr(str(choice))
                                          for choice in action.choices])}
            msg = _('invalid choice: %(value)r (choose from %(choices)s)')
            # A SUGGESTION ONLY BETWEEN STRINGS: the closest choice by
            # `difflib`'s ratio, when one scores 0.6 or better. Subcommand
            # names come through here too, as the choices of the PARSER
            # action.
            if self.suggest_on_error and isinstance(value, str):
                if all(isinstance(choice, str) for choice in action.choices):
                    import difflib
                    suggestions = difflib.get_close_matches(
                        value, action.choices, 1)
                    if suggestions:
                        args['closest'] = suggestions[0]
                        msg = _('invalid choice: %(value)r, maybe you meant '
                                '%(closest)r? (choose from %(choices)s)')
            raise ArgumentError(action, msg % args)

    # -- help ----------------------------------------------------------------

    def format_usage(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        return formatter.format_help()

    def format_help(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()

    def _get_formatter(self):
        formatter = self.formatter_class(prog=self.prog)
        formatter._set_color(self.color)
        return formatter

    def _get_validation_formatter(self):
        # ONE FORMATTER FOR EVERY CHECK `add_argument` MAKES: the checks
        # only read, and building one asks the environment about colour.
        if self._cached_formatter is None:
            self._cached_formatter = self._get_formatter()
        return self._cached_formatter

    def print_usage(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_usage(), file)

    def print_help(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_help(), file)

    def _print_message(self, message, file=None):
        if message:
            file = file or _sys.stderr
            try:
                file.write(message)
            except (AttributeError, OSError):
                pass

    # -- exiting -------------------------------------------------------------

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, _sys.stderr)
        _sys.exit(status)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_usage(_sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, _('%(prog)s: error: %(message)s\n') % args)

    def _warning(self, message):
        args = {'prog': self.prog, 'message': message}
        self._print_message(_('%(prog)s: warning: %(message)s\n') % args,
                            _sys.stderr)
