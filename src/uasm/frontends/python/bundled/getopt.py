"""`getopt`, as ordinary Python this compiler compiles.

COVERAGE: `getopt` and `gnu_getopt` over short options (`ab:c::` -- a flag,
one that needs an argument, one whose argument is optional) and long
options (`name`, `name=`, `name=?`), clustered short options, glued and
separate arguments, `--name=value`, unambiguous abbreviations of long
options, `--` and a lone `-`; `gnu_getopt`'s intermixing, its `+` prefix
and `POSIXLY_CORRECT` that stop at the first operand, and its `-` prefix
that returns operands in order; `GetoptError` and its alias `error`, with
`msg` and `opt`, for every message CPython raises.

Messages are never translated, for the reason `bundled/argparse.py` gives.

## One rule decides every short option

An option letter is looked for in `shortopts` wherever it appears as
itself and not as a `:`. One colon after it means an argument is
REQUIRED -- the rest of this cluster if there is any, else the next
string -- and two mean it is OPTIONAL: the rest of the cluster or
nothing, never the next string. A long option is matched by prefix
against `longopts`; an exact match wins outright, and otherwise the
prefix must pick out one option.
"""

import os as _os

__all__ = ["GetoptError", "error", "getopt", "gnu_getopt"]


class GetoptError(Exception):
    opt = ''
    msg = ''

    def __init__(self, msg, opt=''):
        self.msg = msg
        self.opt = opt
        Exception.__init__(self, msg, opt)

    def __str__(self):
        return self.msg


error = GetoptError


def getopt(args, shortopts, longopts=[]):
    """getopt(args, options[, long_options]) -> opts, args

    Parses command line options and parameter list. Scanning stops at the
    first argument that is not an option, as Unix getopt() does.
    """
    opts = []
    longopts = [longopts] if isinstance(longopts, str) else list(longopts)
    # `-` ALONE IS AN OPERAND -- the usual spelling of standard input -- and
    # `--` ends the options and is itself dropped.
    while args and args[0].startswith('-') and args[0] != '-':
        if args[0] == '--':
            args = args[1:]
            break
        if args[0].startswith('--'):
            opts, args = _do_longs(opts, args[0][2:], longopts, args[1:])
        else:
            opts, args = _do_shorts(opts, args[0][1:], shortopts, args[1:])
    return opts, args


def gnu_getopt(args, shortopts, longopts=[]):
    """getopt(args, options[, long_options]) -> opts, args

    GNU-style scanning: options and operands may be intermixed. A `+` at
    the start of `shortopts`, or `POSIXLY_CORRECT` in the environment,
    stops at the first operand as `getopt` does; a `-` returns the operands
    IN ORDER, each run of them as a `(None, [operands])` entry between the
    options it fell between.
    """
    opts = []
    prog_args = []
    longopts = [longopts] if isinstance(longopts, str) else list(longopts)
    in_order = False
    if shortopts.startswith('-'):
        shortopts = shortopts[1:]
        options_first = False
        in_order = True
    elif shortopts.startswith('+'):
        shortopts = shortopts[1:]
        options_first = True
    elif _os.environ.get("POSIXLY_CORRECT"):
        options_first = True
    else:
        options_first = False
    while args:
        if args[0] == '--':
            prog_args += args[1:]
            break
        if args[0][:2] == '--':
            if in_order and prog_args:
                opts.append((None, prog_args))
                prog_args = []
            opts, args = _do_longs(opts, args[0][2:], longopts, args[1:])
        elif args[0][:1] == '-' and args[0] != '-':
            if in_order and prog_args:
                opts.append((None, prog_args))
                prog_args = []
            opts, args = _do_shorts(opts, args[0][1:], shortopts, args[1:])
        elif options_first:
            prog_args += args
            break
        else:
            prog_args.append(args[0])
            args = args[1:]
    return opts, prog_args


def _do_longs(opts, opt, longopts, args):
    """One `--name` or `--name=value`, and whatever it takes from `args`."""
    optarg = None
    at = opt.find('=')
    if at >= 0:
        opt, optarg = opt[:at], opt[at + 1:]
    has_arg, opt = _long_has_args(opt, longopts)
    if has_arg:
        # AN OPTIONAL LONG ARGUMENT is only ever the one after `=`: the next
        # string stays an operand.
        if optarg is None and has_arg != '?':
            if not args:
                raise GetoptError('option --%s requires argument' % opt, opt)
            optarg, args = args[0], args[1:]
    elif optarg is not None:
        raise GetoptError('option --%s must not have an argument' % opt, opt)
    opts.append(('--' + opt, optarg or ''))
    return opts, args


def _long_has_args(opt, longopts):
    """(takes an argument -- True, False or '?' -- and the full name)."""
    possibilities = [one for one in longopts if one.startswith(opt)]
    if not possibilities:
        raise GetoptError('option --%s not recognized' % opt, opt)
    # AN EXACT NAME WINS, however many longer ones it also abbreviates.
    if opt in possibilities:
        return False, opt
    if opt + '=' in possibilities:
        return True, opt
    if opt + '=?' in possibilities:
        return '?', opt
    if len(possibilities) > 1:
        raise GetoptError("option --%s not a unique prefix; possible "
                          "options: %s" % (opt, ", ".join(possibilities)),
                          opt)
    match = possibilities[0]
    if match.endswith('=?'):
        return '?', match[:-2]
    if match.endswith('='):
        return True, match[:-1]
    return False, match


def _do_shorts(opts, optstring, shortopts, args):
    """One cluster like `-abc` or `-ofile`, left to right."""
    while optstring != '':
        opt, optstring = optstring[0], optstring[1:]
        has_arg = _short_has_arg(opt, shortopts)
        if has_arg:
            # THE REST OF THE CLUSTER IS THE ARGUMENT; with nothing left, a
            # required one is the next string and an optional one is ''.
            if optstring == '' and has_arg != '?':
                if not args:
                    raise GetoptError('option -%s requires argument' % opt,
                                      opt)
                optstring, args = args[0], args[1:]
            optarg, optstring = optstring, ''
        else:
            optarg = ''
        opts.append(('-' + opt, optarg))
    return opts, args


def _short_has_arg(opt, shortopts):
    for i in range(len(shortopts)):
        if opt == shortopts[i] and opt != ':':
            if not shortopts.startswith(':', i + 1):
                return False
            if shortopts.startswith('::', i + 1):
                return '?'
            return True
    raise GetoptError('option -%s not recognized' % opt, opt)
