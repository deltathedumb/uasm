# The Python frontend

`frontends/python` compiles Python. Programs that compile run the same under
CPython — that is the test suite's actual definition, and CPython is the oracle
for every one of them.

There are TWO PATHS, and which one a function takes is decided by its
annotations:

* **Statically typed.** A function whose every parameter is annotated with
  `int`, `float`, `bool` or `None` keeps machine representations — an `int` is
  a 64-bit register, a `float` an xmm register — and nothing is allocated.
* **Dynamic.** The module's top-level statements, any function with an
  unannotated or `object`-annotated parameter. Every value is a runtime object
  carrying its own type, and every operation is a call into the object runtime
  (`link/objects.py`). This is the path ordinary Python takes.

The boundary between them is one place: a dynamic function calling a statically
typed one unwraps each argument to the type it declared and wraps the result
back. Everywhere else a value is one or the other for its whole life, which is
what stops a value's representation depending on the slot it was stored in.

## What runs

The module's top-level statements, exactly as in Python — a file is a script,
and running it runs what is written at the top level.

```python
print(1 + 2)        # a whole program
```

A module whose top level is only definitions falls back to the convention this
frontend started with: `main()` is the entry point, and its return value is the
process exit code. Both shapes compile; the second is the reason every existing
`def main() -> int:` program still works.

## Types

On the STATIC path: `int` (64-bit), `float` (double), `bool`, `None`, with
every parameter, return and declaration annotated.

```python
def area(w: float, h: float) -> float:
    return w * h

def main() -> int:
    print(area(3.0, 4.0))
    return 0
```

`bool` widens to `int` and `int` to `float`, as Python's numeric tower does.
**Nothing narrows implicitly** — losing precision silently is how a program
computes the wrong answer without ever failing:

```python
n: int = 1.5        # error[E0060]: narrowing is never implicit
n: int = int(1.5)   # fine
```

## Where Python and the machine disagree

These are the cases the frontend pays for so no backend has to. Each is
lowered to several instructions, once, here.

| | Python | the machine | 
|---|---|---|
| `//` | floors toward −∞ | truncates toward zero |
| `%` | takes the divisor's sign | takes the dividend's |
| `and`/`or` | yield an **operand**, and short-circuit | — |
| `a < b < c` | evaluates `b` once | — |

```python
-7 // 2    # -4, not -3
-7 % 2     #  1, not -1
-7.5 % 2.0 #  0.5, not -1.5
7.5 // 2.0 #  3.0, not 3.75
```

On the DYNAMIC path: `int`, `float`, `bool`, `complex`, `None`, `str`,
`bytes`, `list`, `tuple`, `dict`, `set`, `frozenset`, exceptions, and
instances of user classes.

`complex` joins the numeric tower for arithmetic and equality and stays out of
it for ORDERING: `1j < 2j` is a `TypeError`, which is the rule that keeps it
from being a third float.
Integers are arbitrary precision. `type(x).__name__` answers correctly for all
of them.

## Statements

`if`/`elif`/`else`, `while`, `for`, `break`, `continue`, `return`, `pass`,
assignment, annotated assignment, augmented assignment. On the dynamic path
also `raise`, `try`/`except`/`else`/`finally`, `assert`, `global`, the `else`
clause of a loop, tuple-unpacking assignment, `for a, b in pairs:`,
subscript assignment, `del`, `with`, and the walrus operator.

`with` runs `__exit__` on every path out -- falling off the end, an exception,
or a `return` -- and a true return from it SWALLOWS the exception rather than
merely observing it. `with a as x, b as y:` is the nested form, so `b`'s
`__exit__` runs before `a`'s.

Unpacking targets nest and may be starred: `a, *rest = xs`, `*init, last = xs`,
`x, *mid, y = xs`, `first, (second, third) = pair`. A starred target is always
bound to a LIST, whatever the source was.

Module-level names are real module storage, so a function reads them and
`global x` writes through. A name a function assigns is local for that whole
function, as in Python, and reading a module name before its assignment has
run is a `NameError` rather than a crash.

`for` iterates a range on either path, and any sequence, string or dict on the
dynamic one. Iteration is BY INDEX -- there is no iterator protocol, so the
length is read once before the first pass and a body that appends to the
sequence it is walking will differ from CPython.

`range` in a `for` header takes 1–3 arguments and lowers to a counter loop with
no allocation. **A three-argument `range` needs a literal step**,
because the step's sign decides whether the loop test is `<` or `>`:

```python
for i in range(5, 0, -1):   # fine
    print(i)
for i in range(5, 0, s):    # error[E0028]: step must be a literal
    print(i)
```

Accepting a runtime step would mean picking one comparison and being silently
wrong about the other — every descending loop running zero times and reporting
success.

## A name must be assigned on every path

```python
if flag > 0:
    later: int = 42
print(later)        # error[E0032]: may be used before it is assigned
```

Python raises `UnboundLocalError` for this at runtime; saying it at compile
time is strictly better, and it is the same rule. A branch that cannot fall
through does not dilute it, so this is fine:

```python
if c:
    x: int = 1
else:
    return 0
print(x)            # every path here assigned x
```

A loop body may run zero times, so neither the loop variable nor anything the
body assigns counts as assigned afterwards — `for i in range(0): pass` then
`print(i)` is an error here and an `UnboundLocalError` in Python.

## Expressions

Arithmetic, comparison and bitwise operators, `not`/`and`/`or`, conditional
expressions, calls, `int()`/`float()`/`bool()`.

Calls take KEYWORD arguments, `*args` and `**kwargs`, and a `def` may declare
`*rest` and `**kw`. A keyword is resolved against the callee's parameter names,
which the function value carries -- a call through a value reaches a `def` the
call site never saw, so the names have to travel with it.

Decorators work on `def` and on `class`, and stack bottom-up. A module-level
`def`'s decorators run at program start rather than where the `def` is written,
because analysis lifts a module-level `def` out of the entry's body; a
decorator naming something the module body ASSIGNS is the case this gets wrong,
and it gets it wrong loudly, with a `NameError`.

`__name__` is `"__main__"`, so the guard at the bottom of a script takes its
branch.

`print` takes any number of arguments, separates them with a space and ends
with a newline, as Python does — `print()` is an empty line.

Augmented assignment (`+=`, `//=`, `**=`, `<<=`, …) is checked as the
operation it stands for, so every rule below applies to it too.

`**` needs a **non-negative integer literal** exponent:

```python
x ** 8      # three multiplications, expanded at compile time
x ** n      # error[E0043]: needs a literal integer exponent
x ** -1     # error[E0044]: negative exponents are float-valued
```

Python's `**` is not one operation — `2 ** 10` is an int, `2 ** -1` is the
float `0.5`. A statically typed expression cannot be both, and a runtime
exponent would force a guess.

`int()`, `float()` and `bool()` are conversions, not calls — they lower to a
coercion or to nothing. `bool(x)` is `x != 0`; a narrowing cast would make
`bool(2)` false. A program that defines its own function with one of these
names shadows the conversion.

## Not implemented

`async`/`await`, keyword-only and positional-only parameters, `match`,
metaclasses, multiple inheritance, and `@`.

`yield` IS supported. A generator is compiled as two functions and an object:
the constructor keeps the name the `def` binds and allocates the generator
without running any of the body, and a step function holds the body, re-entered
once per `next` and opening with a dispatch on the saved state. Every local
lives in the object rather than in a register, because a register does not
survive the return a `yield` compiles to.

`next(g)` and `g.send(v)` are LAZY -- one resume per call. A `for` over a
generator is not: it walks by index, an index walk needs a length, and asking
for one drains the generator into a list. So a `for` over an infinite generator
never starts rather than never ending, and `map`/`filter`/generator
expressions are eager for the same reason.

`import` reaches a TABLE of built-in modules, not a search path: there is no
file system at run time and no second compilation unit, so a module is built
at the `import` statement out of constants and wrappers around runtime entry
points. `math` and `__future__` are what it holds. All four spellings work --
`import m`, `import m as n`, `from m import x`, `from m import x as y` -- and
`import math as m` twice gives `math is m`, because a module is built once per
program.

Format specs ARE supported, in all three spellings that share the
mini-language: `f"{v:>8.2f}"`, `"{:>8}".format(v)` and `format(v, ">8")`.

`map` and `filter` are EAGER: the calls all happen when they are made, and what
comes back is a cursor over the results. Laziness needs a resumable frame,
which is the same thing `yield` needs.

`lambda` IS supported. It is registered as a nested function whose body is one
`return`, so it gets closures, cells and default arguments from exactly the
machinery a `def` uses rather than from a second one.

`sorted`, `min` and `max` take `key=`, and `sorted` also `reverse=`. The key
is computed once per element, before any comparison, and `reverse=True` stays
STABLE -- reversing the result instead would reverse equal elements too.

`f(*xs)` IS supported. The argument count is a value rather than a constant,
so such a call goes through the callee as a VALUE and binds against its own
signature at run time -- which means a wrong count is reported when the call
happens, exactly as CPython reports it for this shape.

Classes ARE supported on the dynamic path, with single inheritance, the
`__init__`/`__str__`/`__repr__`/`__eq__`/`__len__`/`__getitem__` family of
dunders, and the no-argument `super()`. Not the full MRO -- C3 linearisation
is real work and almost nothing needs it. Nested `def` and closures are
supported; `nonlocal` is not, so a closure reads an enclosing variable but
cannot rebind one.

Integers are arbitrary precision.

Comprehensions, f-strings, default arguments, keyword arguments and `*args`
ARE supported on the dynamic path. Default values are evaluated ONCE, where
the `def` runs, so `def f(xs=[])` shares one list across calls — which is
Python's behaviour and the thing an implementation is most tempted to
"fix".

A generator expression is built eagerly as a list, observable if you ask its
type or iterate it twice. `enumerate`, `zip`, `reversed` and `range` used as a
value are lists for the same reason. Iteration is by index and reads the
length once, so a loop that appends to the sequence it is walking differs from
CPython.

Each produces a diagnostic with a code. **The compiler never raises on Python
it does not support** — a result or a diagnostic, never a traceback. That is
enforced by a test that feeds 42 unsupported constructs at the whole pipeline.

## An operand keeps its own type

Python's `and`, `or` and `x if c else y` do not compute a value — they YIELD
ONE OF THEIR OPERANDS. So the result has the type of whichever one was picked,
and on the static path, where every expression has one type, that can only be
right when the operands agree. Where they do not, it is refused:

```python
print(n and x)          # error[E0065]: `and` yields one of its operands
```

Converted, the operands agree and there is one answer:

```python
print(float(n) and x)   # 3.0
```

`n` is an `int` and `x` a `float` throughout this document. Both lines are
STATIC-PATH code: at module level everything is dynamic, every value carries
its own type, and `n and x` is simply Python. The rule here is the static
path's and nowhere else's.

**This used to widen instead**, and the entry here used to defend it: `a and
f` answered `0.0` where CPython says `0`, on the grounds that the values are
equal and the alternative is a runtime tag. That was a false choice. The
alternative is a diagnostic, which costs nothing at run time — and the old
behaviour meant the same source printed `0` at module level and `0.0` inside
an annotated function, which is the static path having its own semantics
rather than being an optimisation of the dynamic one.

Nothing caught it because the conformance corpus is scripts, and a script's
top level is dynamic: the divergence lived only in the half the corpus never
measures.

### In a condition the operands may still differ

Only the TRUTH of a test is observed, and `0` and `0.0` answer that
identically — so nothing is refused where the value cannot be seen:

```python
if n and x:             # fine: only truthiness is read
    print(1)
while n and x:          # fine
    n = 0
if not (n or x):        # fine
    print(2)
```

Assign it, and the value is observed again:

```python
picked = n and x        # error[E0065]: the value is used
```

The position travels down through `and`, `or`, `not` and a conditional's
arms, because truthiness is preserved through all four.

## Floats print exactly as CPython prints them

`print` of a float is the shortest decimal string that reads back as the same
double, with Python's choice of fixed or exponent notation — not C's `%f`, and
not `%g` either, whose notation threshold moves with the digit count and
disagrees for values like `12345678901234567.0`.

This is implemented three times, because there are three runtimes and they must
agree: `py_repr_double` in `link/runtime.py` (hosted, via snprintf/strtod),
Dragon4 in exact integer arithmetic in `link/baremetal.py` (freestanding, no
libc), and `repr` itself in `Interpreter._host`. Each is checked against
`repr()` over hundreds of thousands of doubles.

`x ** n` for an integral `n` does not call libm's `pow`: mingw's is a ulp off
where CPython's is correctly rounded, so `py_pow_int` squares in double-double
and rounds once.

## Diagnostics

Every error has a code, a position, and usually a suggestion:

```
error[E0044]: `**` with a negative exponent is float-valued
 --> prog.py:2:17
  |
2 |     return 2 ** -1
  |                 ^^ -1
  |
  = help: write `1.0 / (2 ** 1)`
```

One unknown name produces one diagnostic, not one per use: an unresolved type
poisons everything derived from it and every operation on a poisoned type is
silent.

The full set the frontend can emit:

| code | what it means |
| --- | --- |
| `E0000` | the source is not valid Python; CPython's own `SyntaxError` text |
| `E0003` | nothing to run: only definitions, and no `main` to call |
| `E0004` | a duplicate parameter name |
| `E0005` | `*args`, `**kwargs`, or keyword-only/positional-only parameters |
| `E0006` | a default argument |
| `E0007` | a decorator |
| `E0008` | `main` takes no parameters (when it is the entry point) |
| `E0009` | `main` must return `int` (when it is the entry point) |
| `E0010` | a missing type annotation |
| `E0011` | a type this frontend does not have |
| `E0012` | a literal that does not fit the machine type it is given |
| `E0013` | an operator applied to two different machine widths |
| `E0014` | `/` applied to an integer width |
| `E0015` | `**` applied to a machine type |
| `E0016` | values that must share one type and do not |
| `E0017` | a memory intrinsic's first argument, which is a type, is not one |
| `E0018` | `alloca()` without a positive literal size |
| `E0019` | a memory intrinsic argument of the wrong type |
| `E0020` | an assignment other than `name = value` |
| `E0021` | a loop target that is not a plain name |
| `E0022` | an unsupported statement |
| `E0023` | a `for` over something other than `range(...)` |
| `E0024` | `range()` with the wrong number of arguments |
| `E0025` | `for ... else` |
| `E0026` | `while ... else` |
| `E0027` | `break` or `continue` outside a loop |
| `E0028` | a `range()` step that is not a literal |
| `E0029` | a `range()` step of zero |
| `E0030` | a name redeclared with another type |
| `E0031` | an undefined name |
| `E0032` | a name used before it is assigned on every path |
| `E0033` | a `ptr` made from something that is not an integer address |
| `E0034` | a `ptr` converted to something that is not 64 bits wide |
| `E0035` | `reserve()` without a literal name |
| `E0036` | one `reserve()` name given two different sizes |
| `E0037` | a top-level statement in a runtime module, which never runs |
| `E0040` | an unsupported expression |
| `E0041` | an operator applied to non-numbers |
| `E0042` | a bitwise operator applied to a float |
| `E0043` | `**` without a literal exponent |
| `E0044` | `**` with a negative exponent |
| `E0045` | an operator this frontend does not lower |
| `E0050` | a call to something other than a name |
| `E0051` | a keyword argument |
| `E0052` | a call to an unknown function |
| `E0053` | a call with the wrong number of arguments |
| `E0054` | a conversion or memory intrinsic with the wrong number of arguments |
| `E0055` | a conversion of something not numeric |
| `E0056` | a function used as a value |
| `E0060` | a type mismatch |
| `E0061` | an exception constructor given the wrong number of arguments |
| `E0062` | an unknown exception type in an `except` clause |
| `E0064` | an async comprehension |
| `E0065` | `and`/`or`/`if-else` yielding operands of different types, where the value is used |
| `E0070` | `rodata()` given anything but a non-empty bytes literal |
| `E0067` | `nonlocal` |
| `E0068` | an unexpected keyword argument |
| `E0069` | two values for one argument |
| `E0073` | a metaclass or a class keyword |
| `E0074` | a nested `def` or `class` in a statically typed function |
| `E0076` | a base class not defined in this module |
| `E0078` | `super()` with arguments |
| `E0079` | `super()` outside a method |
| `E0080` | a literal of a type with no runtime kind |
| `E0081` | `del` of something other than a name or a subscript |
| `E0082` | `with ... as` bound to something other than a plain name |
| `E0083` | `import` of a module that is not built in |
| `E0084` | `from m import x` where `m` has no member `x` |
| `E0085` | a statically typed function used as a value |
| `E0086` | `await` outside an `async def` |
| `E0087` | `async for` outside an `async def` |
| `E0088` | `match` in a statically typed function |
| `E0089` | `async with` outside an `async def` |
| `E0090` | `**` in a class statement |

| `E0092` | a method call on a value that is not a Java object |
| `E0093` | a Java class the class path no longer describes |
| `E0094` | a Java class has no such instance method |
| `E0095` | calling a module rather than something in it |
| `E0096` | a namespace member that is not a Java type |
| `E0097` | constructing an abstract Java class or an interface |
| `E0098` | a Java class with no public constructor |
| `E0099` | a Java class has no such static method |
| `E0100` | an attribute chain too long to be a call |
| `E0101` | wrong number of arguments to a Java method |
| `E0102` | no overload of a Java method takes these argument types |
| `E0103` | something other than a method in a class over a Java type |
| `E0104` | a decorator on a class over a Java type |
| `E0105` | a class over a Java type with a backend that has no such notion |
| `E0106` | an override that does not match the method it overrides, or an abstract method left undefined |
| `E0107` | an `async def` in a class over a Java type |
| `E0108` | a method of such a class whose first parameter is missing or annotated |
| `E0109` | two methods of one such class with the same name |
| `E0110` | two classes over a Java type with the same name |
| `E0121` | a `restype` that is not a `ctypes` scalar type |
| `E0122` | an `argtypes` that is not a list |
| `E0123` | an `argtypes` entry that is not a `ctypes` scalar type |
| `E0124` | a `ctypes` library named by something other than a literal |
| `E0125` | a native call whose function has no `argtypes` |
| `E0126` | a native call with the wrong number of arguments for its `argtypes` |
| `E0127` | a `ctypes` library used as a value rather than called through |
| `E0128` | a `ctypes` name this frontend does not have |
| `E0129` | an import naming a compiled extension module, which has no source to compile |
| `E0130` | `from <native library> import ...`; a declared library is imported whole |
| `E0131` | a declared native library named after a module this compiler already has |
| `E0132` | a relative `import` whose level climbs past its package, or that has no package at all |

Each applies wherever the construct appears, including inside an augmented
assignment -- `x **= n` reports `E0043` exactly as `x = x ** n` does.

### The frontend's own flags

Two codes are about what was passed on the COMMAND LINE rather than about the
program, and are listed here because the flags are this frontend's:
`--host-python` and `--native-library` are declared by `PythonFrontend.options`
and handed to it by the driver, which no longer knows what either means.

| code | what it means |
| --- | --- |
| `E9108` | `--host-python` named an interpreter that would not run |
| `E9109` | `--native-library` named a declaration file that would not parse |

## Warnings

A warning is a program that COMPILES and is worth a sentence about what it
will do.

| code | what it means |
| --- | --- |
| `W0053` | an operation that is provably a `TypeError`, raised at run time |
| `W0091` | `compile()`, `eval()` or `exec()` in a compiled program |

`W0091` USED TO BE AN ERROR -- a refusal. The three are now supported: the
parser, the validator and the code object are bundled Python spliced into the
program that names one, and `eval`/`exec` walk the tree they build. What the
warning says is what that costs -- `compile()` answers whether source is valid
Python through a parser rather than through the compiler that built the
binary, and `eval()` and `exec()` INTERPRET what they are given, which is far
slower than the native code around them.
