"""
Execute short, sandboxed Python snippets — handy as a calculator or for
quick arithmetic, unit conversions, and other simple computations that are
easier to express as code than to reason about in natural language.

The code runs with a restricted set of builtins (no file, network, process,
or import access) and a short wall-clock timeout, so it is safe to use for
throwaway calculations but is not a general-purpose code execution sandbox.
"""

import ast
import builtins
import cmath
import contextlib
import io
import math
import signal
import statistics


# A deliberately small allowlist of builtins: enough for arithmetic, simple
# data wrangling, and formatting, but nothing that touches the filesystem,
# network, process, or import system (no open, exec, eval, __import__, etc).
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bin", "bool", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hex", "int", "len", "list", "map", "max", "min", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
]
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, name)
}

# Modules that are genuinely useful for calculations are exposed directly by
# name instead of allowing `import`, which stays disabled.
_SAFE_GLOBALS = {
    "__builtins__": _SAFE_BUILTINS,
    "math": math,
    "cmath": cmath,
    "statistics": statistics,
}

_TIMEOUT_SECONDS = 5


class _TimeoutError(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds):
    """Best-effort wall-clock timeout so a runaway snippet (e.g. `while True: pass`)
    can't hang the process. Only enforced on platforms with SIGALRM (Unix)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _TimeoutError(f"Execution timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run(code: str):
    """
    Executes `code` and returns (value, stdout_text).

    Mimics a REPL: if the snippet's last statement is a bare expression, its
    value is captured and returned in addition to anything printed.
    """
    tree = ast.parse(code, mode="exec")

    trailing_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing_expr = ast.Expression(tree.body.pop().value)

    local_scope = {}
    stdout = io.StringIO()

    with _time_limit(_TIMEOUT_SECONDS), contextlib.redirect_stdout(stdout):
        if tree.body:
            exec(compile(tree, "<calculator>", "exec"), _SAFE_GLOBALS, local_scope)
        value = None
        if trailing_expr is not None:
            value = eval(compile(trailing_expr, "<calculator>", "eval"), _SAFE_GLOBALS, local_scope)

    return value, stdout.getvalue()


def python_calculator(code: str):
    """
    Runs a short Python snippet in a restricted sandbox and returns its result.

    'code' is the Python source to execute (e.g. "2 + 2" or a few lines that
    end with an expression). Useful as a calculator for arithmetic, math
    functions, unit conversions, and other simple computations.
    """
    code = (code or "").strip()
    if not code:
        return "Error: no code provided."

    try:
        value, printed = _run(code)
    except SyntaxError as e:
        return f"Syntax error: {e.msg} (line {e.lineno}, offset {e.offset})"
    except _TimeoutError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    printed = printed.rstrip("\n")
    if value is None:
        return printed if printed else "(no output; the snippet produced no result and printed nothing)"
    if printed:
        return f"{printed}\n{value!r}"
    return repr(value)


python_calculator_description = {
    "type": "function",
    "function": {
        "name": "python_calculator",
        "description": (
            "Executes a short Python code snippet in a restricted sandbox and returns its "
            "result. Useful as a calculator or for simple Python calculations: arithmetic, "
            "math functions (via the pre-imported 'math', 'cmath', and 'statistics' modules), "
            "unit conversions, and quick data manipulation. File, network, process, and import "
            "access are disabled, and execution is time-limited, so this is only suitable for "
            "small, self-contained computations, not general code execution."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source code to execute, e.g. '2 + 2' or 'math.sqrt(2)'. If the "
                        "last line is a bare expression, its value is returned as the result; "
                        "anything printed with print() is also included in the output."
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
