"""
safe_exec.py -- a real (not just documented) guard around executing LLM-generated
Python code.

Honest scope statement, because it's easy to overclaim security work: this is NOT
a sandbox. It does not run code in a separate process, container, or VM, and a
sufficiently creative attacker with time to hunt for a gap in the blocklist could
likely still find one -- blocklists are inherently incomplete. What this DOES
reliably stop is the class of thing an LLM might plausibly generate by accident or
under a bad prompt: reading/writing arbitrary files, importing os/subprocess/socket
and touching the filesystem or network, calling eval/exec/compile on further
attacker-controlled strings, or reaching for classic sandbox-escape gadgets via
dunder attributes (__globals__, __subclasses__, __mro__, etc.).

For a single-user, local, course-project tool this is a reasonable and honestly-
described mitigation. It is explicitly NOT what you'd want for a multi-tenant or
internet-facing deployment -- that needs real process isolation (a container,
gVisor, or similar), which is out of scope for what could be built in the time
available here. See PROJECT_LOG.md's Security & Limitations section.
"""

import ast


# Modules the generated code is allowed to import directly, on the rare occasion
# a handler's code reaches for something outside the pre-provided namespace
# (exec_namespace.py already provides pd/np/px/go/sm/smf/sklearn pieces/IV2SLS,
# so imports should be rare -- this is a narrow allowance, not a general one).
ALLOWED_IMPORTS = {"re", "math", "itertools", "collections", "statistics", "datetime", "json"}

# Names that must never be reachable inside generated code, whether as a bare
# name, an attribute access, or an import target.
BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "pickle", "marshal", "multiprocessing", "threading", "signal",
    "resource", "platform", "getpass", "requests", "urllib", "http", "io",
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "breakpoint", "exit", "quit", "help",
    "memoryview", "vars", "__builtins__", "__loader__", "__spec__",
}

# Dunder attributes that are classic sandbox-escape gadgets (walking from any
# object back to its class, then to its base classes' subclasses, to reach
# something like os or subprocess indirectly).
BLOCKED_ATTRS = {
    "__globals__", "__code__", "__subclasses__", "__mro__", "__base__",
    "__bases__", "__class__", "__dict__", "__getattribute__", "__reduce__",
    "__reduce_ex__", "__loader__", "__spec__", "__import__", "__builtins__",
}


class UnsafeCodeError(Exception):
    """Raised when generated code fails the pre-execution safety scan."""


def check_code_safety(code: str) -> None:
    """Parses the code and rejects it if it references anything on the
    blocklists above. Raises UnsafeCodeError with a specific reason if so;
    returns None (silently) if the code passes. Call this BEFORE exec()."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Code does not parse as valid Python: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(
                        f"Import of '{alias.name}' is not allowed. Only {sorted(ALLOWED_IMPORTS)} "
                        "may be imported directly -- everything else (pd, np, px, go, sm, smf, "
                        "sklearn pieces) is already provided in the execution namespace."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"Import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Use of '{node.id}' is not allowed in generated code.")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                raise UnsafeCodeError(f"Access to '.{node.attr}' is not allowed in generated code.")


# A deliberately small allowlist of builtins -- enough for typical pandas/numpy/
# sklearn/statsmodels/plotly analysis code, without exposing filesystem, process,
# or introspection primitives. Handed to exec() as the ONLY __builtins__ generated
# code sees (the real, full __builtins__ dict is never passed in).
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "dir", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "hash", "id",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "pow", "print", "property", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip", "True", "False", "None", "NotImplemented",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "StopIteration", "ZeroDivisionError", "RuntimeError",
    "ArithmeticError", "OverflowError", "NotImplementedError", "AssertionError",
]

import builtins as _builtins_module

SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _SAFE_BUILTIN_NAMES}


def safe_exec(code: str, local_vars: dict):
    """Runs check_code_safety() then exec()'s the code with a restricted
    __builtins__ dict instead of the real one. Raises UnsafeCodeError before
    ever calling exec() if the static scan finds anything on the blocklists.

    IMPORTANT -- exec() is called with a SINGLE dict serving as both globals
    and locals, not two separate dicts. Found live (real Groq run, not a
    hypothetical): passing local_vars as a separate `locals` argument while
    globals only has __builtins__ makes Python treat the executed code like a
    function/class body for name resolution. A top-level `x = DummyRegressor()`
    still works, but the instant the generated code uses a list/dict/set
    comprehension or defines a nested function that references df, pd, or any
    sklearn class -- extremely common in real LLM-generated modeling code --
    that inner scope resolves names via the GLOBALS dict only, not locals, and
    raises NameError even though the name is right there in local_vars. Using
    one shared dict for both makes the executed code behave like real
    module-level code, where comprehensions and nested functions can see
    everything in scope, matching what generated code is actually written to
    expect. See PROJECT_LOG.md for the live traceback that caught this."""
    check_code_safety(code)
    local_vars["__builtins__"] = SAFE_BUILTINS
    exec(code, local_vars)
