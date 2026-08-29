"""
Tests the exec() guard in safe_exec.py: confirms it actually blocks the classes of
thing it claims to (file access, process/network modules, eval/exec, dunder-based
sandbox-escape gadgets) and does NOT block normal pandas/numpy/sklearn analysis code
that never touches any of that -- a blocklist that also breaks legitimate code isn't
useful.
"""
import pandas as pd

from safe_exec import safe_exec, check_code_safety, UnsafeCodeError
from exec_namespace import build_exec_namespace

DF = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def _blocked(code):
    try:
        check_code_safety(code)
        return False
    except UnsafeCodeError:
        return True


def test_blocks_file_access():
    assert _blocked("open('/etc/passwd').read()")


def test_blocks_os_import():
    assert _blocked("import os\nos.system('echo hi')")


def test_blocks_os_via_importfrom():
    assert _blocked("from os import system\nsystem('echo hi')")


def test_blocks_subprocess():
    assert _blocked("import subprocess\nsubprocess.run(['ls'])")


def test_blocks_eval_exec():
    assert _blocked("eval('1+1')")
    assert _blocked("exec('x=1')")


def test_blocks_dunder_globals_escape():
    # classic sandbox-escape gadget: walk from an innocuous object back to
    # __globals__ / __subclasses__ to reach something dangerous indirectly
    assert _blocked("().__class__.__bases__[0].__subclasses__()")
    assert _blocked("(lambda: 0).__globals__")


def test_blocks_socket_network():
    assert _blocked("import socket\nsocket.socket()")


def test_allows_normal_pandas_code():
    assert not _blocked("st.write(df.describe())")
    assert not _blocked("model_result = df['a'].mean()\nst.write(model_result)")


def test_allows_label_encoder_without_import():
    # Regression check: LabelEncoder is now pre-provided in exec_namespace.py, so
    # code using it directly (no import statement) must pass the scan -- found via
    # generalization testing on a string-labeled classification target, which
    # previously had no way to encode the target without a blocked import.
    assert not _blocked("y = LabelEncoder().fit_transform(df['a'])\nst.write(y)")


def test_allows_dummy_baseline_models_without_import():
    # DummyRegressor/DummyClassifier are pre-provided so predictive-modeling code
    # can compare a real model against a naive baseline without an import statement.
    assert not _blocked("m = DummyRegressor(strategy='mean').fit(df[['a']], df['b'])\nst.write(m.predict(df[['a']]))")
    assert not _blocked("m = DummyClassifier(strategy='most_frequent').fit(df[['a']], df['b'])\nst.write(m.predict(df[['a']]))")


def test_allows_small_stdlib_allowlist():
    assert not _blocked("import re\nst.write(re.findall(r'\\d+', 'a1b2'))")


def test_safe_exec_actually_runs_allowed_code():
    ns = {"df": DF, "st": type("S", (), {"items": [], "write": lambda self, x: self.items.append(x)})()}
    safe_exec("st.write(df['a'].sum())", ns)
    assert ns["st"].items == [6]


def test_safe_exec_raises_before_running_blocked_code():
    ns = {"df": DF}
    try:
        safe_exec("import os\nos.system('id')", ns)
        assert False, "should have raised UnsafeCodeError"
    except UnsafeCodeError:
        pass


def test_comprehension_referencing_namespace_names_actually_executes():
    # Regression test for a real live bug (not hypothetical): safe_exec() used
    # to call exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars) -- TWO
    # separate dicts for globals and locals. That makes Python treat the
    # executed code like a function/class body: names only present in
    # local_vars (df, DummyRegressor, RandomForestRegressor, ...) raise
    # NameError the moment a dict/list comprehension or nested function
    # references them, because comprehensions resolve free variables via the
    # GLOBALS dict, not locals. A plain top-level statement worked fine, which
    # is exactly why this passed every prior test and safety-scan check, and
    # only broke on a live Groq run once the LLM generated a dict
    # comprehension building multiple models (a completely normal pattern).
    # Uses the REAL production exec_namespace, not a hand-rolled minimal one,
    # so this catches the same class of gap the earlier scan-only tests missed
    # (they called check_code_safety() directly, never safe_exec() end to end).
    ns = build_exec_namespace(DF)
    code = (
        "models = {name: cls() for name, cls in "
        "[('linear', LinearRegression), ('baseline', DummyRegressor)]}\n"
        "for name, m in models.items():\n"
        "    m.fit(df[['a']], df['b'])\n"
        "def score(m):\n"
        "    return m.predict(df[['a']])\n"
        "st.write(score(models['baseline']))\n"
    )
    safe_exec(code, ns)  # must not raise NameError
    assert len(ns["st"].items) == 1


if __name__ == "__main__":
    test_blocks_file_access()
    test_blocks_os_import()
    test_blocks_os_via_importfrom()
    test_blocks_subprocess()
    test_blocks_eval_exec()
    test_blocks_dunder_globals_escape()
    test_blocks_socket_network()
    test_allows_normal_pandas_code()
    test_allows_label_encoder_without_import()
    test_allows_dummy_baseline_models_without_import()
    test_allows_small_stdlib_allowlist()
    test_safe_exec_actually_runs_allowed_code()
    test_safe_exec_raises_before_running_blocked_code()
    test_comprehension_referencing_namespace_names_actually_executes()
    print("All safe_exec tests passed (blocks file/process/network/eval/dunder-escape, allows normal analysis code).")
