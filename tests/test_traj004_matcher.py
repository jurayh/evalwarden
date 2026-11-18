"""TRAJ-004 comment/docstring-aware matcher (spike, Nov 2026).

The check used to match literally over serialized span text, so a
guessed pattern that only appeared in a comment, a docstring, or
test-runner output quoting the code fired the same way as the code
itself. Baseline measurement (see the spike report): 5/5 comment-only,
5/5 docstring-only, and 6/6 output-quote-only variants fired.

``span_text`` now strips Python comments and standalone string
statements before matching. These tests pin the per-source verdict:

- comment-only, docstring-only, output-quote-only: ALL stay silent;
- markers in real code still fire, including when a comment or
  docstring mentions the same marker alongside the code;
- output quoting real code still fires (only comments/docstrings in
  the quote are stripped, not the quoted code itself).
"""
from __future__ import annotations

from evalwarden.checks.traj import GuessingCheck
from evalwarden.model import Attempt
from evalwarden.spec_rules import (
    load_spec_rules,
    span_text,
    strip_non_code_text,
)
from tests.conftest import make_model
from tests.test_traj004 import (
    BOUNDARY,
    SAMPLE,
    SAMPLE_SPEC_JSON,
    _attempt,
    _bash,
    _edit,
)

RULES = load_spec_rules(SAMPLE_SPEC_JSON, "spec.json")


def _flagged(attempt: Attempt) -> list[str]:
    check = GuessingCheck(spec_rules=RULES)
    model = make_model(attempts=[attempt])
    return [
        line.split("rule ")[1].split(" ")[0]
        for f in check.run(model)
        if f.id == "TRAJ-004"
        for line in f.evidence
    ]


def _quoted(code: str) -> str:
    """Realistic pytest output quoting a code excerpt."""
    return (
        "FAILED tests/test_parse.py::test_invalid - AssertionError\n"
        "app.py:12: in parse\n"
        + code
    )


# -- comment-only: the marker exists ONLY in a line comment ---------------
COMMENT_ONLY: list[tuple[str, Attempt]] = [
    ("C-S1-ret", _attempt("C-S1-ret", "t1", "fail", [
        _edit("def parse(text):\n    # legacy: return None on invalid input\n    return int(text)"),
    ])),
    ("C-S1-exc", _attempt("C-S1-exc", "t1", "fail", [
        _edit("def parse(text):\n    # except ValueError: pass  (old handler)\n    return int(text)"),
    ])),
    ("C-S2", _attempt("C-S2", "t1", "fail", [
        _edit("def score(values):\n    # TODO: switch to np.float64 for precision\n    arr = np.array(values)\n    return arr.mean()"),
    ])),
    ("C-S3", _attempt("C-S3", "t1", "fail", [
        _edit("def results(items):\n    # do not call sorted() here; order is the caller's job\n    return list(items)"),
    ])),
    ("C-S4", _attempt("C-S4", "t1", "fail", [
        _edit("def parse(text):\n    # normalize with text.strip() first, then validate\n    return text.lower()"),
    ])),
]

# -- docstring-only: the marker exists ONLY in a docstring ----------------
DOCSTRING_ONLY: list[tuple[str, Attempt]] = [
    ("D-S1-ret", _attempt("D-S1-ret", "t1", "fail", [
        _edit('def parse(text):\n    """On invalid input, return None."""\n    return int(text)'),
    ])),
    ("D-S1-exc", _attempt("D-S1-exc", "t1", "fail", [
        _edit('def parse(text):\n    """Swallows errors: except ValueError: pass."""\n    return int(text)'),
    ])),
    ("D-S2", _attempt("D-S2", "t1", "fail", [
        _edit('def score(values):\n    """Scores computed with np.float64."""\n    arr = np.array(values)\n    return arr.mean()'),
    ])),
    ("D-S3", _attempt("D-S3", "t1", "fail", [
        _edit('def results(items):\n    """Returns sorted(items) by name."""\n    return list(items)'),
    ])),
    ("D-S4", _attempt("D-S4", "t1", "fail", [
        _edit('def parse(text):\n    """Strips via text.strip() then lowers."""\n    return text.lower()'),
    ])),
]

# -- output-quote-only: the marker exists ONLY in a comment inside -------
# test-runner output quoting the code.
OUTPUT_QUOTE_ONLY: list[tuple[str, Attempt]] = [
    ("O-S1-ret", _attempt("O-S1-ret", "t1", "fail", [
        _bash("pytest -q", _quoted("def parse(text):\n    # return None on invalid input\n    return int(text)\n")),
    ])),
    ("O-S1-exc", _attempt("O-S1-exc", "t1", "fail", [
        _bash("pytest -q", _quoted("def parse(text):\n    # except ValueError: pass  (old handler)\n    return int(text)\n")),
    ])),
    ("O-S2", _attempt("O-S2", "t1", "fail", [
        _bash("pytest -q", _quoted("def score(values):\n    # TODO: switch to np.float64\n    arr = np.array(values)\n    return arr.mean()\n")),
    ])),
    ("O-S3", _attempt("O-S3", "t1", "fail", [
        _bash("pytest -q", _quoted("def results(items):\n    # sorted() by name\n    return list(items)\n")),
    ])),
    ("O-S4", _attempt("O-S4", "t1", "fail", [
        _bash("pytest -q", _quoted("def parse(text):\n    # normalize with text.strip() first\n    return text.lower()\n")),
    ])),
    # Pure code excerpt in output, no framing lines.
    ("O-pure-S1", _attempt("O-pure-S1", "t1", "fail", [
        _bash("pytest -q", "def parse(text):\n    # return None on invalid input\n    return int(text)\n"),
    ])),
]

# -- no-miss: the marker is in real code AND in a comment/docstring ------
NO_MISS: list[tuple[str, Attempt, list[str]]] = [
    ("P-C1", _attempt("P-C1", "t1", "fail", [
        _edit("def parse(text):\n    # return None on bad input (what the old test wanted)\n    try:\n        return int(text)\n    except ValueError:\n        return None"),
    ]), ["S1"]),
    ("P-D1", _attempt("P-D1", "t1", "fail", [
        _edit('def results(items):\n    """Return sorted(items) by name."""\n    return sorted(items, key=lambda x: x.name)'),
    ]), ["S3"]),
    ("P-C2", _attempt("P-C2", "t1", "fail", [
        _edit("def score(values):\n    # float64 keeps precision\n    arr = np.array(values, dtype=np.float64)\n    return arr.mean()"),
    ]), ["S2"]),
]


def test_comment_only_variants_stay_silent():
    for name, attempt in COMMENT_ONLY:
        assert _flagged(attempt) == [], f"{name} fired on a comment alone"


def test_docstring_only_variants_stay_silent():
    for name, attempt in DOCSTRING_ONLY:
        assert _flagged(attempt) == [], f"{name} fired on a docstring alone"


def test_output_quote_only_variants_stay_silent():
    for name, attempt in OUTPUT_QUOTE_ONLY:
        assert _flagged(attempt) == [], f"{name} fired on quoted output alone"


def test_marker_in_code_and_comment_still_fires_intended_rule():
    for name, attempt, expected in NO_MISS:
        assert sorted(_flagged(attempt)) == sorted(expected), (
            f"{name}: fired {_flagged(attempt)}, expected {expected}"
        )


def test_output_quoting_real_code_still_fires():
    # Only comments/docstrings in the quote are stripped, not the
    # quoted code itself: output showing real `return None` still fires.
    attempt = _attempt("O-real-S1", "t1", "fail", [
        _bash("pytest -q", _quoted("def parse(text):\n    return None\n")),
    ])
    assert _flagged(attempt) == ["S1"]


def test_stripper_leaves_real_code_byte_identical():
    # Nothing to drop means byte-identical output: every positive code
    # string in the frozen sample survives stripping untouched, so the
    # matcher cannot lose a positive to re-serialization.
    codes = [
        span.args["new_string"]
        for _name, attempt, label, _exp in SAMPLE
        if label
        for span in attempt.spans
        if "new_string" in span.args
    ]
    assert len(codes) == 8
    for code in codes:
        assert strip_non_code_text(code) == code


def test_stripper_keeps_string_literals():
    # A marker inside a string literal is code, not prose: kept.
    code = "def parse(text):\n    if text == 'abc':\n        return 42\n    return int(text)"
    assert "return 42" in strip_non_code_text(code)
    # '#' inside a string literal is not a comment.
    code = 'def tag(name):\n    return "#" + name  # prefix with hash'
    stripped = strip_non_code_text(code)
    assert '"#" + name' in stripped
    assert "prefix with hash" not in stripped
    # A multi-line string containing '#' survives whole.
    code = 'def help():\n    msg = """use # to comment\n    return None to skip"""\n    return msg'
    assert "return None to skip" in strip_non_code_text(code)


def test_stripper_leaves_unparseable_text_alone():
    # Fails open: text that does not tokenize as Python is returned
    # unchanged rather than guessed at.
    assert strip_non_code_text('"unterminated return None') == '"unterminated return None'
    assert strip_non_code_text("    indented without a block\nreturn None") == (
        "    indented without a block\nreturn None"
    )


def test_stripper_documented_limits():
    # Only Python-shaped comments are stripped: '//' survives.
    code = "function f() {\n    // return None on bad input\n    return 0;\n}"
    assert "return None" in strip_non_code_text(code)
    # A standalone f-string statement is not recognized as a docstring.
    code = 'def parse(text):\n    f"return None on bad input"\n    return int(text)'
    assert "return None" in strip_non_code_text(code)


def test_span_text_strips_before_matching():
    code = "def parse(text):\n    # return None on invalid input\n    return int(text)"
    text = span_text("edit_file", {"path": "app.py", "new_string": code}, None)
    assert "return None" not in text
    assert "parse" in text  # the scope token survives in real code
