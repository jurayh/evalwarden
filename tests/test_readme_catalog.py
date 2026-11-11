"""The README catalog must list every registered check, and no other IDs.

The catalog drifted once (DATA-004, TRAJ-003, TRAJ-004, and the whole
DIFF lane shipped while the README still listed 24 checks). This test
pins the README to the registry in both directions.
"""
from __future__ import annotations

import re
from pathlib import Path

from evalwarden.checks import BY_ID

README = Path(__file__).resolve().parent.parent / "README.md"

_CHECK_ID = re.compile(r"\b(?:ENV|GRAD|COST|JUDGE|DATA|TRAJ|NOISE|DIFF)-\d{3}\b")


def test_readme_lists_every_registered_check():
    text = README.read_text(encoding="utf-8")
    mentioned = set(_CHECK_ID.findall(text))
    assert mentioned == set(BY_ID), (
        f"README/registry mismatch: missing from README {sorted(set(BY_ID) - mentioned)}, "
        f"in README but unregistered {sorted(mentioned - set(BY_ID))}"
    )


def test_readme_cli_block_names_the_side_inputs_and_version():
    text = README.read_text(encoding="utf-8")
    for flag in ("--training-file", "--spec-file", "--format", "--version"):
        assert flag in text, f"README CLI block is missing {flag}"
