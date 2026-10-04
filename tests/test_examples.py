"""The README "try it" example must keep working."""

import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "try_it.py"


def test_try_it_example_runs(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert "matched_rows" in capsys.readouterr().out
