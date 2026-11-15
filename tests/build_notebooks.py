"""
Build the companion notebooks: one per chapter, every code cell asserting what it recomputes.

    python tests/build_notebooks.py          # all
    python tests/build_notebooks.py 01 05    # matching ones
    python tests/run_notebooks.py            # execute them (a failed assert fails the notebook)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
sys.path.insert(0, HERE)

from nb_lib import NOTEBOOKS, to_ipynb  # noqa: E402
import nb_chapters  # noqa: E402,F401  (registers one notebook per chapter)




def main():
    filters = sys.argv[1:]
    names = [n for n in NOTEBOOKS if not filters or any(f in n for f in filters)]
    os.makedirs(NB_DIR, exist_ok=True)
    for name in names:
        with open(os.path.join(NB_DIR, name + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(to_ipynb(NOTEBOOKS[name]), f, indent=1, ensure_ascii=False)
            f.write("\n")
        n_assert = sum(src.count("assert ") for k, src in NOTEBOOKS[name] if k == "code")
        print(f"  wrote notebooks/{name}.ipynb  ({len(NOTEBOOKS[name])} cells, {n_assert} asserts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
