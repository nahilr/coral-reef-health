"""Run every plain-assert test module in this directory.

    python tests/run_all.py

No pytest (or any other test dependency) is required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent))
    total = 0
    for path in sorted(root.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tests = [value for name, value in vars(module).items() if name.startswith("test_") and callable(value)]
        for test in tests:
            test()
        total += len(tests)
        print(f"{path.name}: {len(tests)} passed")
    print(f"{total} tests passed")


if __name__ == "__main__":
    main()
