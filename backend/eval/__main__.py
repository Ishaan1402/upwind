"""CLI entrypoint so `python -m backend.eval ...` still works now that
`backend/eval` is a package (see backend/eval/__init__.py)."""

import sys

from backend.eval import main

if __name__ == "__main__":
    sys.exit(main())
