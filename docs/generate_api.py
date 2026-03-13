"""Generate API reference markdown from qamomile package docstrings.

Uses griffe to introspect the qamomile package and generates
MyST-compatible markdown files in docs/api/.

Usage:
    python generate_api.py
"""

from api_gen import main

if __name__ == "__main__":
    main()
