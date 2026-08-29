"""
build_index.py
----------------
Run this once (and again any time you change files in data/) to build
the vector database.

Usage:
    python build_index.py
"""

from vector_store import build_index

if __name__ == "__main__":
    build_index()
