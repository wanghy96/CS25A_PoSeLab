# resources.py
from __future__ import annotations
import os
import sys

def resource_path(rel_path: str) -> str:
    """
    Return absolute path to resource, works for dev and for PyInstaller.
    """
    if hasattr(sys, "_MEIPASS"):
        base = getattr(sys, "_MEIPASS")  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)