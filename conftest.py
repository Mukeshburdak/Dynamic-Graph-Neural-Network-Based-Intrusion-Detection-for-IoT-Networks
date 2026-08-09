"""
conftest.py

Ensures the project root is on sys.path so `from src import ...` resolves
correctly no matter where pytest is invoked from. Pytest automatically
discovers and loads conftest.py files at collection time.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
