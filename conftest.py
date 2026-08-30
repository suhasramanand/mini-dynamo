import os
import sys

# Ensure the repo root is importable so `import common...` works under pytest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
