"""
scripts/run_pipeline.py
------------------------
End-to-end pipeline: load (or build) features → train all models →
run 5-fold CV evaluation → save charts.

This is a convenience wrapper around train_all.py that can be run
directly from the project root.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --rebuild-features
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Delegate to train_all
from scripts.train_all import main

if __name__ == "__main__":
    main()
