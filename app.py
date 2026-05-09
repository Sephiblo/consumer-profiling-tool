"""Streamlit Cloud entrypoint.

The implementation lives in consumer_profiling_tool/. This wrapper makes
deployments work when Streamlit Cloud is configured to run app.py from the
repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "consumer_profiling_tool"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from consumer_profiling_tool.app import main


if __name__ == "__main__":
    main()

