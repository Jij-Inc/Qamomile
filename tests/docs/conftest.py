"""Configure documentation tests for unattended rendering."""

import os

# Set the backend before pytest imports any documentation test module. This is
# intentionally unconditional so a developer's interactive backend cannot open
# a window and block an unattended test run.
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_USE_SHM"] = "0"

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
