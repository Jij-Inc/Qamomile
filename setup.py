import sys
import platform
from setuptools import setup

if "cudaq" in sys.argv or any("cudaq" in arg for arg in sys.argv):
    if platform.system().lower() != "linux":
        sys.stderr.write("ERROR: 'cudaq' is currently only supported on Linux.\n")
        sys.exit(1)

setup()
