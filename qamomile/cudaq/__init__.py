import importlib.util
import sys

# Check if the operating system is linux since CUDA-Q is only supported on Linux systems.
if sys.platform != "linux":
    os_error_message = """
        The 'cudaq' package is currently only available on Linux systems.
        (See https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility)
        You could use a docker image with cudaq installed.
        (See https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#install-docker-image)
    """
    raise OSError(os_error_message)

# Check the availability of the 'cudaq' package.
elif importlib.util.find_spec("cudaq") is None:
    import_error_message = """
            The 'cudaq' package is not found. 
            "Please make sure you have installed Qamomile with the 'cudaq' extra: 'pip install qamomile[cudaq]'"
        """
    raise ImportError(import_error_message)

else:
    from qamomile.cudaq.transpiler import CudaqTranspiler
    from qamomile.cudaq.exceptions import QamomileCudaqTranspileError
    from qamomile.cudaq.parameter_converter import convert_parameter

    __all__ = ["CudaqTranspiler", "QamomileCudaqTranspileError", "convert_parameter"]
