"""Process-local ownership of backend executables and their SDK dependencies."""


class ExecutablePayload:
    """Own reusable SDK templates and isolate mutable state for each launch.

    Backends need no serialization methods to participate in compilation and
    execution. Optional storage codecs belong to the persistence boundary.
    """
