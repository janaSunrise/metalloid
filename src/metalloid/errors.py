"""The errors metalloid raises."""


class MetalError(Exception):
    """Base class of metalloid's errors."""


class CompileError(MetalError):
    """Metal couldn't compile a kernel. The message is the compiler's."""


class KernelNotFoundError(MetalError, LookupError):
    """A library has no kernel with the requested name."""


class AllocationError(MetalError, MemoryError):
    """Metal couldn't allocate a buffer."""
