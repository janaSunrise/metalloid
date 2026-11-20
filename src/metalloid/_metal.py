import ctypes
from ctypes import Structure, c_bool, c_uint64, c_ulong, c_void_p

from metalloid._objc import Object, autorelease_pool, to_str

_metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")

# MTLCreateSystemDefaultDevice needs it loaded, per Apple's documentation.
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")

_metal.MTLCreateSystemDefaultDevice.argtypes = []
_metal.MTLCreateSystemDefaultDevice.restype = c_void_p


class _MTLSize(Structure):
    _fields_ = (("width", c_ulong), ("height", c_ulong), ("depth", c_ulong))


class Device:
    """The system's default GPU."""

    def __init__(self) -> None:
        """Open the default GPU.

        Raises:
            RuntimeError: If the system has no Metal device.
        """
        with autorelease_pool():
            ptr: int | None = _metal.MTLCreateSystemDefaultDevice()

        if ptr is None:
            raise RuntimeError("no Metal device found")

        self._obj = Object(ptr, owned=True)

    @property
    def name(self) -> str:
        """The GPU's full name."""
        return to_str(self._obj.send("name", restype=c_void_p))

    @property
    def has_unified_memory(self) -> bool:
        """Whether the GPU shares its memory with the CPU."""
        unified: bool = self._obj.send("hasUnifiedMemory", restype=c_bool)
        return unified

    @property
    def recommended_max_working_set_size(self) -> int:
        """Bytes the GPU can use without hurting performance."""
        size: int = self._obj.send("recommendedMaxWorkingSetSize", restype=c_uint64)
        return size

    @property
    def max_threads_per_threadgroup(self) -> tuple[int, int, int]:
        """The largest threadgroup as ``(width, height, depth)``."""
        size: _MTLSize = self._obj.send("maxThreadsPerThreadgroup", restype=_MTLSize)
        return size.width, size.height, size.depth
