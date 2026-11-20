import ctypes
from ctypes import Structure, c_bool, c_ubyte, c_uint64, c_ulong, c_void_p
from typing import Any

import numpy as np
import numpy.typing as npt

from metalloid._objc import Object, autorelease_pool, to_str

_metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")

# MTLCreateSystemDefaultDevice needs it loaded, per Apple's documentation.
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")

_metal.MTLCreateSystemDefaultDevice.argtypes = []
_metal.MTLCreateSystemDefaultDevice.restype = c_void_p

# MTLStorageModeShared << MTLResourceStorageModeShift: memory the CPU and GPU share.
_MTL_RESOURCE_STORAGE_MODE_SHARED = 0


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

    @property
    def max_buffer_length(self) -> int:
        """The largest buffer, in bytes, the GPU can allocate."""
        length: int = self._obj.send("maxBufferLength", restype=c_ulong)
        return length

    def new_buffer(self, nbytes: int) -> "Buffer":
        """Allocate ``nbytes`` of zeroed memory shared by the CPU and GPU.

        Raises:
            MemoryError: If Metal can't allocate it. ``nbytes`` must be
                between 1 and `max_buffer_length`.
        """
        with autorelease_pool():
            ptr: int | None = self._obj.send(
                "newBufferWithLength:options:",
                nbytes,
                _MTL_RESOURCE_STORAGE_MODE_SHARED,
                restype=c_void_p,
                argtypes=(c_ulong, c_ulong),
            )

        if ptr is None:
            raise MemoryError(f"Metal could not allocate a {nbytes}-byte buffer")

        return Buffer(Object(ptr, owned=True), nbytes)


class Buffer:
    """GPU memory the CPU reads and writes through NumPy, without copying.

    Attributes:
        nbytes: The buffer's size in bytes.
    """

    def __init__(self, obj: Object, nbytes: int) -> None:
        """Wrap the ``MTLBuffer`` in ``obj``. Use `Device.new_buffer`."""
        self._obj = obj
        self._contents: int = obj.send("contents", restype=c_void_p)
        self.nbytes = nbytes

    def __buffer__(self, flags: int, /) -> memoryview:
        """Expose the buffer's memory to the buffer protocol."""
        return memoryview((c_ubyte * self.nbytes).from_address(self._contents))

    def numpy(
        self, dtype: npt.DTypeLike, shape: int | tuple[int, ...] = -1
    ) -> npt.NDArray[Any]:
        """Return a NumPy array over the buffer's memory.

        Writes to the array are writes to the buffer. The array keeps the
        buffer alive.

        Args:
            dtype: How to read the bytes, e.g. ``np.float32``.
            shape: The array's shape. Defaults to one dimension.
        """
        return np.frombuffer(self, dtype=dtype).reshape(shape)
