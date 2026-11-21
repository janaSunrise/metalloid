import ctypes
from collections.abc import Sequence
from ctypes import Structure, byref, c_bool, c_ubyte, c_uint64, c_ulong, c_void_p
from typing import Any

import numpy as np
import numpy.typing as npt

from metalloid._objc import Object, autorelease_pool, from_str, send, to_str

_metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")

# MTLCreateSystemDefaultDevice needs it loaded, per Apple's documentation.
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")

_metal.MTLCreateSystemDefaultDevice.argtypes = []
_metal.MTLCreateSystemDefaultDevice.restype = c_void_p

# MTLStorageModeShared << MTLResourceStorageModeShift: memory the CPU and GPU share.
_MTL_RESOURCE_STORAGE_MODE_SHARED = 0

# MTLCommandBufferStatusError: the GPU couldn't run the command buffer.
_MTL_COMMAND_BUFFER_STATUS_ERROR = 5


class MetalError(Exception):
    """Base class of metalloid's errors."""


class CompileError(MetalError):
    """Metal couldn't compile a kernel. The message is the compiler's."""


class KernelNotFoundError(MetalError, LookupError):
    """A library has no kernel with the requested name."""


class AllocationError(MetalError, MemoryError):
    """Metal couldn't allocate a buffer."""


def _describe(error: c_void_p) -> str:
    """Return the description of the ``NSError`` in ``error``, if there is one."""
    if error.value is None:
        return "Metal reported no error description"
    return to_str(send(error.value, "localizedDescription", restype=c_void_p))


class _MTLSize(Structure):
    _fields_ = (("width", c_ulong), ("height", c_ulong), ("depth", c_ulong))
    width: int
    height: int
    depth: int


class Device:
    """The system's default GPU."""

    def __init__(self) -> None:
        """Open the default GPU.

        Raises:
            MetalError: If the system has no Metal device.
        """
        with autorelease_pool():
            ptr: int | None = _metal.MTLCreateSystemDefaultDevice()
            if ptr is None:
                raise MetalError("no Metal device found")
            self._obj = Object(ptr, owned=True)

            # One queue for the device's lifetime as per Apple.
            queue = send(self._obj.ptr, "newCommandQueue", restype=c_void_p)
            if queue is None:
                raise MetalError("Metal could not create a command queue")
            self._queue = Object(queue, owned=True)

    @property
    def name(self) -> str:
        """The GPU's full name."""
        return to_str(send(self._obj.ptr, "name", restype=c_void_p))

    @property
    def has_unified_memory(self) -> bool:
        """Whether the GPU shares its memory with the CPU."""
        return send(self._obj.ptr, "hasUnifiedMemory", restype=c_bool)

    @property
    def recommended_max_working_set_size(self) -> int:
        """Bytes the GPU can use without hurting performance."""
        return send(self._obj.ptr, "recommendedMaxWorkingSetSize", restype=c_uint64)

    @property
    def max_threads_per_threadgroup(self) -> tuple[int, int, int]:
        """The largest threadgroup as ``(width, height, depth)``."""
        size = send(self._obj.ptr, "maxThreadsPerThreadgroup", restype=_MTLSize)
        return size.width, size.height, size.depth

    @property
    def max_buffer_length(self) -> int:
        """The largest buffer, in bytes, the GPU can allocate."""
        return send(self._obj.ptr, "maxBufferLength", restype=c_ulong)

    def new_buffer(self, nbytes: int) -> "Buffer":
        """Allocate ``nbytes`` of zeroed memory shared by the CPU and GPU.

        Raises:
            AllocationError: If Metal can't allocate it. ``nbytes`` must be
                between 1 and `max_buffer_length`.
        """
        with autorelease_pool():
            ptr = send(
                self._obj.ptr,
                "newBufferWithLength:options:",
                nbytes,
                _MTL_RESOURCE_STORAGE_MODE_SHARED,
                restype=c_void_p,
                argtypes=(c_ulong, c_ulong),
            )

        if ptr is None:
            raise AllocationError(f"Metal could not allocate a {nbytes}-byte buffer")

        return Buffer(Object(ptr, owned=True), nbytes)

    def compile(self, source: str) -> "Library":
        """Compile Metal Shading Language ``source`` into a library.

        Raises:
            CompileError: If the source doesn't compile.
        """
        with autorelease_pool():
            error = c_void_p()
            ptr = send(
                self._obj.ptr,
                "newLibraryWithSource:options:error:",
                from_str(source),
                None,
                byref(error),
                restype=c_void_p,
                argtypes=(c_void_p, c_void_p, c_void_p),
            )
            if ptr is None:
                raise CompileError(_describe(error))

        return Library(self, Object(ptr, owned=True))

    def run(self, kernel: "Kernel", buffers: Sequence["Buffer"], threads: int) -> None:
        """Run ``kernel`` on ``threads`` GPU threads and wait until it finishes.

        ``buffers[i]`` fills the kernel's ``[[buffer(i)]]`` slot, and each
        thread gets its index in ``[[thread_position_in_grid]]``.

        Raises:
            ValueError: If ``threads`` is less than 1.
            MetalError: If Metal can't encode the work, or the GPU fails to
                run it.
        """
        if threads < 1:
            raise ValueError(f"threads must be at least 1, got {threads}")
        group = min(kernel.max_total_threads_per_threadgroup, threads)

        with autorelease_pool():
            commands = send(self._queue.ptr, "commandBuffer", restype=c_void_p)
            if commands is None:
                raise MetalError("Metal could not create a command buffer")
            encoder = send(commands, "computeCommandEncoder", restype=c_void_p)
            if encoder is None:
                raise MetalError("Metal could not create a compute command encoder")

            send(
                encoder,
                "setComputePipelineState:",
                kernel._obj.ptr,
                restype=None,
                argtypes=(c_void_p,),
            )
            for idx, buffer in enumerate(buffers):
                send(
                    encoder,
                    "setBuffer:offset:atIndex:",
                    buffer._obj.ptr,
                    0,
                    idx,
                    restype=None,
                    argtypes=(c_void_p, c_ulong, c_ulong),
                )
            send(
                encoder,
                "dispatchThreads:threadsPerThreadgroup:",
                _MTLSize(threads, 1, 1),
                _MTLSize(group, 1, 1),
                restype=None,
                argtypes=(_MTLSize, _MTLSize),
            )
            send(encoder, "endEncoding", restype=None)

            send(commands, "commit", restype=None)
            send(commands, "waitUntilCompleted", restype=None)

            status = send(commands, "status", restype=c_ulong)
            if status == _MTL_COMMAND_BUFFER_STATUS_ERROR:
                error = c_void_p(send(commands, "error", restype=c_void_p))
                raise MetalError(_describe(error))


class Buffer:
    """GPU memory the CPU reads and writes through NumPy, without copying.

    Attributes:
        nbytes: The buffer's size in bytes.
    """

    def __init__(self, obj: Object, nbytes: int) -> None:
        """Wrap the ``MTLBuffer`` in ``obj``. Use `Device.new_buffer`."""
        self._obj = obj
        contents = send(obj.ptr, "contents", restype=c_void_p)
        assert contents is not None, "contents is nonnull in MTLBuffer.h"
        self._contents = contents
        self.nbytes = nbytes

    def __buffer__(self, flags: int, /) -> memoryview:
        """Expose the buffer's memory to the buffer protocol.

        ``c_ubyte * nbytes`` is the C type ``unsigned char[nbytes]``.
        ``from_address`` lays it over the buffer's memory without allocating
        or copying, and ``memoryview`` wraps that for the protocol.

        The bytes carry no type; `numpy` gives them one. The view's format is
        ``"<B"``, which ``memoryview`` can't assign items through, so read and
        write through `numpy`.
        """
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


class Library:
    """Compiled kernels, from `Device.compile`."""

    def __init__(self, device: Device, obj: Object) -> None:
        """Wrap the ``MTLLibrary`` in ``obj``, compiled for ``device``."""
        self._device = device
        self._obj = obj

    def kernel(self, name: str) -> "Kernel":
        """Return the kernel function ``name``, ready to run.

        Raises:
            KernelNotFoundError: If the library has no kernel called ``name``.
            CompileError: If Metal can't build a pipeline for it.
        """
        with autorelease_pool():
            ptr = send(
                self._obj.ptr,
                "newFunctionWithName:",
                from_str(name),
                restype=c_void_p,
                argtypes=(c_void_p,),
            )
            if ptr is None:
                raise KernelNotFoundError(f"no kernel named {name!r} in the library")
            function = Object(ptr, owned=True)

            error = c_void_p()
            ptr = send(
                self._device._obj.ptr,
                "newComputePipelineStateWithFunction:error:",
                function.ptr,
                byref(error),
                restype=c_void_p,
                argtypes=(c_void_p, c_void_p),
            )
            if ptr is None:
                raise CompileError(_describe(error))

        return Kernel(Object(ptr, owned=True))


class Kernel:
    """A kernel function compiled into a pipeline the GPU can run."""

    def __init__(self, obj: Object) -> None:
        """Wrap the ``MTLComputePipelineState`` in ``obj``."""
        self._obj = obj

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        """The most threads one threadgroup can have for this kernel."""
        return send(self._obj.ptr, "maxTotalThreadsPerThreadgroup", restype=c_ulong)

    @property
    def thread_execution_width(self) -> int:
        """Threads the GPU runs in lock-step; size threadgroups in multiples."""
        return send(self._obj.ptr, "threadExecutionWidth", restype=c_ulong)
