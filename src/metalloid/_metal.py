import ctypes
import math
from collections.abc import Sequence
from ctypes import (
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_ubyte,
    c_uint64,
    c_ulong,
    c_void_p,
)
from typing import Any

import numpy as np
import numpy.typing as npt

from metalloid._objc import AutoreleasePool, Object, from_str, send, to_str

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


def _threadgroup(kernel: "Kernel", dimensions: int) -> _MTLSize:
    """Return the largest threadgroup for ``kernel`` on a grid of ``dimensions``.

    A 2D or 3D threadgroup is one SIMD-group wide and as tall as the limit
    allows, as in Apple's "Calculating threadgroup and grid sizes".
    """
    total = kernel.max_total_threads_per_threadgroup
    if dimensions == 1:
        return _MTLSize(total, 1, 1)
    width = kernel.thread_execution_width
    return _MTLSize(width, total // width, 1)


class Device:
    """The system's default GPU."""

    def __init__(self) -> None:
        """Open the default GPU.

        Raises:
            MetalError: If the system has no Metal device.
        """
        with AutoreleasePool():
            ptr: int | None = _metal.MTLCreateSystemDefaultDevice()
            if ptr is None:
                raise MetalError("no Metal device found")
            self._obj = Object(ptr, owned=True)

            # One queue for the device's lifetime as per Apple.
            queue = send(self._obj.ptr, "newCommandQueue", restype=c_void_p)
            if queue is None:
                raise MetalError("Metal could not create a command queue")
            self._queue = Object(queue, owned=True)

    def __repr__(self) -> str:
        return f"<Device {self.name!r}>"

    @property
    def name(self) -> str:
        """The GPU's full name."""
        with AutoreleasePool():
            return to_str(send(self._obj.ptr, "name", restype=c_void_p))

    @property
    def has_unified_memory(self) -> bool:
        """Whether the GPU shares its memory with the CPU."""
        with AutoreleasePool():
            return send(self._obj.ptr, "hasUnifiedMemory", restype=c_bool)

    @property
    def recommended_max_working_set_size(self) -> int:
        """Bytes the GPU can use without hurting performance."""
        with AutoreleasePool():
            return send(self._obj.ptr, "recommendedMaxWorkingSetSize", restype=c_uint64)

    @property
    def max_threads_per_threadgroup(self) -> tuple[int, int, int]:
        """The largest threadgroup as ``(width, height, depth)``."""
        with AutoreleasePool():
            size = send(self._obj.ptr, "maxThreadsPerThreadgroup", restype=_MTLSize)
        return size.width, size.height, size.depth

    @property
    def max_buffer_length(self) -> int:
        """The largest buffer, in bytes, the GPU can allocate."""
        with AutoreleasePool():
            return send(self._obj.ptr, "maxBufferLength", restype=c_ulong)

    def new_buffer(
        self, shape: int | tuple[int, ...], dtype: npt.DTypeLike
    ) -> "Buffer":
        """Allocate a zeroed buffer of ``shape`` and ``dtype``, like `numpy.zeros`.

        Raises:
            TypeError: If ``dtype`` holds Python objects.
            ValueError: If the buffer would hold no bytes, or more than
                `max_buffer_length`.
            AllocationError: If Metal can't allocate it.
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        dtype = np.dtype(dtype)
        nbytes = self._check_size(shape, dtype)

        with AutoreleasePool():
            ptr = send(
                self._obj.ptr,
                "newBufferWithLength:options:",
                nbytes,
                _MTL_RESOURCE_STORAGE_MODE_SHARED,
                restype=c_void_p,
                argtypes=(c_ulong, c_ulong),
            )
            if ptr is None:
                raise AllocationError(f"Metal could not allocate {nbytes} bytes")

            return Buffer(Object(ptr, owned=True), shape, dtype)

    def new_buffer_from(self, array: npt.NDArray[Any]) -> "Buffer":
        """Allocate a buffer holding a copy of ``array``, with its shape and dtype.

        Raises:
            TypeError: If ``array`` isn't a NumPy array, or holds Python
                objects. A list would become ``float64`` or ``int64``, which
                rarely match the kernel's types.
            ValueError: If ``array`` is empty, or larger than
                `max_buffer_length`.
            AllocationError: If Metal can't allocate it.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected a NumPy array, got {type(array).__name__}")
        # Metal copies one block of bytes, in C order.
        if not array.flags.c_contiguous:
            array = array.copy(order="C")
        nbytes = self._check_size(array.shape, array.dtype)

        with AutoreleasePool():
            ptr = send(
                self._obj.ptr,
                "newBufferWithBytes:length:options:",
                array.ctypes.data,
                nbytes,
                _MTL_RESOURCE_STORAGE_MODE_SHARED,
                restype=c_void_p,
                argtypes=(c_void_p, c_ulong, c_ulong),
            )
            if ptr is None:
                raise AllocationError(f"Metal could not allocate {nbytes} bytes")

            return Buffer(Object(ptr, owned=True), array.shape, array.dtype)

    def _check_size(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
        """Return the bytes a buffer of ``shape`` and ``dtype`` needs."""
        if dtype.hasobject:
            raise TypeError(f"the GPU can't read Python objects, got dtype {dtype}")
        nbytes = math.prod(shape) * dtype.itemsize
        limit = self.max_buffer_length
        if not 0 < nbytes <= limit:
            raise ValueError(f"a buffer holds 1 to {limit} bytes, got {nbytes}")
        return nbytes

    def compile(self, source: str) -> "Library":
        """Compile Metal Shading Language ``source`` into a library.

        Raises:
            CompileError: If the source doesn't compile.
        """
        with AutoreleasePool():
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

    def run(
        self,
        kernel: "Kernel",
        args: Sequence["Buffer | np.generic"],
        threads: int | tuple[int, ...],
    ) -> None:
        """Run ``kernel`` on a grid of GPU threads and wait until it finishes.

        ``args[i]`` fills the kernel's ``[[buffer(i)]]`` slot: a `Buffer`, or
        a NumPy scalar such as ``np.float32(2.5)``, which Metal copies when
        the work is encoded.

        ``threads`` is the grid: ``n``, ``(width, height)``, or ``(width,
        height, depth)``. Each thread gets its position in
        ``[[thread_position_in_grid]]``, ``x`` first. An array of shape
        ``(rows, cols)`` takes ``threads=(cols, rows)``.

        Raises:
            TypeError: If an argument is neither. A Python ``float`` or
                ``int`` has no size the kernel can rely on.
            ValueError: If ``threads`` has more than 3 sizes, or one below 1.
            MetalError: If Metal can't encode the work, or the GPU fails to
                run it.
        """
        grid = (threads,) if isinstance(threads, int) else tuple(threads)
        if not 1 <= len(grid) <= 3 or min(grid) < 1:
            raise ValueError(f"threads takes 1 to 3 sizes of at least 1, got {threads}")

        # Metal aborts the process if an encoder is freed before endEncoding.
        for arg in args:
            if not isinstance(arg, Buffer | np.generic):
                raise TypeError(
                    f"expected a Buffer or a NumPy scalar such as np.float32(2.5), "
                    f"got {type(arg).__name__}"
                )

        width, height, depth = grid + (1,) * (3 - len(grid))
        group = _threadgroup(kernel, len(grid))

        with AutoreleasePool():
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
            for idx, arg in enumerate(args):
                if isinstance(arg, Buffer):
                    send(
                        encoder,
                        "setBuffer:offset:atIndex:",
                        arg._obj.ptr,
                        0,
                        idx,
                        restype=None,
                        argtypes=(c_void_p, c_ulong, c_ulong),
                    )
                else:
                    send(
                        encoder,
                        "setBytes:length:atIndex:",
                        arg.tobytes(),
                        arg.nbytes,
                        idx,
                        restype=None,
                        argtypes=(c_char_p, c_ulong, c_ulong),
                    )
            send(
                encoder,
                "dispatchThreads:threadsPerThreadgroup:",
                _MTLSize(width, height, depth),
                group,
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
    """GPU memory shaped like a NumPy array, shared by the CPU and GPU.

    Attributes:
        shape: The array's shape.
        dtype: The array's element type.
        nbytes: The buffer's size in bytes.
    """

    def __init__(
        self, obj: Object, shape: tuple[int, ...], dtype: np.dtype[Any]
    ) -> None:
        """Wrap the ``MTLBuffer`` in ``obj``. Use `Device.new_buffer`."""
        self._obj = obj
        contents = send(obj.ptr, "contents", restype=c_void_p)
        assert contents is not None, "contents is nonnull in MTLBuffer.h"
        self._contents = contents
        self.shape = shape
        self.dtype = dtype
        self.nbytes = math.prod(shape) * dtype.itemsize

    def __repr__(self) -> str:
        return f"<Buffer {self.shape} {self.dtype}>"

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

    def numpy(self) -> npt.NDArray[Any]:
        """Return a NumPy array over the buffer's memory, without copying.

        Writes to the array are writes to the buffer. The array keeps the
        buffer alive. To read the bytes as another type, use the array's
        ``view``.
        """
        return np.frombuffer(self, dtype=self.dtype).reshape(self.shape)


class Library:
    """Compiled kernels, from `Device.compile`."""

    def __init__(self, device: Device, obj: Object) -> None:
        """Wrap the ``MTLLibrary`` in ``obj``, compiled for ``device``."""
        self._device = device
        self._obj = obj

    def __repr__(self) -> str:
        return f"<Library {self.function_names!r}>"

    @property
    def function_names(self) -> list[str]:
        """The names of the library's functions, kernels among them, sorted."""
        with AutoreleasePool():
            array = send(self._obj.ptr, "functionNames", restype=c_void_p)
            assert array is not None, "functionNames is nonnull in MTLLibrary.h"
            count = send(array, "count", restype=c_ulong)
            names = (
                send(array, "objectAtIndex:", i, restype=c_void_p, argtypes=(c_ulong,))
                for i in range(count)
            )
            return sorted(to_str(name) for name in names)

    def kernel(self, name: str) -> "Kernel":
        """Return the kernel function ``name``, ready to run.

        Raises:
            KernelNotFoundError: If the library has no kernel called ``name``.
            CompileError: If Metal can't build a pipeline for it.
        """
        with AutoreleasePool():
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

        return Kernel(Object(ptr, owned=True), name)


class Kernel:
    """A kernel function compiled into a pipeline the GPU can run.

    Attributes:
        name: The kernel function's name in its source.
    """

    def __init__(self, obj: Object, name: str) -> None:
        """Wrap the ``MTLComputePipelineState`` in ``obj``. Use `Library.kernel`."""
        self._obj = obj
        self.name = name

    def __repr__(self) -> str:
        return f"<Kernel {self.name!r}>"

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        """The most threads one threadgroup can have for this kernel."""
        with AutoreleasePool():
            return send(self._obj.ptr, "maxTotalThreadsPerThreadgroup", restype=c_ulong)

    @property
    def thread_execution_width(self) -> int:
        """Threads the GPU runs in lock-step; size threadgroups in multiples."""
        with AutoreleasePool():
            return send(self._obj.ptr, "threadExecutionWidth", restype=c_ulong)
