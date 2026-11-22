import os
import subprocess
import sys
import weakref
from ctypes import c_float, c_void_p

import numpy as np
import pytest

from metalloid import CompileError, Device, KernelNotFoundError
from metalloid._objc import send

ADD = """
kernel void add(const device float *a [[buffer(0)]],
                const device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}
"""


def test_device_name() -> None:
    assert Device().name.startswith("Apple")


def test_device_has_unified_memory() -> None:
    assert Device().has_unified_memory


def test_device_working_set_is_positive() -> None:
    assert Device().recommended_max_working_set_size > 0


def test_device_max_threads_per_threadgroup() -> None:
    assert min(Device().max_threads_per_threadgroup) > 0


def test_new_buffer_is_zeroed() -> None:
    buffer = Device().new_buffer((2, 3), np.float32)
    array = buffer.numpy()
    assert buffer.nbytes == 24
    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert not array.any()


def test_numpy_writes_reach_metal() -> None:
    buffer = Device().new_buffer(4, np.float32)
    buffer.numpy()[:] = [1.5, 2.5, 3.5, 4.5]
    contents = send(buffer._obj.ptr, "contents", restype=c_void_p)
    assert contents is not None
    assert list((c_float * 4).from_address(contents)) == [1.5, 2.5, 3.5, 4.5]


def test_array_keeps_buffer_alive() -> None:
    buffer = Device().new_buffer(4, np.float32)
    alive = weakref.ref(buffer)
    array = buffer.numpy()
    del buffer
    assert alive() is not None
    del array
    assert alive() is None


def test_new_buffer_from_copies_array() -> None:
    array = np.arange(6, dtype=np.int16).reshape(2, 3)
    buffer = Device().new_buffer_from(array)
    array[:] = 0
    np.testing.assert_array_equal(buffer.numpy(), [[0, 1, 2], [3, 4, 5]])
    assert buffer.numpy().dtype == np.int16


def test_new_buffer_from_lays_out_rows() -> None:
    columns = np.arange(6, dtype=np.float32).reshape(2, 3).T
    np.testing.assert_array_equal(Device().new_buffer_from(columns).numpy(), columns)


def test_new_buffer_from_rejects_lists() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        Device().new_buffer_from([1.0, 2.0])  # type: ignore[arg-type]


def test_buffers_reject_python_objects() -> None:
    device = Device()
    with pytest.raises(TypeError, match="Python objects"):
        device.new_buffer(4, object)
    with pytest.raises(TypeError, match="Python objects"):
        device.new_buffer_from(np.array([None]))


@pytest.mark.parametrize("shape", [0, (3, 0)])
def test_buffers_reject_no_bytes(shape: int | tuple[int, ...]) -> None:
    device = Device()
    with pytest.raises(ValueError, match="got 0"):
        device.new_buffer(shape, np.float32)
    with pytest.raises(ValueError, match="got 0"):
        device.new_buffer_from(np.zeros(shape, np.float32))


def test_new_buffer_rejects_more_than_max() -> None:
    device = Device()
    with pytest.raises(ValueError, match="1 to"):
        device.new_buffer(device.max_buffer_length + 1, np.uint8)


def test_kernel_reports_thread_limits() -> None:
    kernel = Device().compile(ADD).kernel("add")
    assert kernel.thread_execution_width > 0
    assert kernel.max_total_threads_per_threadgroup >= kernel.thread_execution_width


def test_compile_error_carries_compiler_message() -> None:
    with pytest.raises(CompileError, match="undeclared identifier 'c'"):
        Device().compile(ADD.replace("b[i]", "c[i]"))


def test_missing_kernel_raises() -> None:
    with pytest.raises(KernelNotFoundError, match="nope"):
        Device().compile(ADD).kernel("nope")


def test_function_names_are_sorted() -> None:
    scale = "kernel void scale(device float *x [[buffer(0)]]) { x[0] *= 2; }"
    assert Device().compile(ADD + scale).function_names == ["add", "scale"]


@pytest.mark.parametrize("n", [1, 1000, 100_003])
def test_run_adds_arrays(n: int) -> None:
    device = Device()
    add = device.compile(ADD).kernel("add")
    a = np.arange(n, dtype=np.float32)
    b = 2 * a
    out = device.new_buffer(n, np.float32)

    device.run(add, [device.new_buffer_from(a), device.new_buffer_from(b), out], n)

    np.testing.assert_array_equal(out.numpy(), a + b)


SAXPY = """
kernel void saxpy(constant float &alpha [[buffer(0)]],
                  const device float *x [[buffer(1)]],
                  device float *y [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    y[i] = alpha * x[i] + y[i];
}
"""


def test_run_passes_scalars() -> None:
    device = Device()
    saxpy = device.compile(SAXPY).kernel("saxpy")
    x = device.new_buffer_from(np.array([1, 2, 3, 4], np.float32))
    y = device.new_buffer_from(np.full(4, 10, np.float32))

    device.run(saxpy, [np.float32(2.5), x, y], threads=4)

    np.testing.assert_array_equal(y.numpy(), [12.5, 15, 17.5, 20])


@pytest.mark.parametrize("number", [2.5, 2])
def test_run_rejects_python_numbers(number: float) -> None:
    device = Device()
    saxpy = device.compile(SAXPY).kernel("saxpy")
    x, y = device.new_buffer(4, np.float32), device.new_buffer(4, np.float32)
    with pytest.raises(TypeError, match="NumPy scalar"):
        device.run(saxpy, [number, x, y], threads=4)  # type: ignore[list-item]


POSITION = """
kernel void position(device uint *out [[buffer(0)]],
                     uint3 p [[thread_position_in_grid]],
                     uint3 n [[threads_per_grid]]) {
    out[(p.z * n.y + p.y) * n.x + p.x] = p.x + 1000 * p.y + 1000000 * p.z;
}
"""


@pytest.mark.parametrize("grid", [(1000,), (5, 3), (999, 70), (5, 3, 2)])
def test_run_grid_positions(grid: tuple[int, ...]) -> None:
    device = Device()
    position = device.compile(POSITION).kernel("position")
    shape = (1, 1, *grid[::-1])[-3:]  # (depth, rows, cols)
    out = device.new_buffer(shape, np.uint32)

    device.run(position, [out], threads=grid)

    z, y, x = np.indices(shape)
    np.testing.assert_array_equal(out.numpy(), x + 1000 * y + 1_000_000 * z)


@pytest.mark.parametrize("threads", [0, -1, (4, 0), (), (1, 1, 1, 1)])
def test_run_rejects_bad_grids(threads: int | tuple[int, ...]) -> None:
    device = Device()
    add = device.compile(ADD).kernel("add")
    buffers = [device.new_buffer(4, np.float32) for _ in range(3)]
    with pytest.raises(ValueError, match="1 to 3 sizes"):
        device.run(add, buffers, threads=threads)


LEAK_PROBE = f"""
import sys
import numpy as np
from metalloid import Device

print("START", file=sys.stderr, flush=True)
device = Device()
library = device.compile({ADD!r})
add = library.kernel("add")
buffers = [device.new_buffer(4, np.float32) for _ in range(3)]
buffers[0].numpy(), device.new_buffer_from(np.zeros(4, np.float32))
repr(device), repr(library), repr(add), repr(buffers[0])
device.has_unified_memory, device.recommended_max_working_set_size
device.max_threads_per_threadgroup, device.max_buffer_length
add.max_total_threads_per_threadgroup, add.thread_execution_width
device.run(add, buffers, threads=4)
del device, library, add, buffers
"""


def test_nothing_autoreleases_outside_a_pool() -> None:
    result = subprocess.run(
        [sys.executable, "-c", LEAK_PROBE],
        env={**os.environ, "OBJC_DEBUG_MISSING_POOLS": "YES"},
        capture_output=True,
        text=True,
        check=True,
    )
    # Loading Metal leaks before START; that's Apple's, not ours.
    ours = result.stderr.partition("START")[2]
    assert "MISSING POOLS" not in ours
