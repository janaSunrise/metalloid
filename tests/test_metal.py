import weakref
from ctypes import c_float, c_void_p

import numpy as np
import pytest

from metalloid._metal import CompileError, Device

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
    buffer = Device().new_buffer(16)
    assert buffer.nbytes == 16
    assert not buffer.numpy(np.float32).any()


def test_numpy_writes_reach_metal() -> None:
    buffer = Device().new_buffer(16)
    buffer.numpy(np.float32)[:] = [1.5, 2.5, 3.5, 4.5]
    contents = buffer._obj.send("contents", restype=c_void_p)
    assert list((c_float * 4).from_address(contents)) == [1.5, 2.5, 3.5, 4.5]


def test_numpy_shape() -> None:
    assert Device().new_buffer(24).numpy(np.float32, (2, 3)).shape == (2, 3)


def test_array_keeps_buffer_alive() -> None:
    buffer = Device().new_buffer(16)
    alive = weakref.ref(buffer)
    array = buffer.numpy(np.float32)
    del buffer
    assert alive() is not None
    del array
    assert alive() is None


def test_new_buffer_rejects_zero_bytes() -> None:
    with pytest.raises(MemoryError):
        Device().new_buffer(0)


def test_new_buffer_rejects_more_than_max() -> None:
    device = Device()
    with pytest.raises(MemoryError):
        device.new_buffer(device.max_buffer_length + 1)


def test_kernel_reports_thread_limits() -> None:
    kernel = Device().compile(ADD).kernel("add")
    assert kernel.thread_execution_width > 0
    assert kernel.max_total_threads_per_threadgroup >= kernel.thread_execution_width


def test_compile_error_carries_compiler_message() -> None:
    with pytest.raises(CompileError, match="undeclared identifier 'c'"):
        Device().compile(ADD.replace("b[i]", "c[i]"))


def test_missing_kernel_raises() -> None:
    with pytest.raises(LookupError, match="nope"):
        Device().compile(ADD).kernel("nope")
