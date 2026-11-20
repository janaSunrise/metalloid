from metalloid._metal import Device


def test_device_name() -> None:
    assert Device().name.startswith("Apple")


def test_device_has_unified_memory() -> None:
    assert Device().has_unified_memory


def test_device_working_set_is_positive() -> None:
    assert Device().recommended_max_working_set_size > 0


def test_device_max_threads_per_threadgroup() -> None:
    assert min(Device().max_threads_per_threadgroup) > 0
