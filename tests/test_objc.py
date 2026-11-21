from ctypes import c_char_p, c_ulong, c_ushort, c_void_p

import pytest

from metalloid._objc import (
    AutoreleasePool,
    Object,
    from_str,
    get_class,
    release,
    retain,
    selector,
    send,
    to_str,
)


def make_object() -> int:
    obj = send(get_class("NSObject"), "new", restype=c_void_p)
    assert obj is not None
    return obj


def retain_count(obj: int) -> int:
    return send(obj, "retainCount", restype=c_ulong)


def test_get_class_returns_full_pointer() -> None:
    assert get_class("NSObject") > 0xFFFF_FFFF


def test_get_class_unknown_raises() -> None:
    with pytest.raises(LookupError, match="NoSuchClass"):
        get_class("NoSuchClass")


def test_selectors_are_interned() -> None:
    assert selector("length") == selector("length")
    assert selector("length") != selector("hash")


def test_send_returns_integer() -> None:
    with AutoreleasePool():
        assert send(from_str("metalloid"), "length", restype=c_ulong) == 9


def test_send_round_trips_string() -> None:
    with AutoreleasePool():
        string = from_str("héllo")
        assert send(string, "UTF8String", restype=c_char_p) == "héllo".encode()


def test_from_str_round_trips() -> None:
    with AutoreleasePool():
        assert to_str(from_str("héllo")) == "héllo"


def test_to_str_rejects_nil() -> None:
    with pytest.raises(ValueError, match="nil"):
        to_str(None)


def test_to_str_rejects_string_without_utf8() -> None:
    lone_surrogate = (c_ushort * 1)(0xD800)
    with AutoreleasePool():
        string = send(
            get_class("NSString"),
            "stringWithCharacters:length:",
            lone_surrogate,
            1,
            restype=c_void_p,
            argtypes=(c_void_p, c_ulong),
        )
        with pytest.raises(ValueError, match="UTF-8"):
            to_str(string)


def test_send_rejects_wrong_argument_count() -> None:
    with pytest.raises(TypeError):
        send(get_class("NSObject"), "hash", 1, restype=c_ulong)


def test_new_object_is_owned() -> None:
    obj = make_object()
    assert retain_count(obj) == 1
    release(obj)


def test_retain_and_release_balance() -> None:
    obj = make_object()
    assert retain(obj) == obj
    assert retain_count(obj) == 2
    release(obj)
    assert retain_count(obj) == 1
    release(obj)


def test_pool_releases_autoreleased_objects() -> None:
    obj = make_object()
    with AutoreleasePool():
        send(retain(obj), "autorelease", restype=c_void_p)
        assert retain_count(obj) == 2
    assert retain_count(obj) == 1
    release(obj)


def test_pool_drains_on_exception() -> None:
    def autorelease_then_fail(obj: int) -> None:
        with AutoreleasePool():
            send(retain(obj), "autorelease", restype=c_void_p)
            raise RuntimeError

    obj = make_object()
    with pytest.raises(RuntimeError):
        autorelease_then_fail(obj)
    assert retain_count(obj) == 1
    release(obj)


def test_object_releases_when_collected() -> None:
    ptr = make_object()
    wrapper = Object(ptr)
    assert retain_count(ptr) == 2
    del wrapper
    assert retain_count(ptr) == 1
    release(ptr)


def test_object_adopts_owned_pointer() -> None:
    ptr = retain(make_object())
    wrapper = Object(ptr, owned=True)
    assert retain_count(ptr) == 2
    del wrapper
    assert retain_count(ptr) == 1
    release(ptr)
