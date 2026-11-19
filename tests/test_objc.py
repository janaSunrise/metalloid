from ctypes import c_char_p, c_ulong, c_void_p

import pytest

from metalloid._objc import (
    autorelease_pool,
    get_class,
    release,
    retain,
    selector,
    send,
)


def make_string(text: str) -> int:
    string: int = send(
        get_class("NSString"),
        "stringWithUTF8String:",
        text.encode(),
        restype=c_void_p,
        argtypes=(c_char_p,),
    )
    return string


def make_object() -> int:
    obj: int = send(get_class("NSObject"), "new", restype=c_void_p)
    return obj


def retain_count(obj: int) -> int:
    count: int = send(obj, "retainCount", restype=c_ulong)
    return count


def test_get_class_returns_full_pointer() -> None:
    assert get_class("NSObject") > 0xFFFF_FFFF


def test_get_class_unknown_raises() -> None:
    with pytest.raises(LookupError, match="NoSuchClass"):
        get_class("NoSuchClass")


def test_selectors_are_interned() -> None:
    assert selector("length") == selector("length")
    assert selector("length") != selector("hash")


def test_send_returns_integer() -> None:
    assert send(make_string("metalloid"), "length", restype=c_ulong) == 9


def test_send_round_trips_string() -> None:
    string = make_string("héllo")
    assert send(string, "UTF8String", restype=c_char_p) == "héllo".encode()


def test_send_rejects_wrong_argument_count() -> None:
    with pytest.raises(TypeError):
        send(make_string("x"), "length", 1, restype=c_ulong)


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
    with autorelease_pool():
        send(retain(obj), "autorelease", restype=c_void_p)
        assert retain_count(obj) == 2
    assert retain_count(obj) == 1
    release(obj)


def test_pool_drains_on_exception() -> None:
    def autorelease_then_fail(obj: int) -> None:
        with autorelease_pool():
            send(retain(obj), "autorelease", restype=c_void_p)
            raise RuntimeError

    obj = make_object()
    with pytest.raises(RuntimeError):
        autorelease_then_fail(obj)
    assert retain_count(obj) == 1
    release(obj)
