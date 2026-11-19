import ctypes
from ctypes import c_char_p, c_ulong, c_void_p

import pytest

from metalloid._objc import get_class, selector, send

ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")


def make_string(text: str) -> int:
    string: int = send(
        get_class("NSString"),
        "stringWithUTF8String:",
        text.encode(),
        restype=c_void_p,
        argtypes=(c_char_p,),
    )
    return string


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
