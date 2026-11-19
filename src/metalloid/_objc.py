"""Bridge to the Objective-C runtime through ctypes."""

import ctypes
from collections.abc import Callable, Generator
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_void_p
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _ctypes import _CData

type CType = type[_CData]

type Signature = tuple[CType | None, tuple[CType, ...]]
"""A method's return type (``None`` for void) and argument types."""

_libobjc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")

_libobjc.objc_getClass.argtypes = [c_char_p]
_libobjc.objc_getClass.restype = c_void_p

_libobjc.sel_registerName.argtypes = [c_char_p]
_libobjc.sel_registerName.restype = c_void_p

# Defines NSAutoreleasePool
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")

_senders: dict[Signature, Callable[..., Any]] = {}
"""``objc_msgSend`` cast to each signature seen so far."""


def get_class(name: str) -> int:
    """Return the class registered under ``name``.

    Raises:
        LookupError: If no loaded library defines the class.
    """
    cls: int | None = _libobjc.objc_getClass(name.encode())
    if cls is None:
        raise LookupError(f"Objective-C class {name!r} is not loaded")
    return cls


@cache
def selector(name: str) -> int:
    """Return the selector for the method ``name``, e.g. ``"length"``."""
    sel: int = _libobjc.sel_registerName(name.encode())
    return sel


def _msg_send(restype: CType | None, argtypes: tuple[CType, ...]) -> Callable[..., Any]:
    """Return ``objc_msgSend`` cast to ``restype (*)(id, SEL, *argtypes)``."""
    key: Signature = (restype, argtypes)
    if key not in _senders:
        prototype = CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)
        _senders[key] = prototype(("objc_msgSend", _libobjc))
    return _senders[key]


def send(
    receiver: int,
    name: str,
    *args: Any,
    restype: CType | None,
    argtypes: tuple[CType, ...] = (),
) -> Any:
    """Send the message ``name`` to ``receiver`` and return the result.

    Args:
        receiver: The object or class receiving the message.
        name: The selector name, e.g. ``"stringWithUTF8String:"``.
        *args: The message arguments, one per entry in ``argtypes``.
        restype: The method's return type, or ``None`` for ``void``.
        argtypes: The method's argument types, without ``self`` and ``_cmd``.

    Raises:
        TypeError: If the number of ``args``, ``argtypes``, and colons in
            ``name`` differ. ctypes passes extra arguments through silently.
    """
    arity = name.count(":")
    if len(argtypes) != arity or len(args) != arity:
        raise TypeError(
            f"{name!r} takes {arity} arguments, "
            f"got {len(args)} args and {len(argtypes)} argtypes"
        )
    return _msg_send(restype, argtypes)(receiver, selector(name), *args)


def retain(obj: int) -> int:
    """Take ownership of ``obj`` and return it."""
    result: int = send(obj, "retain", restype=c_void_p)
    return result


def release(obj: int) -> None:
    """Give up ownership of ``obj``, freeing it if no one else owns it."""
    send(obj, "release", restype=None)


@contextmanager
def autorelease_pool() -> Generator[None]:
    """Release the objects autoreleased inside the block when it exits.

    Without a pool, autoreleased objects are never released.
    """
    pool = send(get_class("NSAutoreleasePool"), "new", restype=c_void_p)
    try:
        yield
    finally:
        send(pool, "drain", restype=None)
