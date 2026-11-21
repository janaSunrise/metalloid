"""Bridge to the Objective-C runtime through ctypes."""

import ctypes
import weakref
from collections.abc import Callable
from ctypes import CFUNCTYPE, Structure, c_char_p, c_void_p
from functools import cache
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from _ctypes import _CData, _SimpleCData

type CType = type[_CData]

type SimpleCType[T] = type[_SimpleCData[T]]
"""A fundamental type, such as ``c_ulong``, that ctypes returns as a ``T``."""

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


@overload
def send[T](
    receiver: int,
    name: str,
    *args: Any,
    restype: SimpleCType[T],
    argtypes: tuple[CType, ...] = (),
) -> T: ...


@overload
def send[S: Structure](
    receiver: int,
    name: str,
    *args: Any,
    restype: type[S],
    argtypes: tuple[CType, ...] = (),
) -> S: ...


@overload
def send(
    receiver: int,
    name: str,
    *args: Any,
    restype: None,
    argtypes: tuple[CType, ...] = (),
) -> None: ...


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

    Returns:
        The result as ctypes converts it: an ``int`` for ``c_ulong``, an ``int``
        or ``None`` (nil) for ``c_void_p``, an instance for a ``Structure``.

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
    """Take ownership of ``obj`` and return it, as ``retain`` returns ``self``."""
    send(obj, "retain", restype=c_void_p)
    return obj


def release(obj: int) -> None:
    """Give up ownership of ``obj``, freeing it if no one else owns it."""
    send(obj, "release", restype=None)


def from_str(text: str) -> int:
    """Return a new ``NSString`` holding ``text``.

    The string is autoreleased: use it inside `AutoreleasePool`.

    Raises:
        ValueError: If Foundation can't make an ``NSString`` from ``text``.
    """
    string = send(
        get_class("NSString"),
        "stringWithUTF8String:",
        text.encode(),
        restype=c_void_p,
        argtypes=(c_char_p,),
    )
    if string is None:
        raise ValueError("stringWithUTF8String: returned nil")
    return string


def to_str(string: int | None) -> str:
    """Return the text of the ``NSString`` at ``string``.

    Raises:
        ValueError: If ``string`` is nil, or has no UTF-8 form because it
            holds half of a surrogate pair.
    """
    if string is None:
        raise ValueError("expected an NSString, got nil")
    text = send(string, "UTF8String", restype=c_char_p)
    if text is None:
        raise ValueError("the NSString has no UTF-8 form")
    return text.decode()


_NSAutoreleasePool = get_class("NSAutoreleasePool")


class AutoreleasePool:
    """Release the objects autoreleased inside the ``with`` block when it exits.

    Without a pool, autoreleased objects are never released.
    """

    __slots__ = ("_pool",)

    def __enter__(self) -> None:
        pool = send(_NSAutoreleasePool, "new", restype=c_void_p)
        if pool is None:
            raise MemoryError("Foundation could not create an autorelease pool")
        self._pool = pool

    def __exit__(self, *exc_info: object) -> None:
        send(self._pool, "drain", restype=None)


def _release_in_pool(obj: int) -> None:
    """Release ``obj`` inside a pool, since finalizers run outside ours.

    Freeing an object can autorelease others: a freed ``MTLBuffer``
    autoreleases its device.
    """
    with AutoreleasePool():
        release(obj)


class Object:
    """An Objective-C object kept alive by this Python object.

    The object is released when this wrapper is garbage collected.

    Attributes:
        ptr: The object's address, or a tagged pointer.
    """

    def __init__(self, ptr: int, *, owned: bool = False) -> None:
        """Take ownership of the object at ``ptr``.

        Args:
            ptr: The object's address, or a tagged pointer.
            owned: Whether the caller already owns ``ptr``, as with ``new``
                or ``NS_RETURNS_RETAINED``. That ownership passes to this
                wrapper. Otherwise the wrapper retains ``ptr``.
        """
        self.ptr = ptr if owned else retain(ptr)
        weakref.finalize(self, _release_in_pool, ptr)
