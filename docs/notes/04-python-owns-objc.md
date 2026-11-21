# 04 · A Python object that owns an Objective-C object

Sources:
- [Data model: `object.__del__`](https://docs.python.org/3/reference/datamodel.html#object.__del__)
- [`weakref.finalize`](https://docs.python.org/3/library/weakref.html#weakref.finalize)

## The problem

`send` returns a bare `int`. It doesn't know what it points to or who owns it,
and nothing stops a message to it after `release`. Every Metal object we will
get is one of these ints that we must release exactly once.

Python already counts references to its own objects and frees them at zero. If
a Python object owns the Objective-C object, Python's count decides when to
send `release`.

## `__del__` is not reliable

Python's data model says so, and points to the alternative:

> It is not guaranteed that `__del__()` methods are called for objects that
> still exist when the interpreter exits. `weakref.finalize` provides a
> straightforward way to register a cleanup function to be called when an
> object is garbage collected.

## `weakref.finalize`

> finalize(obj, func, *args, **kwargs) returns a callable finalizer object
> which will be called when obj is garbage collected. The first time the
> finalizer is called it evaluates func(*arg, **kwargs) and returns the
> result. After this the finalizer is dead, and calling it just returns None.

> When the program exits any remaining finalizers for which the atexit
> attribute is true will be run in reverse order of creation. By default
> atexit is true.

It runs at most once, and it also runs at exit.

## The callback must not hold `self`

The finalizer keeps `func` and `args` alive. If either refers to the object,
the object is never collected:

```py
weakref.finalize(self, release, ptr)       # ptr is an int: collected, count 2 → 1
weakref.finalize(self, self.close)         # bound method holds self: count stays 2
```

Both lines were run on an `NSObject`. The second one leaks, even after
`gc.collect()`.

## The class

```py
class Object:
    def __init__(self, ptr: int, *, owned: bool = False) -> None:
        self.ptr = ptr if owned else retain(ptr)
        weakref.finalize(self, release, ptr)
```

`owned` says which of note 03's rules applies:

| Where `ptr` came from                         | We own it? | `Object(...)`              |
| --------------------------------------------- | ---------- | -------------------------- |
| `new`, `alloc`, `copy`, `NS_RETURNS_RETAINED` | yes        | `Object(ptr, owned=True)`  |
| anything else, e.g. `stringWithUTF8String:`   | no         | `Object(ptr)`, retains it  |

### Ownership as tickets

Think of each ownership as a ticket. The object lives while anyone holds one;
`release` hands one back. The wrapper always hands back one ticket when Python
deletes it, so it must hold one. `owned` says where it comes from:

```py
ptr = send(NSObject, "new", restype=c_void_p)  # new gives you a ticket
obj = Object(ptr, owned=True)                  # you hand it to obj

ptr = from_str("hello")                        # the pool holds the only ticket
obj = Object(ptr)                              # obj retains: its own ticket
```

Without that `retain`, the wrapper would hand back the pool's ticket. When the
pool drains it hands it back too, and the object is freed twice. That's why
the default retains: a wrong `owned=True` crashes later, a wrong default only
leaks.

```py
string = Object(from_str("metalloid"))
string.send("length", restype=c_ulong)     # 9
del string                                 # release runs here
```

`del` doesn't call the finalizer directly:

> `del x` doesn't directly call `x.__del__()` - the former decrements the
> reference count for `x` by one, and the latter is only called when `x`'s
> reference count reaches zero.

In CPython that happens right away when nothing else refers to `x`.
