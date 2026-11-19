# 02 · Calling C from Python with ctypes

Source: [ctypes](https://docs.python.org/3/library/ctypes.html).

## Loading a library

```py
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
```

`objc.objc_getClass` is the exported function of that name. ctypes knows its
address, not its types: a C binary does not record them, only headers do.

## Undeclared types default to int

> By default functions are assumed to return the C int type.

```py
objc.objc_getClass(b"NSObject")             # 0x1e62918    wrong
objc.objc_getClass.restype = c_void_p
objc.objc_getClass(b"NSObject")             # 0x201e62918  right
```

An `int` is 4 bytes, a pointer is 8. The top bits were dropped without an error,
and using that pointer would crash.

### Why real pointers are above 4 GiB

Every 64-bit macOS executable starts with a `__PAGEZERO` segment covering
`0`–`0xFFFF_FFFF`, mapped with no access:

```sh
$ otool -l $(which python3) | grep -A5 __PAGEZERO
  segname __PAGEZERO
   vmaddr 0x0000000000000000      # starts at 0
   vmsize 0x0000000100000000      # 4 GiB long
  fileoff 0
 filesize 0
  maxprot 0x00000000              # no read, write, or execute
```

Nothing loads there, so every valid pointer is at least `0x1_0000_0000`. A
pointer truncated to 32 bits always lands in page zero and crashes instead of
silently reading other memory. It is also why a test can check
`get_class("NSObject") > 0xFFFF_FFFF` to catch a missing `restype`.

Rule: declare `argtypes` and `restype` for every C function, copied from its
header.

```py
# Class objc_getClass(const char *name);
objc.objc_getClass.argtypes = [c_char_p]
objc.objc_getClass.restype = c_void_p
```

`argtypes` also converts arguments, `bytes` to `const char *`, and rejects
the wrong kinds. It does not reject extra arguments: C allows variadic
functions, so ctypes passes them through.

## Selectors are interned

```py
objc.sel_registerName(b"length")            # 0x20fc0dc50
objc.sel_registerName(b"length")            # 0x20fc0dc50
```

Same name, same pointer, as note 01 says.

## Casting objc_msgSend with CFUNCTYPE

`CFUNCTYPE(restype, *argtypes)` creates a C function type. Calling the type with
an address gives a function with those types at that address. This is the cast
from note 01.

> Calling the type with an address binds the function to that address, returning a callable object.

```py
address = ctypes.cast(objc.objc_msgSend, c_void_p).value

# + (instancetype)stringWithUTF8String:(const char *)s
StringWithUTF8 = CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_char_p)
# - (NSUInteger)length
Length = CFUNCTYPE(c_ulong, c_void_p, c_void_p)

s = StringWithUTF8(address)(NSString, sel(b"stringWithUTF8String:"), b"metalloid")
Length(address)(s, sel(b"length"))          # 9
```

One address, a different function type per method signature. Every type starts
with the hidden `self` and `_cmd`.

## Binding by name

A prototype can also bind to an exported name, so ctypes finds the address:

```py
Length(("objc_msgSend", objc))              # same as Length(address)
```

Why not set `argtypes` on `objc.objc_msgSend` itself? It is a single object with
one `argtypes` and one `restype`; setting them for `length` breaks the next
`stringWithUTF8String:`. Each prototype call makes a separate function object
for the same C function, with its own fixed types.

## Type mapping

| Objective-C           | ctypes     | Why                              |
| --------------------- | ---------- | -------------------------------- |
| `id`, `Class`, `SEL`  | `c_void_p` | all pointers                     |
| `const char *`        | `c_char_p` | NUL-terminated bytes             |
| `NSUInteger`          | `c_ulong`  | 64-bit unsigned on Apple silicon |

## Tagged pointers

`s` was `0xb918efaa3537f451`, not an address. Short strings are tagged pointers:
the characters are stored in the pointer itself. Messaging works the same.

## Open questions

- Who frees the string `stringWithUTF8String:` returned?
