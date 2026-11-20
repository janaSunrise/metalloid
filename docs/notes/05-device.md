# 05 · The device

Sources:
- [`MTLCreateSystemDefaultDevice()`](https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice())
- SDK headers `Metal/MTLDevice.h`, `Metal/MTLTypes.h`, `objc/objc.h`.
- [Procedure Call Standard for the Arm 64-bit Architecture](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)

## Getting the GPU

The first Metal call is a C function, not a message:

```objc
/*!
 @brief Returns a reference to the preferred system default Metal device.
 */
id <MTLDevice> __nullable MTLCreateSystemDefaultDevice(void) NS_RETURNS_RETAINED;
```

Every part of that line maps to Python:

| Header                | Python                                         |
| --------------------- | ---------------------------------------------- |
| `id <MTLDevice>`      | `restype = c_void_p`, without it: `0xb7fdb0`, not `0x100b7fdb0` |
| `(void)`              | `argtypes = []`                                |
| `__nullable`          | may return `None`, `Device()` raises `RuntimeError` |
| `NS_RETURNS_RETAINED` | we own it: `Object(ptr, owned=True)`           |

Every call returns the same device, and every call hands us one more ticket.

Apple's page adds a requirement the header doesn't mention:

> In macOS, in order for the system to provide a default Metal device object,
> you need to link to the Core Graphics framework. You usually need to do this
> explicitly if you're writing apps that don't use graphics by default, such as
> command line tools.

Python already had Core Graphics loaded when it started on this machine, but
nothing promises that, so `_metal.py` loads it explicitly.

## Call Metal inside a pool

Any Metal or Foundation call may autorelease objects, and the headers don't
say which ones do. So follow Apple's rule from note 03: Metal calls belong
inside an autorelease pool. `Device()` creates the device inside
`autorelease_pool()`.

To see why, run the call without a pool. `OBJC_DEBUG_MISSING_POOLS` makes the
runtime print a line for every object that leaks:

```sh
$ OBJC_DEBUG_MISSING_POOLS=YES python -c "
import ctypes
metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
metal.MTLCreateSystemDefaultDevice()
"
objc[...]: MISSING POOLS: ... Object ... of class NSUserDefaults autoreleased with no pool in place - just leaking
```

Counting the leaked objects by class:

| Class                        | Leaked |
| ---------------------------- | ------ |
| `NSUserDefaults`             | 3      |
| `NSBundle`                   | 2      |
| `__NSDictionaryM`            | 2      |
| `__NSSingleEntryDictionaryI` | 1      |

None of these are ours. Metal reads its settings and bundles while it sets up,
and autoreleases what it used.

## Properties are messages

```objc
@property (readonly) BOOL hasUnifiedMemory;
```

A property declares a getter method with the same name, so reading it is
`send(device, "hasUnifiedMemory", restype=...)`. The type comes from the
declaration:

| Property                       | Header type  | ctypes          |
| ------------------------------ | ------------ | --------------- |
| `name`                         | `NSString *` | `c_void_p`, then `to_str` |
| `hasUnifiedMemory`             | `BOOL`       | `c_bool`        |
| `recommendedMaxWorkingSetSize` | `uint64_t`   | `c_uint64`      |
| `maxThreadsPerThreadgroup`     | `MTLSize`    | a `Structure`   |

`to_str` sends `UTF8String` and decodes the bytes. `c_char_p` copies the C
string as soon as it returns, so the Python `str` doesn't depend on the
`NSString` staying alive.

`BOOL` depends on the platform. `objc.h` makes it `bool` when the compiler
defines `__OBJC_BOOL_IS_BOOL`, and `signed char` otherwise:

```sh
$ clang -dM -E -x objective-c -arch arm64 /dev/null | grep OBJC_BOOL
#define __OBJC_BOOL_IS_BOOL 1
```

On Apple silicon it is `bool`, so `c_bool`.

## Returning a struct

```objc
typedef struct {
    NSUInteger width, height, depth;
} MTLSize;
```

A `ctypes.Structure` describes the same layout, field by field:

```py
class _MTLSize(Structure):
    _fields_ = (("width", c_ulong), ("height", c_ulong), ("depth", c_ulong))
```

Three 8-byte fields make 24 bytes, too large for return registers. The Arm
procedure call standard returns a struct over 16 bytes in memory: the caller
reserves the space and passes its address in register `x8`.

`x8` is not an argument register, so `self` and `_cmd` stay in `x0` and `x1`
and `objc_msgSend` works unchanged. On Intel the hidden address took the first
argument slot, which is why it needed `objc_msgSend_stret` (note 01). ctypes
sets up `x8` itself when `restype` is a `Structure`.

## This machine

```py
device = Device()
device.name                                # 'Apple M1 Pro'
device.has_unified_memory                  # True
device.recommended_max_working_set_size    # 12713115648 bytes, 11.8 GiB of 16 GiB RAM
device.max_threads_per_threadgroup         # (1024, 1024, 1024)
```

## Open questions

- `maxBufferLength` is 8.9 GiB, smaller than the 11.8 GiB working set. What
  limits a single buffer?
