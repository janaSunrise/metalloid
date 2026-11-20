# Reading Metal's headers

Metal is closed source. What Apple ships instead are its headers: every class,
method, type, and constant, most with a doc comment above. They are the exact
declarations the compiler uses, so when a web page and a header disagree,
trust the header.

## Apple's documentation

[Apple's Metal documentation](https://developer.apple.com/documentation/metal)
covers the same API, with overviews and requirements the headers leave out.
Read both: the header for exact types, the page for how to use them.

## Reading a method

```objc
/*!
 @method newBufferWithLength:options:
 @brief Create a buffer by allocating new memory.
 */
- (nullable id <MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
```

| Part                           | Meaning                                         |
| ------------------------------ | ----------------------------------------------- |
| `-`                            | an instance method, sent to an object; `+` is sent to a class |
| `nullable`                     | may return nil, which ctypes gives as `None`    |
| `id <MTLBuffer>`               | an object conforming to `MTLBuffer`: a pointer  |
| `newBufferWithLength:options:` | the selector, the keywords joined, colons kept  |
| `(NSUInteger)length`           | the first argument: its type, then its name     |
| `(MTLResourceOptions)options`  | the second argument, a typedef to look up       |
| the name starts with `new`     | we own the result (note 03)                     |

Doc comments come in two styles: older `/*! @brief ... */` blocks, as here,
and newer `///` lines. Both sit directly above the declaration.

Each colon is one argument. Put together in Python:

```py
ptr = send(
    device,
    "newBufferWithLength:options:",
    1024,
    MTLResourceStorageModeShared,
    restype=c_void_p,
    argtypes=(c_ulong, c_ulong),
)
if ptr is None:
    ...                                  # nullable: allocation failed
buffer = Object(ptr, owned=True)         # new: ours
```

## Reading a property

```objc
@property (readonly) BOOL hasUnifiedMemory;
```

A property declares a getter method with the same name, so this one is read
with `send(device, "hasUnifiedMemory", restype=c_bool)`. Without `readonly`
there is also a setter, `setHasUnifiedMemory:`.

## Reading a type

Follow every typedef down to a C type before choosing the ctypes type.

`NS_ENUM` is a set of named integers. The first argument is the storage type:

```objc
typedef NS_ENUM(NSUInteger, MTLStorageMode) {
    MTLStorageModeShared  = 0,
    MTLStorageModePrivate = 2,
    ...
};
```

`NS_OPTIONS` is the same, but the values are bit flags combined with `|`.
They are often built from other constants:

```objc
typedef NS_OPTIONS(NSUInteger, MTLResourceOptions) {
    MTLResourceStorageModeShared  = MTLStorageModeShared  << MTLResourceStorageModeShift,
    MTLResourceStorageModePrivate = MTLStorageModePrivate << MTLResourceStorageModeShift,
    ...
};
```

Follow each name. `MTLResourceStorageModeShift` is `#define`d as `4` in the
same header, so `MTLResourceStorageModeShared` is `0 << 4 = 0` and
`MTLResourceStorageModePrivate` is `2 << 4 = 32`.

A struct lists its fields in memory order, and a `ctypes.Structure` repeats
them in the same order (note 05):

```objc
typedef struct { NSUInteger width, height, depth; } MTLSize;
```

## Annotations

Macros after a declaration add facts the types don't carry:

| Annotation                                  | Meaning                                     |
| ------------------------------------------- | ------------------------------------------- |
| `MTL_EXTERN`                                | a C function, called directly, not sent     |
| `NS_RETURNS_RETAINED`                       | the caller owns the result                  |
| `API_AVAILABLE(macos(10.15))`               | added in macOS 10.15                        |
| `API_UNAVAILABLE(ios)`                      | missing on iOS                              |
| `API_DEPRECATED_WITH_REPLACEMENT("X", ...)` | still works; use `X` instead                |
| `OBJC_ARC_UNAVAILABLE`                      | ARC code can't call it; ctypes can          |
