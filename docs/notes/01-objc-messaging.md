# 01 · Objective-C messaging

Sources:

- [Objective-C Runtime Programming Guide › Messaging](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjCRuntimeGuide/Articles/ocrtHowMessagingWorks.html)
- SDK headers: `objc/message.h`, `objc/runtime.h`, `Metal/MTLDevice.h`

## A method call is a message

```objc
[device newBufferWithLength:1024 options:0]
```

- **Receiver**: `device`, the object getting the message.
- **Selector**: `newBufferWithLength:options:`, the method name. One colon per argument.
- **Arguments**: `1024` and `0`.

## The compiler turns it into a C call

```c
objc_msgSend(device, @selector(newBufferWithLength:options:), 1024, 0);
```

`objc_msgSend` finds the method, calls it with the receiver and arguments, and
returns whatever the method returns.

## How the method is found

1. The object's `isa` pointer leads to its class.
2. The class maps selectors to implementations, checking a cache first, then its
   dispatch table.
3. If the selector is missing, the search repeats in the superclass, up to `NSObject`.

An implementation is a plain C function. Before the message's own arguments it
receives **two hidden ones**: the receiver (`self`) and the selector (`_cmd`). The
method's signature above is really:

```c
id impl(id self, SEL _cmd, NSUInteger length, MTLResourceOptions options);
```

## Selectors

A selector is an interned method name. `sel_registerName("length")` returns the
same pointer every time, so lookup compares pointers, not strings.

## Protocols

`MTLDevice` is a protocol, a list of messages an object promises to answer, not
a class. The class behind it is private to the GPU driver.

Sending a message only needs the object, so the class name never matters. Metal
hands out the first object through a plain C function:

```objc
id <MTLDevice> MTLCreateSystemDefaultDevice(void) NS_RETURNS_RETAINED;
```

## Calling objc_msgSend from outside Objective-C

The guide says never to call `objc_msgSend` directly. That advice is for
Objective-C code, where the compiler generates the call. A bridge has no
compiler, so it does the compiler's job. `message.h` says how:

> These functions must be cast to an appropriate function pointer type before
> being called.

The header declares `void objc_msgSend(void)` on purpose, so it cannot be called
without a cast. The type to cast to is the implementation's type: for each
method, call `objc_msgSend` as if it were exactly that function.

On arm64 there is no `objc_msgSend_stret`; the header marks it
`OBJC_ARM64_UNAVAILABLE`. One `objc_msgSend` covers every return type, structs
included.

## The bridge

| Function                       | Library | Purpose                            |
| ------------------------------ | ------- | ---------------------------------- |
| `MTLCreateSystemDefaultDevice` | Metal   | get the first object, the GPU      |
| `objc_getClass`                | libobjc | look up a class by name            |
| `sel_registerName`             | libobjc | turn a method name into a selector |
| `objc_msgSend`                 | libobjc | send a message                     |

## Open questions

- What does `NS_RETURNS_RETAINED` oblige us to do?
