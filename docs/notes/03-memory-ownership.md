# 03 · Who owns an object

Sources:
- [Memory Management Policy](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmRules.html),
- [Using Autorelease Pool Blocks](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html),
- SDK headers `objc/NSObject.h`, `Foundation/NSAutoreleasePool.h`, `Metal/MTLDevice.h`.

## Reference counting

Every object carries a count of its owners. `retain` adds one, `release`
removes one, and at zero the object is freed. Swift and ARC (Automatic Reference Counting)
insert these calls for you. ctypes does not, so we make them ourselves.

## The rules

- You own any object you create
- You can take ownership of an object using retain
- When you no longer need it, you must relinquish ownership of an object you own
- You must not relinquish ownership of an object you do not own

"Create" means a method whose name begins with `alloc`, `new`, `copy`, or
`mutableCopy`. The name alone tells you whether you own the result.

```py
obj = send(NSObject, "new", restype=c_void_p)   # count 1, ours
retain(obj)                                     # count 2
release(obj)                                    # count 1
release(obj)                                    # count 0, freed
```

Release too few times and the object leaks. Release too many and it is freed
while still in use, and the next message to it crashes.

## Autorelease is a deferred release

`stringWithUTF8String:` does not start with `alloc`, `new`, or `copy`, so the
caller does not own the string. But someone has to release it. The method
sends it `autorelease`:

> You use `autorelease` when you need to send a deferred `release` message

The release happens when the current autorelease pool ends:

> At the end of the autorelease pool block, objects that received an
> `autorelease` message within the block are sent a `release` message.

```py
pool = send(NSAutoreleasePool, "new", restype=c_void_p)
s = send(NSString, "stringWithUTF8String:", b"a string too long to be tagged", ...)
retain(s)                                       # count 2: the pool's and ours
send(pool, "drain", restype=None)               # count 1: ours
```

## No pool, no release

A Cocoa app gets pools from its run loop. A Python script is a
"Foundation-only program", so it has none:

> If you are writing a Foundation-only program or if you detach a thread, you
> need to create your own autorelease pool block.

The runtime can report what happens otherwise:

```sh
$ OBJC_DEBUG_MISSING_POOLS=YES python script.py
objc[20497]: MISSING POOLS: (0x201e6ac40) Object 0x102759c20 of class NSObject
autoreleased with no pool in place - just leaking
```

Each thread has its own stack of pools, so a pool must be made on the thread
that autoreleases.

## Making a pool from Python

`@autoreleasepool { }` is compiler syntax, out of reach from ctypes. The
`NSAutoreleasePool` class does the same through messages:

```objc
@interface NSAutoreleasePool : NSObject
- (void)drain;
@end
```

`new` pushes a pool, `drain` releases its objects and the pool itself. In
`_objc.py` this is a context manager, so the drain also runs when the block
raises:

```py
with autorelease_pool():
    s = make_string("...")      # released when the block exits
```

The class lives in Foundation, not libobjc, so `_objc.py` loads Foundation.

## NS_RETURNS_RETAINED

Functions like `MTLCreateSystemDefaultDevice` are not methods, so the naming
rule cannot apply. The header states ownership instead:

```objc
id <MTLDevice> MTLCreateSystemDefaultDevice(void) NS_RETURNS_RETAINED;
```

`NS_RETURNS_RETAINED` is `__attribute__((ns_returns_retained))`: the caller
owns the result and must release it. The attribute does the job of `new` in
the name.

## retainCount is not your count

`retainCount` counts every owner, not just us, so the tests only read it on
`NSObject`s they made themselves:

```py
retainCount(MTLCreateSystemDefaultDevice())     # 5: Metal holds its own references
retainCount(tagged_string)                      # 18446744073709551615, NSUIntegerMax
```

A tagged pointer is not allocated, so retain and release do nothing to it.

## Open questions

- Python frees its objects by reference counting too. Can a Python object own
  an Objective-C one, so `release` runs when it is garbage collected?
