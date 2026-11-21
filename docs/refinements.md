# Refinements

## Errors

- [x] One base exception, `MetalError`, for every failure from Metal.
      `CompileError`, `KernelNotFoundError` (also a `LookupError`), and
      `AllocationError` (also a `MemoryError`) derive from it. Bad arguments
      stay `ValueError` and `TypeError`.
- [ ] `new_buffer(0)` raises `AllocationError`. It's a bad argument, not a lack of
      memory: check `nbytes` first and raise `ValueError` with the valid range.
- [ ] Objective-C exceptions abort the process and can't be caught (an
      unrecognized selector, mutating an immutable string). Consider a debug
      mode that checks `respondsToSelector:` before sending.
- [x] `to_str(nil)` fails with `'NoneType' has no attribute 'decode'`. It
      raises `ValueError`, as does a string with no UTF-8 form.
- [ ] Messaging nil silently returns 0 or `None`, which hides bugs. The type
      checker now rejects a receiver that may be `None`, and `Object(None)`,
      but nothing checks at runtime.
- [ ] `Buffer.numpy()` passes on NumPy's errors when the size isn't a multiple
      of the dtype or the shape doesn't fit. Say it plainly, e.g. "16 bytes
      can't hold shape (3, 2) of float32".

## API

- [x] Public API: nothing is exported from `metalloid`, so
      `from metalloid import Device` fails. `Device` and the errors are
      exported. `Buffer`, `Library`, and `Kernel` aren't yet, so they can't be
      used in annotations without importing `metalloid._metal`.
- [x] `__repr__` for `Device`, `Buffer`, `Library`, and `Kernel`, e.g.
      `<Device 'Apple M1 Pro'>`, `<Buffer 16 bytes>`, `<Library ['add']>`,
      `<Kernel 'add'>`.
- [ ] `Device()` opens the GPU on every call. Consider a cached default device.
- [ ] Buffers can only be created empty. Add creating one from a NumPy array,
      or with a dtype and shape directly.
- [ ] `Library` reaches into `Device._obj`. Classes shouldn't use each other's
      private attributes.
- [ ] `Buffer`, `Library`, and `Kernel` have public constructors but should
      only be made by `Device` and `Library`.

## Correctness

- [ ] Autorelease pools: creation calls have one, property reads don't. Decide
      on a pool in every public method (note 05).
- [ ] Nothing stops the CPU reading a buffer while the GPU writes it (note 06).
      `run` waits, so this holds today. Revisit if `run` stops waiting
      (note 08).
- [ ] `run` doesn't check that the buffers hold `threads` elements, or that
      every slot the kernel uses is bound. The kernel then reads and writes
      past the end, with no error.
- [x] `make_string` in `tests/test_objc.py` runs without a pool, leaks, and
      duplicates `from_str`. The tests use `from_str` in a pool.
- [ ] `from_str` stops at a NUL character: `from_str("a\0b")` is `"a"`. Raise
      `ValueError` instead, as Python's own C-facing functions do
      (`os.stat("a\0b")`: "embedded null character").

## Documentation

- [ ] Link Apple's Metal documentation in docstrings, e.g. `Device.new_buffer`
      to `newBufferWithLength:options:`, so readers can go from metalloid to
      the API it wraps.

## Typing

- [x] `send` returns `Any`, so callers annotate every result
      (`size: int = ...`). `send` is overloaded on `restype` instead (note
      02), and `Object.send` is gone so there's one place to type. Plain
      `CDLL` functions still return `Any`.

## Performance

- [ ] Measure the cost of `send`: the arity check, the cache lookup, and the
      ctypes call, against the time a whole `run` takes.
- [ ] Each pool is two extra messages through `NSAutoreleasePool`.
      `objc_autoreleasePoolPush` and `objc_autoreleasePoolPop` are faster but
      aren't in the public headers.

## Project

- [ ] Importing on Linux or an Intel Mac should fail with a clear message, not
      a ctypes error.
