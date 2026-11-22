# 09 · Arguments and grids

Sources:
- [`newBufferWithBytes:length:options:`](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytes:length:options:))
- [`setBytes:length:atIndex:`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:))
- [Calculating threadgroup and grid sizes](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

Note 08 ran one kind of kernel: float arrays in, one row of threads. This note
adds what real kernels need: buffers that know what they hold, single numbers
next to the arrays, and grids in two and three dimensions.

## Buffers that know their shape

Metal knows a buffer's size and nothing else (note 06). Until now `numpy()`
asked for the dtype and shape on every call. A `Buffer` now records them when
it's made, as NumPy does:

```py
device.new_buffer((2, 3), np.float32)       # zeroed, like np.zeros
device.new_buffer_from(array)               # a copy of array
```

`new_buffer_from` uses the other way Metal makes a buffer:

```objc
- (nullable id <MTLBuffer>)newBufferWithBytes:(const void *)pointer
                                       length:(NSUInteger)length
                                      options:(MTLResourceOptions)options;
```

The header calls it "Create a buffer by allocating new memory and specifing the
initial contents to be copied into it."

`array.ctypes.data` is the address of the array's bytes. One call allocates and copies,
where `new_buffer` plus a NumPy copy writes the memory twice, zeros first.

Metal copies one block of bytes, so the array must be one block in C order,
rows one after another. A transposed array isn't, and `new_buffer_from` copies
it into C order (row-major) first. The copy also means changes to `array`
afterwards don't reach the buffer.

Two kinds of input are rejected:

- **A list**, `TypeError`. NumPy would make it `float64` or `int64`, which
  rarely match the kernel's `float` or `int`. See below for why sizes matter.
- **`dtype=object`**, `TypeError`. Its elements are pointers to Python
  objects, meaningless to the GPU.

A buffer's size is its elements times the dtype's bytes:
`new_buffer((1000, 1000), np.float32)` is 4,000,000 bytes.

Metal returns nil for a size of 0, as from `np.zeros((3, 0))`, and for anything over
`max_buffer_length`. That used to surface as `AllocationError`, but it's a bad argument,
not a lack of memory, so both constructors check the size first and raise `ValueError`.

## Scalar arguments

`y = alpha * x + y`, known as saxpy, needs one number next to two arrays:

```metal
kernel void saxpy(constant float &alpha [[buffer(0)]],
                  const device float *x [[buffer(1)]],
                  device float *y [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
{
    y[i] = alpha * x[i] + y[i];
}
```

`alpha` takes a buffer slot like the arrays do, with two differences:

- **`&`, not `*`.** A reference to one `float`, not a pointer to many.
- **`constant`, not `device`.** The read-only address space, for values every
  thread reads and none writes.

Without a shortcut, `alpha` would need a buffer of its own:

```py
alpha = device.new_buffer_from(np.array([2.5], np.float32))
```

That's a whole `MTLBuffer`, allocated by Metal, wrapped, and later released,
to hold 4 bytes. The shortcut is to hand the encoder the bytes, and it copies
them into the command buffer itself:

```objc
- (void)setBytes:(const void *)bytes
          length:(NSUInteger)length
         atIndex:(NSUInteger)index;
```

The header calls it "Set the data (by copy) for a given buffer binding point."
Metal copies the bytes into the command buffer while encoding, so there's no
`MTLBuffer`, and nothing on our side to keep alive. Apple's page sets the
limit:

> This method only works for data smaller than 4 kilobytes that doesn't
> persist. Create an MTLBuffer instance if your data exceeds 4 KB, needs to
> persist on the GPU, or you access results on the CPU.

A scalar is a few bytes, and the kernel only reads it, so it fits.

### The size must match

The kernel reads as many bytes as its type needs, whatever we sent. A Python
`float` is 8 bytes, a Metal `float` is 4:

```py
np.float64(2.5).tobytes()   # 00 00 00 00 00 00 04 40
np.float32(2.5).tobytes()   # 00 00 20 40
```

Sent as 8 bytes, `alpha` reads the first 4 of them: `0.0`, with no error.

So `run` takes NumPy scalars, whose dtype states their size, and raises
`TypeError` for a plain `float` or `int`. The same table applies to buffers:

| Metal   | NumPy        |
| ------- | ------------ |
| `float` | `np.float32` |
| `half`  | `np.float16` |
| `int`   | `np.int32`   |
| `uint`  | `np.uint32`  |

## Grids in two and three dimensions

`dispatchThreads` always took a width, height, and depth, we passed 1 for the
last two. `threads=(width, height)` fills them in, and the kernel declares its
position as a vector:

```metal
uint2 p [[thread_position_in_grid]]     // p.x, p.y
uint2 n [[threads_per_grid]]            // the grid's width and height
```

`x` comes first, and `x` is the column. An array of shape `(rows, cols)` takes
`threads=(cols, rows)`, and the thread at `p` owns element `p.y * n.x + p.x`.
Three dimensions add `p.z`, with shape `(depth, rows, cols)` and
`threads=(cols, rows, depth)`.

The threadgroup gets a shape too. Apple's recipe for the largest one:

```swift
let w = pipelineState.threadExecutionWidth
let h = pipelineState.maxTotalThreadsPerThreadgroup / w
let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
```

One SIMD-group wide, as tall as the limit allows: 32 × 32 here. One row of
threads is still `(1024, 1, 1)`.

A threadgroup larger than the whole grid is fine. A 5 × 3 grid with 32 × 32
threadgroups ran correctly with Metal's API validation on
(`MTL_DEBUG_LAYER=1`), which aborts on any misuse. So `run` doesn't shrink the
group to fit a small grid.

## Check before encoding

`run` checks every argument and the grid before it makes the encoder. An
encoder freed before `endEncoding` isn't an error Python can catch:

```
-[_MTLCommandEncoder dealloc]:134: failed assertion `Command encoder released without endEncoding'
```

Metal aborts the process. A `TypeError` raised halfway through encoding would
leave the encoder to the pool, and the pool's drain would kill Python.

## Using it

```metal
kernel void scale(constant float &alpha [[buffer(0)]],
                  device float *m [[buffer(1)]],
                  uint2 p [[thread_position_in_grid]],
                  uint2 n [[threads_per_grid]])
{
    m[p.y * n.x + p.x] *= alpha;
}
```

```py
scale = device.compile(source).kernel("scale")
m = device.new_buffer_from(np.arange(6, dtype=np.float32).reshape(2, 3))
m                                       # <Buffer (2, 3) float32>

device.run(scale, [np.float32(10), m], threads=(3, 2))

m.numpy()                               # array([[ 0., 10., 20.],
                                        #        [30., 40., 50.]], dtype=float32)
```

## Open questions

- Every thread still works alone. How do the threads of one threadgroup share
  work, say to sum an array?
