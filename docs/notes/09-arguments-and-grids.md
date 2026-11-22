# 09 · Arguments and grids

Sources:
- [`setBytes:length:atIndex:`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:))
- [`newBufferWithBytes:length:options:`](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytes:length:options:))
- [Calculating threadgroup and grid sizes](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

## A scalar next to the arrays

BLAS, the standard library of vector and matrix routines, calls this operation
saxpy: **s**ingle-precision **a** times **x** **p**lus **y**.

```
y ← alpha · x + y
```

In NumPy it's `y += alpha * x`. As a kernel:

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
- **`constant`, not `device`.** Read-only memory, which every thread reads and
  none writes.

A buffer holding one float would fill the slot:

```py
alpha = device.new_buffer_from(np.array([2.5], dtype=np.float32))
```

A whole `MTLBuffer`, allocated by Metal, wrapped, and later released,
to hold 4 bytes. The encoder has a shortcut:

```objc
- (void)setBytes:(const void *)bytes
          length:(NSUInteger)length
         atIndex:(NSUInteger)index;
```

Reading it the usual way:

- **`void`**: nothing comes back, so there's nothing to own.
- **`bytes`**: where the value starts in our memory.
- **`length`**: how many bytes it takes.
- **`atIndex`**: the slot, the same numbering as `setBuffer:offset:atIndex:`.

The header calls it "Set the data (by copy) for a given buffer binding point."
Metal copies the bytes into the command buffer while encoding. The value is
captured right then, and our copy can go away immediately. Apple's page sets the
limit:

> This method only works for data smaller than 4 kilobytes that doesn't
> persist. Create an MTLBuffer instance if your data exceeds 4 KB, needs to
> persist on the GPU, or you access results on the CPU.

A number is a few bytes, and the kernel only reads it, so it fits.

### The bytes must match the type

`setBytes` copies whatever we give it. The kernel then reads as many bytes as
its type needs. A Python `float` is 8 bytes, a Metal `float` is 4:

```py
np.float64(2.5).tobytes()   # 00 00 00 00 00 00 04 40
np.float32(2.5).tobytes()   # 00 00 20 40
```

Sent as 8 bytes, `alpha` reads the first 4 of them and sees `0.0`. Nothing
reports an error.

So the value has to arrive with its size stated. A NumPy scalar does that:
`np.float32(2.5)` is exactly the 4 bytes a Metal `float` reads. Integers have
the same trap. NumPy's default `int64` is 8 bytes, Metal's `int` is 4:

| Metal   | NumPy        |
| ------- | ------------ |
| `float` | `np.float32` |
| `half`  | `np.float16` |
| `int`   | `np.int32`   |
| `uint`  | `np.uint32`  |
| `long`  | `np.int64`   |

## Arrays in memory

Memory is one line of bytes. A 2 × 3 array has to be laid along it somehow,
and NumPy's default is **C order**, also called row-major: the first row, then
the second.

```
a = [[0, 1, 2],
     [3, 4, 5]]

memory:   0 1 2  3 4 5
         └row 0┘└row 1┘
```

NumPy records the layout as **strides**: how many bytes to step to move one
place along each axis. For `a` in `float32`, the strides are `(12, 4)`. The
next row is 12 bytes on, three floats, and the next column is 4 bytes on, one
float.

A transpose moves no data. `a.T` is a new view of the same memory with the
strides swapped, `(4, 12)`:

```
a.T = [[0, 3],
       [1, 4],
       [2, 5]]

memory, unchanged:  0 1 2 3 4 5
```

Row 0 of `a.T` is `0, 3`, but those are 12 bytes apart. Read in a straight
line, the memory is still `a`'s rows, not `a.T`'s. NumPy reports this as
`a.T.flags.c_contiguous == False`.

A kernel only sees a straight line of bytes. It has no strides, so it can only
read an array that is in C order.

### Copying an array into a buffer

A second way to make a buffer copies existing bytes in:

```objc
- (nullable id <MTLBuffer>)newBufferWithBytes:(const void *)pointer
                                       length:(NSUInteger)length
                                      options:(MTLResourceOptions)options;
```

The header calls it "Create a buffer by allocating new memory and specifing the
initial contents to be copied into it." It starts with `new`, so we own the
result, as with `newBufferWithLength:options:` in note 06.

`array.ctypes.data` is the address of an array's first byte, and `length` is
its size in bytes. Metal copies that many bytes in a straight line, so an array
that isn't in C order has to be copied into C order first.

Because the buffer holds a copy, it no longer depends on the array. Changing
the array afterwards doesn't change the buffer.

### The size of a buffer

A buffer's size is its elements times the bytes per element:
`(1000, 1000)` of `float32` is 4,000,000 bytes. Metal returns nil for a size of
0, from an empty array such as `np.zeros((3, 0))`, and for anything over
`maxBufferLength` (note 06).

Metal still knows only the size. Which dtype and shape the bytes have is ours
to remember, so a metalloid `Buffer` records them when it's made.

## Grids with rows and columns

`dispatchThreads` has always taken a width, a height, and a depth. Filling them
in gives each thread a 2D or 3D position, which the kernel reads as a vector:

```metal
uint2 p [[thread_position_in_grid]]     // p.x, p.y
uint2 n [[threads_per_grid]]            // the grid's width and height
```

Metal puts `x`, the width, first. `x` runs along a row and `y` counts rows. So
an array of shape `(rows, cols)` needs a grid of `(cols, rows)`: the two orders
are reversed.

To find its element, a thread uses C order. Element `[y, x]` comes after `y`
full rows of `n.x` elements, plus `x` more:

```
index = p.y * n.x + p.x
```

For `p = (1, 1)` in a 3-wide grid, that's `1 * 3 + 1 = 4`, the `4` in `a`
above. Three dimensions add `p.z` and a depth. An array of shape
`(depth, rows, cols)` needs a grid of `(cols, rows, depth)`.

### The threadgroup's shape

A threadgroup gets a shape too. Apple's recipe for the largest one:

```swift
let w = pipelineState.threadExecutionWidth
let h = pipelineState.maxTotalThreadsPerThreadgroup / w
let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
```

One SIMD-group wide and as tall as the limit allows: 32 × 32 here. The width
isn't arbitrary. The 32 threads of a SIMD-group then lie along one row, so they
read 32 neighbouring elements.

A threadgroup can be larger than the whole grid. A 5 × 3 grid with 32 × 32
threadgroups runs correctly, even with Metal's API validation on
(`MTL_DEBUG_LAYER=1`), which aborts on any misuse. `dispatchThreads` provides
partial threadgroups (note 08), and a group that overhangs the grid is simply
partial.

## An encoder has to end

The encoder makes one demand note 08 didn't test: once created, it has to reach
`endEncoding`. An encoder freed without it isn't an error Python can catch:

```
-[_MTLCommandEncoder dealloc]:134: failed assertion `Command encoder released without endEncoding'
```

Metal aborts the process. Suppose an argument turns out to be wrong halfway
through encoding and Python raises. The encoder is left to the pool, the pool
drains, and the drain kills Python. So everything that can fail has to be
checked before the encoder exists.

## Using it

Scaling a matrix in place:

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

The shape is `(2, 3)`, two rows of three, so the grid is `(3, 2)`.

## Open questions

- Every thread still works alone. How do the threads of one threadgroup share
  work, say to sum an array?
