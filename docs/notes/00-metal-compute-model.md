# 00 · The Metal compute model

Source: [Performing Calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu).

## The GPU runs one function many times

A CPU runs a few threads fast. A GPU runs one small function, a **kernel**,
across thousands of threads at once, each on its own slice of data.

Adding two arrays of a million floats is a million threads, where thread `i`
computes `c[i] = a[i] + b[i]`.

Kernels are written in **Metal Shading Language (MSL)** and compiled while the program runs.

## Unified memory model

A discrete GPU has its own memory, so data is copied over before a kernel runs
and copied back after.

Apple silicon CPUs and GPUs share the same RAM, so a buffer the GPU writes can be
read by the CPU directly.

metalloid builds on this: NumPy reads and writes GPU buffers in place, with no copies.

## The objects

```
MTLDevice                             the GPU; creates everything else
 ├─ MTLLibrary                        compiled MSL source
 │   └─ MTLFunction                   one kernel in that library
 │       └─ MTLComputePipelineState   the kernel, ready to run
 ├─ MTLBuffer                         memory the kernel reads and writes
 └─ MTLCommandQueue                   where work is submitted
     └─ MTLCommandBuffer              one batch of work
         └─ MTLComputeCommandEncoder  records pipeline + buffers + thread count
```

## Record, commit, wait

A kernel is not called like a function.

1. **Record**: An encoder writes down which pipeline to use, which buffers to
   bind, and how many threads to launch.
2. **Commit**: The command buffer is handed to the GPU.
3. **Wait**: The GPU runs it asynchronously. The CPU blocks only when it needs
   the result.

## Open questions

- What is pipeline and buffer in this context?
