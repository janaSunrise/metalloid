# metalloid

Write and dispatch [Metal](https://developer.apple.com/metal/) compute kernels
on Apple silicon GPUs from Python.

metalloid is built from first principles: it binds to the Objective-C runtime
and `Metal.framework`. Apple silicon has unified memory, so GPU buffers can be shared
with NumPy without copying.

## Requirements

- macOS on Apple silicon (M1 or newer)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Example

```py
import numpy as np
from metalloid import Device

ADD = """
kernel void add(const device float *a   [[buffer(0)]],
                const device float *b   [[buffer(1)]],
                device float       *out [[buffer(2)]],
                uint i [[thread_position_in_grid]])
{
    out[i] = a[i] + b[i];
}
"""

device = Device()
add = device.compile(ADD).kernel("add")

n = 1000
a, b, out = (device.new_buffer(4 * n) for _ in range(3))
a.numpy(np.float32)[:] = np.arange(n)
b.numpy(np.float32)[:] = 2 * np.arange(n)

device.run(add, [a, b, out], threads=n)
out.numpy(np.float32)[:4]  # array([0., 3., 6., 9.], dtype=float32)
```

## Development

```sh
uv sync
uv run pre-commit install  # enable git hooks
uv run pytest              # run tests
```
