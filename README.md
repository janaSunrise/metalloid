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

## Development

```sh
uv sync
uv run pre-commit install  # enable git hooks
uv run pytest              # run tests
```
