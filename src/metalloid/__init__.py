"""Bindings for Apple's Metal framework via the Objective-C runtime and ctypes."""

from metalloid._metal import Device
from metalloid.errors import (
    AllocationError,
    CompileError,
    KernelNotFoundError,
    MetalError,
)

__all__ = [
    "AllocationError",
    "CompileError",
    "Device",
    "KernelNotFoundError",
    "MetalError",
]
