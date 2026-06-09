
# clow: lower-level wrappers around cudarc structures

This library wraps around some key structures of Cudarc.
Key features are type-safe device pointer types and fat-pointer types, as well as an alternative
to cudarc's `CudaSlice` struct that does not contain any event tracking logic at all.

# Why?

For certain applications, we need to be able to shove pointers to GPU memory onto the GPU as-is,
many of which require that a device pointers lifetime cannot be constrained to a local Rust scope.
This clashes with the event-tracking and `SyncOnDrop` logic in Cudarc.
The alternative types that Clow provides, work on a lower level than the Cudarc alternatives and
allow more flexibility.
For example, Clow's `ClowView` has an internal layout that matches a fat pointer, whereas Cudarc's
`CudaView` has a layout that cannot be broken down to a simple 16-byte pointer value.

# Versioning

Make sure the Cudarc version matches; otherwise Cudarc will probably fail to compile against your
local architecture.
You can export the specific cuda version to use in `CUDARC_CUDA_VERSION` to CUDARC without specifying
the features flag in the Cargo.toml directly.
