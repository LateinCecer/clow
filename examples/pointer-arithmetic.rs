//! Demonstrates ClowPtr operations: offset, cast, null checks, and from_raw_parts.

use clow::prelude::*;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;

    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();

    // Compile a kernel that takes a pre-offset pointer
    let ptx = compile_ptx(
        r#"
extern "C" __global__ void fill_offset(float* data, float value, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}
"#,
    )?;
    let module = ctx.load_module(ptx)?;

    // Allocate a buffer of f32
    let buffer = stream.clow_alloc_zeros::<f32>(n)?;
    let base_ptr = buffer.as_device_ptr();

    // ClowPtr::null and is_null
    let null_ptr = ClowPtr::<f32>::null();
    assert!(null_ptr.is_null());
    assert!(!base_ptr.is_null());

    // Use offset to get a pointer to the middle of the buffer
    let half = n / 2;
    let middle_ptr = unsafe { base_ptr.offset(half as isize) };

    // Launch kernel writing to the second half of the buffer
    let kernel = module.load_function("fill_offset")?;
    let block_size = 256u32;
    let grid_size = (half as u32 + block_size - 1) / block_size;

    unsafe {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&middle_ptr);
        builder.arg(&42.0f32);
        builder.arg(&half);
        builder.launch(LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    stream.synchronize()?;

    let host = stream.clow_clone_dtoh(&buffer)?;

    // First half should be zeros, second half should be 42.0
    for i in 0..half {
        assert_eq!(host[i], 0.0);
    }
    for i in half..n {
        assert_eq!(host[i], 42.0);
    }

    println!("Pointer offset demo: first half zeros, second half 42.0");

    // Demonstrate cast: treat f32 buffer as u32 (same size, different interpretation)
    let as_u32 = base_ptr.cast::<u32>();
    println!("Cast f32 pointer to u32: {:?} -> {:?}", base_ptr, as_u32);

    // Demonstrate from_raw_parts with Cudarc's raw pointer
    let (raw_ptr, _) = buffer.stream().alloc_zeros::<f32>(1)?.device_ptr(&stream);
    let clow_from_raw = ClowPtr::<f32>::from_raw_parts(raw_ptr);
    println!("Created ClowPtr from raw CUdeviceptr: {:?}", clow_from_raw);

    Ok(())
}
