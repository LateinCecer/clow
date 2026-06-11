//! Demonstrates the ClowStream trait methods for memory allocation and transfers.

use clow::prelude::*;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;

    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();

    // --- clow_alloc_zeros ---
    let zeros = stream.clow_alloc_zeros::<f32>(n)?;
    let host = stream.clow_clone_dtoh(&zeros)?;
    assert!(host.iter().all(|&x| x == 0.0));
    println!("clow_alloc_zeros: allocated {} zeros", n);

    // --- clow_clone_htod ---
    let host_data: Vec<u32> = (0..n).map(|i| i as u32).collect();
    let _device = stream.clow_clone_htod(&host_data)?;
    println!("clow_clone_htod: copied {} u32 values to device", n);

    // --- clow_memset_zeros ---
    let mut buffer = stream.clow_clone_htod(&host_data)?;
    stream.clow_memset_zeros(&mut buffer)?;
    stream.synchronize()?;
    let back = stream.clow_clone_dtoh(&buffer)?;
    assert!(back.iter().all(|&x| x == 0u32));
    println!("clow_memset_zeros: cleared buffer");

    // --- clow_clone_dtoh ---
    let src = stream.clow_clone_htod(&vec![7i32; n])?;
    let dst = stream.clow_clone_dtoh(&src)?;
    assert!(dst.iter().all(|&x| x == 7));
    println!("clow_clone_dtoh: retrieved {} values, all equal to 7", n);

    // --- clow_memcpy_dtod ---
    let src = stream.clow_clone_htod(&vec![1.0f32; n])?;
    let mut dst = stream.clow_alloc_zeros::<f32>(n)?;
    stream.clow_memcpy_dtod(&stream, &src, &mut dst)?;
    stream.synchronize()?;
    let result = stream.clow_clone_dtoh(&dst)?;
    assert!(result.iter().all(|&x| x == 1.0));
    println!("clow_memcpy_dtod: device-to-device copy");

    // --- clow_clone_dtod (allocates destination) ---
    let cloned = stream.clow_clone_dtod(&stream, &src)?;
    let result = stream.clow_clone_dtoh(&cloned)?;
    assert!(result.iter().all(|&x| x == 1.0));
    println!("clow_clone_dtod: allocated and copied in one step");

    // --- unsafe clow_alloc (uninitialized memory) ---
    let uninit = unsafe { stream.clow_alloc::<f32>(n) }?;

    // Fill with a kernel to show the buffer is usable
    let ptx = compile_ptx(
        r#"
extern "C" __global__ void fill(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}
"#,
    )?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("fill")?;

    let uninit_ptr = uninit.as_device_ptr();
    unsafe {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&uninit_ptr);
        builder.arg(&3.14f32);
        builder.arg(&n);
        builder.launch(LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    stream.synchronize()?;
    let result = stream.clow_clone_dtoh(&uninit)?;
    assert!(result.iter().all(|&x| x == 3.14));
    println!("clow_alloc + kernel fill: filled {} elements with 3.14", n);

    // --- Buffer info ---
    let buf = stream.clow_alloc_zeros::<f32>(n)?;
    println!("Buffer info: len={}, num_bytes={}, ordinal={}, is_empty={}",
        buf.len(), buf.num_bytes(), buf.ordinal(), buf.is_empty());

    // --- Async drop ---
    let async_buf = unsafe { stream.clow_alloc::<f32>(n) }?;
    async_buf.drop_async();
    println!("drop_async: freed buffer asynchronously on stream");

    Ok(())
}
