//! Basic example showing how to allocate device memory with Clow, launch a kernel,
//! and retrieve results.

use clow::prelude::*;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;

    // Set up CUDA context and stream
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();

    // Compile a simple elementwise kernel at runtime
    let ptx = compile_ptx(
        r#"
extern "C" __global__ void multiply(const float* in, float* out, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scalar;
    }
}
"#,
    )?;
    let module = ctx.load_module(ptx)?;

    // Allocate ClowSlice buffers and copy host data to device
    let input_host = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let input = stream.clow_clone_htod(&input_host)?;
    let output = stream.clow_alloc_zeros::<f32>(n)?;

    // Launch kernel with ClowPtr arguments
    let kernel = module.load_function("multiply")?;
    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    let input_ptr = input.as_device_ptr();
    let output_ptr = output.as_device_ptr();

    unsafe {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&2.0f32);
        builder.arg(&n);
        builder.launch(LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    // Synchronize before reading back
    stream.synchronize()?;

    // Copy results back to host
    let output_host = stream.clow_clone_dtoh(&output)?;

    // Verify results
    for i in 0..n {
        assert_eq!(output_host[i], input_host[i] * 2.0);
    }

    println!("Successfully multiplied {} elements by 2.0", n);
    println!("First 10 results: {:?}", &output_host[..10]);

    Ok(())
}
