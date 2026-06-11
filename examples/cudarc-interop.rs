//! Shows how Clow types interoperate with Cudarc's CudaSlice, CudaView, and DevicePtr.
//! Clow implements ClowViewable and ClowViewableMut for both Clow and Cudarc types.

use clow::prelude::*;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;

    let ctx = Arc::new(CudaContext::new(0)?);
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.default_stream();

    let ptx = compile_ptx(
        r#"
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#,
    )?;
    let module = ctx.load_module(ptx)?;

    // --- Using Cudarc types with Clow traits ---

    // CudaSlice implements ClowViewable and ClowViewableMut
    let cudarc_a = stream.clone_htod(&vec![1.0f32; n])?;
    let cudarc_b = stream.clone_htod(&vec![2.0f32; n])?;
    let cudarc_c = stream.alloc_zeros::<f32>(n)?;

    // Get ClowPtr from Cudarc types via as_device_ptr()
    let ptr_a = cudarc_a.as_device_ptr();
    let ptr_b = cudarc_b.as_device_ptr();
    let ptr_c = cudarc_c.as_device_ptr();

    // Get ClowView from Cudarc types
    let view_a = cudarc_a.get_view();
    println!("CudaSlice -> ClowView: {:?} elements", view_a.len());

    // Launch kernel with ClowPtr derived from CudaSlice
    let kernel = module.load_function("add")?;
    unsafe {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&ptr_a);
        builder.arg(&ptr_b);
        builder.arg(&ptr_c);
        builder.arg(&n);
        builder.launch(LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    stream.synchronize()?;
    let result = stream.clone_dtoh(&cudarc_c)?;
    assert_eq!(result[0], 3.0);
    println!("Cudarc CudaSlice + ClowPtr: 1.0 + 2.0 = {}", result[0]);

    // --- Using Clow types alongside Cudarc ---

    let clow_a = stream.clow_clone_htod(&vec![10.0f32; n])?;
    let clow_b = stream.clow_clone_htod(&vec![20.0f32; n])?;
    let clow_c = stream.clow_alloc_zeros::<f32>(n)?;

    // ClowSlice provides as_device_ptr() directly - no sync guards
    let clow_ptr_a = clow_a.as_device_ptr();
    let clow_ptr_b = clow_b.as_device_ptr();
    let clow_ptr_c = clow_c.as_device_ptr();

    unsafe {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&clow_ptr_a);
        builder.arg(&clow_ptr_b);
        builder.arg(&clow_ptr_c);
        builder.arg(&n);
        builder.launch(LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    stream.synchronize()?;
    let result = stream.clow_clone_dtoh(&clow_c)?;
    assert_eq!(result[0], 30.0);
    println!("ClowSlice: 10.0 + 20.0 = {}", result[0]);

    // --- Converting Cudarc raw pointers to ClowPtr ---
    // When you have a DevicePtr and need a ClowPtr for kernel args:
    let cudarc_d = stream.alloc_zeros::<f32>(n)?;
    let (raw_ptr, _sync) = cudarc_d.device_ptr(&stream);
    let clow_ptr_d: ClowPtr<f32> = ClowPtr::from_raw_parts(raw_ptr);
    println!("CudaSlice raw ptr -> ClowPtr: {:?}", clow_ptr_d);

    Ok(())
}
