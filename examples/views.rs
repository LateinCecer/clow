//! Demonstrates ClowView and ClowViewMut: fat pointers that carry length information.
//! Views can be passed directly to kernels as structured arguments.

use clow::prelude::*;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;

    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();

    // Compile a kernel that takes a fat pointer (ptr + len) directly.
    // ClowView<T> has #[repr(C)] layout { CUdeviceptr, u64 } which matches this struct.
    let ptx = compile_ptx(
        r#"
struct FatPtr {
    unsigned long long ptr;
    unsigned long long len;
};

extern "C" __global__ void sum_view(FatPtr in_view, FatPtr out_view) {
    float* in_data = (float*)in_view.ptr;
    float* out_data = (float*)out_view.ptr;
    unsigned long long in_len = in_view.len;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_len) {
        out_data[idx] += in_data[idx];
    }
}
"#,
    )?;
    let module = ctx.load_module(ptx)?;

    // Allocate and fill input buffer
    let input_host = vec![1.0f32; n];
    let mut input = stream.clow_clone_htod(&input_host)?;
    stream.clow_memcpy_htod(&vec![3.14; n], &mut input)?;

    let mut output = stream.clow_alloc_zeros::<f32>(n)?;

    // Create views from slices
    let in_view = input.get_view();
    let out_view = output.get_view_mut();

    println!("Input view: {:?} elements", in_view.len());
    println!("Output view: {:?} elements", out_view.len());

    // Slice a view using index()
    let half = n / 2;
    if let Some(first_half) = in_view.index(0..half) {
        println!("Sliced view (first half): {:?} elements", first_half.len());

        // Launch kernel using the sliced view
        let kernel = module.load_function("sum_view")?;
        unsafe {
            let mut builder = stream.launch_builder(&kernel);
            builder.arg(&first_half);
            builder.arg(&out_view);
            builder.launch(LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: ((half as u32 + 255) / 256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
    }

    stream.synchronize()?;

    let result = stream.clow_clone_dtoh(&output)?;
    for i in 0..half {
        assert!((result[i] - 3.14).abs() < 1e-5, "result[{}] != 3.14", i);
    }
    println!("Sum view: first {} elements all equal 3.14", half);

    // Demonstrate ClowViewMut derefs to ClowView
    let out_view_ref: &ClowView<f32> = &out_view;
    println!("ViewMut derefs to View: {:?} elements", out_view_ref.len());

    // Convert view back to raw pointer
    let ptr: ClowPtr<f32> = in_view.into();
    println!("View into ClowPtr: {:?}", ptr);

    Ok(())
}
