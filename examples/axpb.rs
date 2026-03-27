use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use clow::prelude::ClowStream;
use clow::prelude::ClowPointable;

const PTX_SRC: &str = "
extern \"C\" __global__
void axpb(
    double* y,
    const double* x,
    const double* a,
    const double* b,
    const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    y[i] = (*a) * x[i] + (*b);
}
";

fn main() -> Result<(), DriverError> {
    let ptx = compile_ptx(PTX_SRC).unwrap();

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(ptx)?;
    let f = module.load_function("axpb")?;

    let x_host = [1.0f64, 2.0, 3.0, 4.0];
    let y_host = [0.0f64; 4];

    let a_host = [2.0f64];
    let b_host = [1.0f64];
    let n_host = [4u32];

    println!("Before launch x {:?}", x_host);
    println!("Before launch y {:?}", y_host);

    // --- device buffers ---
    let x_dev = stream.clow_clone_htod(&x_host)?;
    let y_dev = stream.clow_clone_htod(&y_host)?;

    let a_dev = stream.clow_clone_htod(&a_host)?;
    let b_dev = stream.clow_clone_htod(&b_host)?;
    let n_dev = stream.clow_clone_htod(&n_host)?;

    // --- pointers ---
    let x_ptr = x_dev.as_device_ptr();
    let y_ptr = y_dev.as_device_ptr();
    let a_ptr = a_dev.as_device_ptr();
    let b_ptr = b_dev.as_device_ptr();
    let n_ptr = n_dev.as_device_ptr();

    let mut launch = stream.launch_builder(&f);

    launch.arg(&y_ptr);
    launch.arg(&x_ptr);
    launch.arg(&a_ptr);
    launch.arg(&b_ptr);
    launch.arg(&n_ptr);

    let cfg = LaunchConfig::for_num_elems(n_host[0]);

    unsafe {
        launch.launch(cfg)?;
    }

    stream.synchronize()?; // simpler than event

    // --- copy back ---
    let y_host = stream.clow_clone_dtoh(&y_dev)?;

    println!("======================");
    println!("After launch y {:?}", y_host);

    Ok(())
}
