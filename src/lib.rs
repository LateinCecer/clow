#[allow(unused)]

mod buffer;
mod pointer;
mod stream;
mod context;
pub mod prelude;

#[cfg(test)]
mod test {
    use crate::pointer::{ClowPointable, ClowPtr, ClowViewable, ClowViewableMut};
    use crate::prelude::ClowStream;
    use cudarc::driver::{CudaContext, CudaModule, CudaStream, DevicePtr, DriverError, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::CompileOptions;
    use std::fs;
    use std::sync::Arc;

    struct TestExecutor {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    impl TestExecutor {
        fn new(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(fs::read_to_string("tests/benchmark.cu")?, CompileOptions {
                include_paths: vec!["tests/".to_string()],
                use_fast_math: Some(false),
                // options: vec!["-O3".to_string()],
                .. Default::default()
            })?;
            let stream = ctx.default_stream();
            let module = ctx.load_module(ptx)?;
            Ok(TestExecutor {
                stream,
                module,
            })
        }

        /// Executes the kernel behind the executor with raw GPU pointers.
        ///
        /// # Safety
        ///
        /// Since this method accepts raw GPU pointers, it is impossible for this method to check
        /// whether the pointers lead to buffers that have to correct size.
        /// Therefore, the caller must make sure that the allocated buffers have a length that is
        /// greater or equal to `n`.
        unsafe fn exec_internal(
            &self,
            n: usize,
            input: ClowPtr<u8>,
            output: ClowPtr<u8>,
        ) -> Result<(), DriverError> {
            let div_kernel = self.module.load_function("gpu_div_kernel_vec")?;
            let mut builder = self.stream.launch_builder(&div_kernel);
            builder.arg(&input);
            builder.arg(&output);
            builder.arg(&2i32);

            let num_threadlets = 1024u32;
            let launch_config = LaunchConfig {
                block_dim: (num_threadlets, 1, 1),
                grid_dim: ((n as u32).div_ceil(num_threadlets), 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(launch_config)?; }
            Ok(())
        }

        /// Executes the kernel behind the executor.
        /// This is a safe wrapper around [Self::exec_internal].
        /// Since the arguments must have a valid length specifier, we can make sure that the
        /// kernel dispatch is actually safe.
        ///
        /// # Safety
        ///
        /// You can call this method with either [CudaSlice], [CudaView]/[CudaViewMut] or
        /// [ClowSlice], [ClowView]/[ClowViewMut] parameters.
        /// This implementation will ignore the inbuild synchronization mechanics in Cudarc.
        /// If event tracking is not disabled for the [CudaContext] this method will cause a panic!
        /// You can disable event tracking by calling [CudaContext::disable_event_tracking],
        /// however, be aware that this means you have to deal with synchronization yourself.
        fn exec_view(
            &self,
            input: &impl ClowViewable<u8>,
            output: &mut impl ClowViewableMut<u8>,
        ) -> Result<(), DriverError> {
            assert_eq!(input.len(), output.len());
            let input_ptr = input.as_device_ptr();
            let output_ptr = output.as_device_ptr();
            unsafe { self.exec_internal(input.len(), input_ptr, output_ptr) }
        }

        /// Safe wrapper around [Self::exec_internal] that takes implementations of Cudarc's
        /// [DevicePtr] trait.
        /// This method will deal with synchronizations correctly if event tracking is enabled!
        fn exec(
            &self,
            input: &impl DevicePtr<u8>,
            output: &mut impl DevicePtr<u8>,
        ) -> Result<(), DriverError> {
            assert_eq!(input.len(), output.len());
            let (input_ptr, _input_sync) = input.device_ptr(&self.stream);
            let (output_ptr, _output_sync) = output.device_ptr(&self.stream);
            unsafe {
                self.exec_internal(
                    input.len(),
                    ClowPtr::from_raw_parts(input_ptr),
                    ClowPtr::from_raw_parts(output_ptr),
                )
            }
        }
    }

    #[test]
    fn test_cudarc() -> Result<(), Box<dyn std::error::Error>> {
        let n = 10000;
        let ctx = CudaContext::new(0)?;
        let executor = TestExecutor::new(&ctx)?;

        // create buffers
        let stream = ctx.default_stream();
        let input = stream.clone_htod(&vec![32u8; n])?;
        let mut output = stream.alloc_zeros::<u8>(n)?;

        // execute
        executor.exec(&input, &mut output)?;
        let out_host = stream.clone_dtoh(&output)?;
        out_host.into_iter().for_each(|x| assert_eq!(x, 16));
        Ok(())
    }

    #[test]
    fn test_cudarc_disabled_event_tracking() -> Result<(), Box<dyn std::error::Error>> {
        let n = 10000;
        let ctx = CudaContext::new(0)?;
        unsafe { ctx.disable_event_tracking() };
        let executor = TestExecutor::new(&ctx)?;

        // create buffers
        let stream = ctx.default_stream();
        let input = stream.clone_htod(&vec![32u8; n])?;
        let output = stream.alloc_zeros::<u8>(n)?;

        // execute
        // NOTE: since we have disabled event tracking, we have to synchronize data manually,
        //       even though we're using CudaSlice instances.
        let ev = executor.stream.record_event(None)?;
        unsafe {
            executor.exec_internal(n, input.as_device_ptr(), output.as_device_ptr())?;
        }
        executor.stream.wait(&ev)?;

        let out_host = stream.clone_dtoh(&output)?;
        out_host.into_iter().for_each(|x| assert_eq!(x, 16));
        Ok(())
    }

    #[test]
    fn test_clow() -> Result<(), Box<dyn std::error::Error>> {
        let n = 10000;
        let ctx = CudaContext::new(0)?;
        let executor = TestExecutor::new(&ctx)?;

        // create buffers
        let stream = ctx.default_stream();
        let input = stream.clow_clone_htod(&vec![32u8; n])?;
        let output = stream.clow_alloc_zeros::<u8>(n)?;

        // execute
        // NOTE: ClowSlice instances do not have any internal synchronization mechanics, so we have
        //       to synchronize data manually. This is true regardless of whether event tracking is
        //       enabled in the CudaContext or not.
        let ev = executor.stream.record_event(None)?;
        unsafe {
            executor.exec_internal(n, input.as_device_ptr(), output.as_device_ptr())?;
        }
        executor.stream.wait(&ev)?;

        let out_host = stream.clow_clone_dtoh(&output)?;
        out_host.into_iter().for_each(|x| assert_eq!(x, 16));
        Ok(())
    }

    #[test]
    fn test_cudarc_view() -> Result<(), Box<dyn std::error::Error>> {
        let n = 10000;
        let ctx = CudaContext::new(0)?;
        unsafe { ctx.disable_event_tracking() };
        let executor = TestExecutor::new(&ctx)?;

        // create buffers
        let stream = ctx.default_stream();
        let input = stream.clone_htod(&vec![32u8; n])?;
        let mut output = stream.alloc_zeros::<u8>(n)?;

        // execute
        // NOTE: since we have disabled event tracking, we have to synchronize data manually,
        //       even though we're using CudaSlice instances.
        let ev = executor.stream.record_event(None)?;
        executor.exec_view(&input, &mut output)?;
        executor.stream.wait(&ev)?;

        let out_host = stream.clone_dtoh(&output)?;
        out_host.into_iter().for_each(|x| assert_eq!(x, 16));
        Ok(())
    }

    #[test]
    fn test_clow_view() -> Result<(), Box<dyn std::error::Error>> {
        let n = 10000;
        let ctx = CudaContext::new(0)?;
        let executor = TestExecutor::new(&ctx)?;

        // create buffers
        let stream = ctx.default_stream();
        let input = stream.clow_clone_htod(&vec![32u8; n])?;
        let mut output = stream.clow_alloc_zeros::<u8>(n)?;

        // execute
        let ev = executor.stream.record_event(None)?;
        executor.exec_view(&input, &mut output)?;
        executor.stream.wait(&ev)?;

        let out_host = stream.clow_clone_dtoh(&output)?;
        out_host.into_iter().for_each(|x| assert_eq!(x, 16));
        Ok(())
    }
}
