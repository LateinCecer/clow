
use cudarc::driver::{result, sys, CudaStream, DriverError};

#[derive(Debug)]
pub struct ClowEvent {
    pub(crate) inner: sys::CUevent,
}

unsafe impl Send for ClowEvent {}
unsafe impl Sync for ClowEvent {}

impl ClowEvent {
    pub fn cu_event(&self) -> sys::CUevent {
        self.inner
    }

    pub fn new(
        flags: Option<sys::CUevent_flags>,
    ) -> Result<Self, DriverError> {
        let flags = flags.unwrap_or(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let cu_event = result::event::create(flags)?;
        Ok(Self { inner: cu_event })
    }

    pub unsafe fn wait(&self, stream: &CudaStream) -> Result<(), DriverError> {
        unsafe {
            result::stream::wait_event(
                stream.cu_stream(),
                self.inner,
                sys::CUevent_wait_flags_enum::CU_EVENT_WAIT_DEFAULT,
            )
        }
    }

    pub unsafe fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        unsafe { result::event::record(self.inner, stream.cu_stream()) }
    }

    pub unsafe fn synchronize(&self) -> Result<(), DriverError> {
        unsafe { result::event::synchronize(self.cu_event()) }
    }

    pub unsafe fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        unsafe { self.synchronize() }?;
        unsafe { self.synchronize() }?;
        unsafe { result::event::elapsed(self.inner, end.inner) }
    }

    pub fn query(&self) -> bool {
        unsafe { result::event::query(self.inner) }.is_ok()
    }
}
