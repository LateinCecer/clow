use std::marker::PhantomData;
use crate::buffer::ClowSlice;
use crate::pointer::ClowPtr;
use crate::pointer::ClowViewable;
use crate::pointer::ClowViewableMut;
use cudarc::driver::{result, CudaStream, DeviceRepr, DriverError, HostSlice, ValidAsZeroBits};
use std::sync::Arc;


pub trait ClowStream {
    fn clow_null<T>(
        self: &Arc<Self>,
    ) -> Result<ClowSlice<T>, DriverError>;
    unsafe fn clow_alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<ClowSlice<T>, DriverError>;
    fn clow_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<ClowSlice<T>, DriverError>;
    fn clow_memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError>;
    fn clow_memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        dst: &mut Src,
    ) -> Result<ClowSlice<T>, DriverError>;
    fn clow_clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<ClowSlice<T>, DriverError>;
    fn clow_memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError>;
    fn clow_memcpy_dtov<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        dst: &mut Src,
    ) -> Result<Vec<T>, DriverError>;
    fn clow_clone_dtoh<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError>;
    fn clow_memcpy_dtoh<T: DeviceRepr, Src: ClowViewable<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError>;
    fn clow_memcpy_dtod<T, Src: ClowViewable<T>, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        src_stream: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError>;
    fn clow_clone_dtod<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        src_stream: &Arc<Self>,
        src: &Src,
    ) -> Result<ClowSlice<T>, DriverError>;
}

pub(crate) const HAS_ASYNC_ALLOC: bool = true;

impl ClowStream for CudaStream {
    fn clow_null<T>(
        self: &Arc<Self>,
    ) -> Result<ClowSlice<T>, DriverError> {
        self.context().bind_to_thread()?;
        let ptr = if HAS_ASYNC_ALLOC {
            unsafe { result::malloc_async(self.cu_stream(), 0) }?
        } else {
            unsafe { result::malloc_sync(0) }?
        };
        Ok(ClowSlice {
            ptr: ClowPtr { ptr, _t: PhantomData },
            len: 0,
            stream: self.clone(),
        })
    }

    unsafe fn clow_alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<ClowSlice<T>, DriverError> {
        self.context().bind_to_thread()?;
        let ptr = if HAS_ASYNC_ALLOC {
            unsafe { result::malloc_async(self.cu_stream(), len * std::mem::size_of::<T>())? }
        } else {
            unsafe { result::malloc_sync(len * std::mem::size_of::<T>())? }
        };
        Ok(ClowSlice {
            ptr: ClowPtr { ptr, _t: PhantomData },
            len,
            stream: self.clone(),
        })
    }

    fn clow_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<ClowSlice<T>, DriverError> {
        let mut dst = unsafe { <CudaStream as ClowStream>::clow_alloc::<T>(self, len) }?;
        self.clow_memset_zeros(&mut dst)?;
        Ok(dst)
    }

    fn clow_memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        self.context().bind_to_thread()?;
        let num_bytes = dst.num_bytes();
        unsafe { result::memset_d8_async(dst.as_device_ptr().ptr, 0, num_bytes, self.cu_stream()) }
    }

    fn clow_memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &mut Src,
    ) -> Result<ClowSlice<T>, DriverError> {
        let mut dst = unsafe { self.clow_alloc::<T>(src.len()) }?;
        self.clow_memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    fn clow_clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<ClowSlice<T>, DriverError> {
        let mut dst = unsafe { self.clow_alloc::<T>(src.len()) }?;
        self.clow_memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    fn clow_memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.context().bind_to_thread()?;
        let (src, _record_src) = unsafe { src.stream_synced_slice(self) };
        unsafe { result::memcpy_htod_async(dst.as_device_ptr().ptr, src, self.cu_stream()) }
    }

    fn clow_memcpy_dtov<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        src: &mut Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        self.clow_memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    fn clow_clone_dtoh<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        self.clow_memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    fn clow_memcpy_dtoh<T: DeviceRepr, Src: ClowViewable<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.context().bind_to_thread()?;
        let (dst, _record_dst) = unsafe { dst.stream_synced_mut_slice(self) };
        unsafe { result::memcpy_dtoh_async(dst, src.as_device_ptr().ptr, self.cu_stream()) }
    }

    fn clow_memcpy_dtod<T, Src: ClowViewable<T>, Dst: ClowViewableMut<T>>(
        self: &Arc<Self>,
        src_stream: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.context().bind_to_thread()?;

        let num_bytes = src.num_bytes();
        let src_ctx = src_stream.context();
        let dst_ctx = self.context();

        let src = src.as_device_ptr();
        let dst = dst.as_device_ptr();

        if src_ctx == dst_ctx {
            unsafe { result::memcpy_dtod_async(dst.ptr, src.ptr, num_bytes, self.cu_stream()) }
        } else {
            unsafe { result::memcpy_peer_async(
                dst_ctx.cu_ctx(),
                dst.ptr,
                src_ctx.cu_ctx(),
                src.ptr,
                num_bytes,
                self.cu_stream(),
            ) }
        }
    }

    fn clow_clone_dtod<T: DeviceRepr, Src: ClowViewable<T>>(
        self: &Arc<Self>,
        src_stream: &Arc<Self>,
        src: &Src,
    ) -> Result<ClowSlice<T>, DriverError> {
        let mut dst = unsafe { self.clow_alloc::<T>(src.len()) }?;
        self.clow_memcpy_dtod(src_stream, src, &mut dst)?;
        Ok(dst)
    }
}
