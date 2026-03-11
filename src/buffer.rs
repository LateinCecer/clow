use crate::pointer::{ClowPointable, ClowPtr, ClowSized, ClowView, ClowViewMut, ClowViewable, ClowViewableMut};
use crate::stream::HAS_ASYNC_ALLOC;
use cudarc::driver::{result, CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DeviceSlice, DriverError, SyncOnDrop};
use std::marker::PhantomData;
use std::mem;
use std::mem::ManuallyDrop;
use std::sync::Arc;

#[derive(Debug)]
pub struct ClowSlice<T: ?Sized> {
    pub(crate) ptr: ClowPtr<T>,
    pub(crate) len: usize,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl<T: ?Sized> Send for ClowSlice<T> {}
unsafe impl<T: ?Sized> Sync for ClowSlice<T> {}

impl<T: ?Sized> Drop for ClowSlice<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        if self.len > 0 {
            let ptr = mem::replace(&mut self.ptr, ClowPtr::null());
            let ctx = self.stream.context();
            if HAS_ASYNC_ALLOC {
                ctx.record_err(unsafe {
                    result::free_async(self.ptr.ptr, self.stream.cu_stream())
                });
            } else {
                ctx.record_err(self.stream.synchronize());
                ctx.record_err(unsafe { result::free_sync(ptr.ptr) });
            }
        }
        self.len = 0;
    }
}

impl<T: ?Sized> ClowPointable<T> for ClowSlice<T> {
    fn as_device_ptr(&self) -> ClowPtr<T> {
        self.ptr
    }
}

impl<T> ClowSized<T> for ClowSlice<T> {
    fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<T> ClowViewable<T> for ClowSlice<T> {
    fn get_view(&self) -> ClowView<T> {
        ClowView::from_ptr(self.ptr, self.len)
    }
}

impl<T> ClowViewableMut<T> for ClowSlice<T> {
    fn get_view_mut(&mut self) -> ClowViewMut<T> {
        ClowViewMut::from_ptr(self.ptr, self.len)
    }
}

impl<T> ClowSlice<T> {}

impl<T> ClowSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ordinal(&self) -> usize {
        self.stream.context().ordinal()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.context()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub unsafe fn uninit(len: usize, stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        let ptr = if len > 0 && size_of::<T>() > 0 {
            unsafe { result::malloc_sync(len)? }
        } else {
            0 as _
        };
        Ok(ClowSlice {
            ptr: ClowPtr { ptr, _t: PhantomData },
            len,
            stream,
        })
    }

    pub unsafe fn uninit_async(len: usize, stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        let ptr = if len > 0 && size_of::<T>() > 0 {
            unsafe { result::malloc_async(stream.cu_stream(), len)? }
        } else {
            0 as _
        };
        Ok(ClowSlice {
            ptr: ClowPtr { ptr, _t: PhantomData },
            len,
            stream,
        })
    }

    pub fn drop_async(self) {
        if self.ptr.is_null() {
            return;
        }
        let me = ManuallyDrop::new(self);
        let ctx = me.stream.context();
        ctx.record_err(unsafe { result::free_async(me.ptr.ptr, me.stream.cu_stream()) });
    }

    pub fn from_raw_parts(ptr: ClowPtr<T>, len: usize, stream: Arc<CudaStream>) -> Self {
        Self {
            ptr,
            len,
            stream,
        }
    }
}


macro_rules! impl_clow_slice(
    ($T:ident) => (
impl<T> ClowPointable<T> for $T<T> {
    fn as_device_ptr(&self) -> ClowPtr<T> {
        let (ptr, sync) = self.device_ptr(self.stream());
        match sync {
            SyncOnDrop::Record(Some(_)) => panic!("recording is enabled for this slice!"),
            SyncOnDrop::Sync(Some(_)) => panic!("synchronization is enabled for this slice!"),
            _ => (),
        }
        ClowPtr {
            ptr,
            _t: PhantomData,
        }
    }
}

impl<T> ClowSized<T> for $T<T> {
    fn num_bytes(&self) -> usize {
        <Self as DeviceSlice<T>>::num_bytes(self)
    }

    fn len(&self) -> usize {
        <Self as DeviceSlice<T>>::len(self)
    }
}

impl<T> ClowViewable<T> for $T<T> {
    fn get_view(&self) -> ClowView<T> {
        let ptr = self.as_device_ptr();
        ClowView::from_ptr(ptr, self.len())
    }
}

impl<T> ClowViewableMut<T> for $T<T> {
    fn get_view_mut(&mut self) -> ClowViewMut<T> {
        let ptr = self.as_device_ptr();
        ClowViewMut::from_ptr(ptr, self.len())
    }
}
    );

    (<$l:lifetime> $T:ident) => (
impl<$l, T> ClowPointable<T> for $T<$l, T> {
    /// Returns a Clow device pointer to the underlying data.
    ///
    /// # Warning
    ///
    /// If this method is called in a Cudarc context in which event tracking is enabledd, it will
    /// initiate a panic!
    /// If you need event tracking in Cudarc, you can use the following code to get the pointer
    /// _as long as the pointer does not leave the local scope_:
    ///
    /// ```
    /// # use cudarc::driver::CudaContext;
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let slice = stream.alloc_zeros(1000).unwrap();
    /// let (ptr, _) = slice.device_ptr(&stream);
    /// let device_ptr = ClowPtr::from_raw_parts(ptr);
    /// ```
    fn as_device_ptr(&self) -> ClowPtr<T> {
        let (ptr, sync) = self.device_ptr(self.stream());
        match sync {
            SyncOnDrop::Record(Some(_)) => panic!("recording is enabled for this slice!"),
            SyncOnDrop::Sync(Some(_)) => panic!("synchronization is enabled for this slice!"),
            _ => (),
        }
        ClowPtr {
            ptr,
            _t: PhantomData,
        }
    }
}

impl<$l, T> ClowSized<T> for $T<$l, T> {
    fn num_bytes(&self) -> usize {
        <Self as DeviceSlice<T>>::num_bytes(self)
    }

    fn len(&self) -> usize {
        <Self as DeviceSlice<T>>::len(self)
    }
}

impl<$l, T> ClowViewable<T> for $T<$l, T> {
    fn get_view(&self) -> ClowView<T> {
        let ptr = self.as_device_ptr();
        ClowView::from_ptr(ptr, self.len())
    }
}

impl<$l, T> ClowViewableMut<T> for $T<$l, T> {
    fn get_view_mut(&mut self) -> ClowViewMut<T> {
        let ptr = self.as_device_ptr();
        ClowViewMut::from_ptr(ptr, self.len())
    }
}
    );
);

impl_clow_slice!(CudaSlice);
impl_clow_slice!(<'a> CudaView);
impl_clow_slice!(<'a> CudaViewMut);
