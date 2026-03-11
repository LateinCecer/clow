use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::DeviceRepr;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use std::fmt;

/// This pointer representation should have the exact same layout as a device pointer.
/// Thus, we can use this wrapper directly on the GPU.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClowPtr<T: ?Sized> {
    pub(crate) ptr: CUdeviceptr,
    pub(crate) _t: PhantomData<T>,
}

unsafe impl<T: ?Sized> Send for ClowPtr<T> {}
unsafe impl<T: ?Sized> Sync for ClowPtr<T> {}
unsafe impl<T: ?Sized> DeviceRepr for ClowPtr<T> {}



impl<T: ?Sized> Clone for ClowPtr<T> {
    fn clone(&self) -> Self {
        ClowPtr {
            ptr: self.ptr,
            _t: PhantomData,
        }
    }
}

impl<T: ?Sized> Copy for ClowPtr<T> {}

pub trait ClowPointable<T: ?Sized> {
    /// Returns a device pointer to the underlying data from the object.
    fn as_device_ptr(&self) -> ClowPtr<T>;
}

pub trait ClowSized<T>: ClowPointable<T> {
    /// Returns the size of the view in bytes
    fn num_bytes(&self) -> usize;
    fn len(&self) -> usize;
}

pub trait ClowViewable<T>: ClowPointable<T> + ClowSized<T> {
    fn get_view(&self) -> ClowView<T>;
}

pub trait ClowViewableMut<T>: ClowPointable<T> + ClowSized<T> {
    fn get_view_mut(&mut self) -> ClowViewMut<T>;
}

impl<T: ?Sized> ClowPointable<T> for ClowPtr<T> {
    fn as_device_ptr(&self) -> ClowPtr<T> {
        *self
    }
}

impl<T: ?Sized> fmt::Pointer for ClowPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = self.ptr as *const c_void;
        fmt::Pointer::fmt(&ptr, f)
    }
}

impl<T: ?Sized> ClowPtr<T> {
    pub fn from_raw_parts(ptr: CUdeviceptr) -> Self {
        Self {
            ptr,
            _t: PhantomData,
        }
    }

    pub fn null() -> Self {
        Self {
            ptr: 0,
            _t: PhantomData,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr == 0
    }

    pub unsafe fn offset(self, count: isize) -> Self
    where T: Sized {
        let ptr = self.ptr + (count * size_of::<T>() as isize) as u64;
        Self {
            ptr,
            _t: PhantomData,
        }
    }

    pub unsafe fn wrapping_offset(self, count: isize) -> Self
    where T: Sized {
        let ptr = self.ptr + (count * size_of::<T>() as isize) as u64;
        Self {
            ptr,
            _t: PhantomData,
        }
    }

    pub fn cast<U: ?Sized + DeviceRepr>(self) -> ClowPtr<U> {
        ClowPtr {
            ptr: self.ptr,
            _t: PhantomData,
        }
    }
}

impl<T> From<cudarc::driver::sys::CUdeviceptr> for ClowPtr<T> {
    fn from(value: cudarc::driver::sys::CUdeviceptr) -> Self {
        Self {
            ptr: value,
            _t: PhantomData,
        }
    }
}

/// This structure should have the exact same layout as a fat pointer on the GPU.
/// That means that we can mark this as `DeviceRepr` and upload it to the GPU as part of a kernel
/// parameter or as part of a larger structure.
#[repr(C)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClowView<T: ?Sized> {
    ptr: ClowPtr<T>,
    len: u64,
}

unsafe impl<T: ?Sized> Send for ClowView<T> {}
unsafe impl<T: ?Sized> Sync for ClowView<T> {}
unsafe impl<T: ?Sized> DeviceRepr for ClowView<T> {}

impl<T: ?Sized> Clone for ClowView<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, len: self.len }
    }
}

impl<T: ?Sized> Copy for ClowView<T> {}

/// This structure should have the exact same layout as a fat pointer on the GPU.
/// That means that we can mark this as `DeviceRepr` and upload it to the GPU as part of a kernel
/// parameter or as part of a larger structure.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClowViewMut<T: ?Sized>(ClowView<T>);

unsafe impl<T: ?Sized> Send for ClowViewMut<T> {}
unsafe impl<T: ?Sized> Sync for ClowViewMut<T> {}
unsafe impl<T: ?Sized> DeviceRepr for ClowViewMut<T> {}


impl<T: ?Sized> Clone for ClowViewMut<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: ?Sized> Copy for ClowViewMut<T> {}

impl<T: ?Sized> From<ClowView<T>> for ClowPtr<T> {
    fn from(value: ClowView<T>) -> Self {
        value.ptr
    }
}

impl<T: ?Sized> ClowView<T> {
    pub fn from_ptr(ptr: ClowPtr<T>, len: usize) -> Self {
        Self { ptr, len: len as u64 }
    }

    pub fn index(&self, range: Range<usize>) -> Option<Self>
    where T: Sized {
        if range.end <= self.len as usize {
            let ptr = unsafe { self.ptr
                .offset((range.start * std::mem::size_of::<T>()) as isize) };
            Some(Self { ptr, len: range.len() as u64 })
        } else {
            None
        }
    }
}

impl<T> ClowPointable<T> for ClowView<T> {
    fn as_device_ptr(&self) -> ClowPtr<T> {
        self.ptr
    }
}

impl<T> ClowSized<T> for ClowView<T> {
    fn num_bytes(&self) -> usize {
        self.len as usize * size_of::<T>()
    }

    fn len(&self) -> usize {
        self.len as usize
    }
}

impl<T: ?Sized> From<ClowViewMut<T>> for ClowPtr<T> {
    fn from(value: ClowViewMut<T>) -> Self {
        value.0.ptr
    }
}

impl<T: ?Sized> ClowViewMut<T> {
    pub fn from_ptr(ptr: ClowPtr<T>, len: usize) -> Self {
        Self(ClowView::from_ptr(ptr, len))
    }

    pub fn index_mut(&mut self, range: Range<usize>) -> Option<Self>
    where T: Sized {
        self.0.index(range).map(|va| Self(va))
    }
}

impl<T: ?Sized> Deref for ClowViewMut<T> {
    type Target = ClowView<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ?Sized> DerefMut for ClowViewMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


impl<T> ClowPointable<T> for ClowViewMut<T> {
    fn as_device_ptr(&self) -> ClowPtr<T> {
        self.ptr
    }
}

impl<T> ClowSized<T> for ClowViewMut<T> {
    fn num_bytes(&self) -> usize {
        self.len as usize * size_of::<T>()
    }

    fn len(&self) -> usize {
        self.len as usize
    }
}

#[cfg(feature = "cust")]
impl<T: ?Sized + cust::memory::DeviceCopy> From<cust::memory::DevicePointer<T>> for ClowPtr<T> {
    fn from(value: cust::memory::DevicePointer<T>) -> Self {
        Self {
            ptr: value.as_raw(),
            _t: PhantomData,
        }
    }
}

#[cfg(feature = "cust")]
impl<T: ?Sized + cust::memory::DeviceCopy> From<ClowPtr<T>> for cust::memory::DevicePointer<T> {
    fn from(value: ClowPtr<T>) -> Self {
        cust::memory::DevicePointer::from_raw(value.ptr)
    }
}

#[cfg(feature = "cust")]
impl<T: ?Sized + cust::memory::DeviceCopy> From<cust::memory::DeviceSlice<T>> for ClowView<T> {
    fn from(value: cust::memory::DeviceSlice<T>) -> Self {
        Self {
            ptr: value.as_device_ptr().into(),
            len: value.len() as u64,
        }
    }
}

#[cfg(feature = "cust")]
impl<T: ?Sized + cust::memory::DeviceCopy> From<ClowView<T>> for cust::memory::DeviceSlice<T> {
    fn from(value: ClowView<T>) -> Self {
        unsafe { cust::memory::DeviceSlice::from_raw_parts(value.ptr.into(), value.len as usize) }
    }
}
