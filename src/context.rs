use std::sync::atomic::{AtomicU32, Ordering};
use cudarc::driver::{sys, DriverError};
use log::warn;

const ERR_STACK_SIZE: usize = 16;

/// Structure to capture errors in environments in which error capturing is not possible;
/// for example in manual drops.
#[derive(Debug)]
pub struct ErrorSink {
    err_state: [AtomicU32; ERR_STACK_SIZE],
    err_stack: AtomicU32,
}

impl ErrorSink {
    pub fn new() -> Self {
        Self {
            err_state: [const { AtomicU32::new(0) }; ERR_STACK_SIZE],
            err_stack: AtomicU32::new(0),
        }
    }

    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            let idx = self.err_stack.fetch_add(1, Ordering::Relaxed) as usize;
            if idx < ERR_STACK_SIZE {
                self.err_state[idx].store(err.0 as u32, Ordering::Relaxed)
            }
        }
    }

    /// Check the error stack for any errors.
    /// The error stack is traversed last-in-last-out, so the most recent error is returned first.
    /// Each call of this method can take off one error in the stack.
    pub fn check_err(&self) -> Result<(), DriverError> {
        if let Ok(mut idx) = self.err_stack.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |old| if old > 0 {
                Some(u32::min(old, ERR_STACK_SIZE as u32) - 1)
            } else {
                None
            },
        ) {
            if idx >= ERR_STACK_SIZE as u32 {
                // there was an overflow in the error stack
                warn!("error stack overflow detected");
                idx = ERR_STACK_SIZE as u32 - 1;
            }
            let err_state = self.err_state[idx as usize].swap(0, Ordering::Relaxed);
            Err(DriverError(unsafe {
                std::mem::transmute::<u32, sys::cudaError_enum>(err_state)
            }))
        } else {
            Ok(())
        }
    }
}
