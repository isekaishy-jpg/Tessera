//! The program's one error type: a message naming the step or call that failed.

use core::error::Error;
use core::fmt;

/// A failed step. It is built with [`fail`] or [`Context::ctx`] and read through `Display`.
#[derive(Debug)]
pub struct Failure(String);

/// A step's result.
pub type Outcome<T> = Result<T, Failure>;

/// Fails with `msg`.
pub fn fail<T>(msg: impl Into<String>) -> Outcome<T> {
    Err(Failure(msg.into()))
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for Failure {}

/// Names the call behind an error: `vkCreateBuffer: ERROR_OUT_OF_DEVICE_MEMORY: ...`.
pub trait Context<T> {
    /// The error as a [`Failure`] prefixed with `what`.
    fn ctx(self, what: impl fmt::Display) -> Outcome<T>;
}

impl<T, E: fmt::Display> Context<T> for Result<T, E> {
    fn ctx(self, what: impl fmt::Display) -> Outcome<T> {
        self.map_err(|e| Failure(format!("{what}: {e}")))
    }
}
