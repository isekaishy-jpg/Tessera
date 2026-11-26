//! The program's one error type: a message naming the step or call that failed.

use core::fmt;

/// A failed step.
#[derive(Debug)]
pub struct Failure(pub String);

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

/// Names the call behind an error: `vkCreateBuffer: ERROR_OUT_OF_DEVICE_MEMORY: ...`.
pub trait Context<T> {
    /// The error as a [`Failure`] prefixed with `what`.
    fn ctx(self, what: &str) -> Outcome<T>;
}

impl<T, E: fmt::Display> Context<T> for Result<T, E> {
    fn ctx(self, what: &str) -> Outcome<T> {
        self.map_err(|e| Failure(format!("{what}: {e}")))
    }
}
