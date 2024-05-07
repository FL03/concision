/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Operations
pub use self::pad::*;

pub(crate) mod pad;

pub mod fft;

pub(crate) mod prelude {
    pub use super::fft::prelude::*;
    pub use super::pad::*;
}

#[cfg(test)]
mod tests {}
