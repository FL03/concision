/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Operations
pub use self::pad::*;

pub(crate) mod pad;

pub mod fft;

#[cfg(test)]
mod tests {}
