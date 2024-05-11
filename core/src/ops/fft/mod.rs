/*
   Appellation: fft <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Fast Fourier Transform
//!
//!
pub use self::prelude::*;

pub(crate) mod fft;
pub(crate) mod utils;

pub mod cmp;
pub mod plan;

pub trait Fft<T> {
    type Data;

    fn rfft(&self) -> Self;
}

pub(crate) mod prelude {
    pub use super::cmp::*;
    pub use super::fft::*;
    pub use super::plan::*;
    pub use super::utils::*;
    pub use super::Fft;
}

#[cfg(test)]
mod tests {}
