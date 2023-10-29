/*
   Appellation: decode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Decode
pub use self::{decoder::*, params::*, utils::*};

pub(crate) mod decoder;
pub(crate) mod params;

pub trait Decode {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_decoder() {}
}
