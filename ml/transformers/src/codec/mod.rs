/*
   Appellation: codec <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Codec
pub use self::utils::*;

pub mod decode;
pub mod encode;

pub trait Codec {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_codec() {}
}
