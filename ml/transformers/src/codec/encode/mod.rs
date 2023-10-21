/*
   Appellation: encode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Encode
pub use self::{embed::*, encoder::*, utils::*};

pub(crate) mod embed;
pub(crate) mod encoder;

pub trait Encode {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_encoder() {}
}
