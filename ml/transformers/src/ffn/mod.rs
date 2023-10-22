/*
   Appellation: decode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Decode
pub use self::{network::*, utils::*};

pub(crate) mod network;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_ffn() {}
}
