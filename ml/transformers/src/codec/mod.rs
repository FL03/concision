/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
pub use self::{decode::*, encode::*, utils::*};

pub(crate) mod decode;
pub(crate) mod encode;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_codec() {}
}
