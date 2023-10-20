/*
   Appellation: transform <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Transform
pub use self::{transformer::*, utils::*};

pub(crate) mod transformer;

pub trait Transform {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_transformer() {}
}
