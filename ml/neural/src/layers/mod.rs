/*
    Appellation: layers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{layer::*, utils::*};

pub(crate) mod layer;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_layer() {}
}