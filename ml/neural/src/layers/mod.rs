/*
    Appellation: layers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{features::*, kinds::*, layer::*, sublayer::*, utils::*};

pub(crate) mod features;
pub(crate) mod kinds;
pub(crate) mod layer;
pub(crate) mod sublayer;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_layer() {}
}
