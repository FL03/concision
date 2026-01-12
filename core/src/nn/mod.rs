/*
    Appellation: nn <module>
    Created At: 2025.11.28:14:59:44
    Contrib: @FL03
*/
//! This module provides network specific implementations and traits supporting the development
//! of neural network models.
//!
#[doc(inline)]
pub use self::{layer::*, traits::*};

pub mod layer;

mod traits {
    #[doc(inline)]
    pub use self::{context::*, layer::*, model::*, neural_network::*};

    mod context;
    mod layer;
    mod model;
    mod neural_network;
}

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::traits::*;
}

#[cfg(test)]
mod tests {

}
