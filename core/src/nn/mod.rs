/*
    Appellation: nn <module>
    Created At: 2025.11.28:14:59:44
    Contrib: @FL03
*/
//! This module provides network specific implementations and traits supporting the development
//! of neural network models.
//!
#[doc(inline)]
pub use self::traits::*;

mod traits {
    #[doc(inline)]
    pub use self::{model::*, neural_network::*};

    mod model;
    mod neural_network;
}

pub(crate) mod prelude {
    pub use super::traits::*;
}
