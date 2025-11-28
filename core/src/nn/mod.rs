/*
    Appellation: nn <module>
    Created At: 2025.11.28:14:59:44
    Contrib: @FL03
*/
//! This module provides network specific implementations and traits supporting the development
//! of neural network models.
//!
#[doc(inline)]
pub use self::{neural_network::*, types::*};

mod neural_network;

mod traits {}

mod types {
    #[doc(inline)]
    pub use self::depth::*;

    mod depth;
}

#[allow(unused)]
pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::types::*;
}
