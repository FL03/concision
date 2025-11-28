/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//! This module implments various layers for a neural network
#[doc(inline)]
pub use self::{layer::LayerBase, traits::*, types::*};

pub(crate) mod layer;
pub mod sequential;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::{activate::*, layers::*};

    mod activate;
    mod layers;
    mod store;
}

pub(crate) mod types {
    #[doc(inline)]
    pub use self::aliases::*;

    mod aliases;
}

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::types::*;
}
