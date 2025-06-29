/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//! This module implments various layers for a neural network
#[doc(inline)]
pub use self::{layer::LayerBase, traits::*};

pub(crate) mod layer;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod activate;
    mod layers;

    mod prelude {
        #[doc(inline)]
        pub use super::activate::*;
        #[doc(inline)]
        pub use super::layers::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::layer::*;
}
