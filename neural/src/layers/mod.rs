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

pub(crate) mod types {
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::layer::*;
    #[doc(inline)]
    pub use super::types::*;
}
