/*
    Appellation: model <module>
    Contrib: @FL03
*/
//! This module provides the scaffolding for creating models and layers in a neural network.

#[doc(inline)]
pub use self::{layout::*, params::*, trainer::Trainer, traits::*};

pub mod layout;
pub mod params;
pub mod trainer;

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod models;

    mod prelude {
        #[doc(inline)]
        pub use super::models::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::layout::*;
    #[doc(inline)]
    pub use super::params::*;
    #[doc(inline)]
    pub use super::trainer::*;
    #[doc(inline)]
    pub use super::traits::*;
}
