/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! This module implements various activation functions for neural networks.
//!
//! ## Traits
//!
//! - [Heavyside]
//! - [LinearActivation]
//! - [Sigmoid]
//! - [Softmax]
//! - [ReLU]
//! - [Tanh]
//!
#[doc(inline)]
pub use self::prelude::*;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod activate;
    mod unary;

    mod prelude {
        #[doc(inline)]
        pub use super::activate::*;
        #[doc(inline)]
        pub use super::unary::*;
    }
}

pub(crate) mod utils {
    #[doc(inline)]
    pub use self::prelude::*;

    mod non_linear;
    mod simple;

    mod prelude {
        #[doc(inline)]
        pub use super::non_linear::*;
        #[doc(inline)]
        pub use super::simple::*;
    }
}

mod impls {
    mod impl_binary;
    mod impl_linear;
    mod impl_nonlinear;
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::traits::*;
    #[doc(inline)]
    pub use super::utils::*;
}
