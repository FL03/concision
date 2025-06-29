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

    mod ndactivate;
    mod rho;
    mod unary;

    mod prelude {
        #[doc(inline)]
        pub use super::ndactivate::*;
        #[doc(inline)]
        pub use super::rho::*;
        #[doc(inline)]
        pub use super::unary::*;
    }
}

pub(crate) mod utils {
    #[doc(inline)]
    pub use self::prelude::*;

    mod funcs;

    mod prelude {
        #[doc(inline)]
        pub use super::funcs::*;
    }
}

mod impls {
    mod impl_binary;
    mod impl_linear;
    mod impl_nonlinear;
}

pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::utils::*;
}
