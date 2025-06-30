/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Activation functions for neural networks and their components. These functions are often 
//! used to introduce non-linearity into the model, allowing it to learn more complex patterns 
//! in the data.
//!
//! ## Overview
//! 
//! This module works to provide a complete set of activation utilities for neural networks, 
//! manifesting in a number of traits, utilities, and other primitives used to define various 
//! approaches to activation functions.
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
