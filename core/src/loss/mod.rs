/*
    Appellation: loss <module>
    Contrib: @FL03
*/
//! # Loss Functions
//!
//! Loss functions are used to measure the difference between the predicted output and the
//! actual output of a model. Over the years, several leading approaches have been researched
//! and developed, each with its own strengths and weaknesses. This module provides a
//! collection of traits and implementations for various loss functions, including
//! entropic loss, standard loss, and others.
#[doc(inline)]
pub use self::traits::*;

mod traits {
    //! this module implements the various traits of the loss module
    #[doc(inline)]
    pub use self::prelude::*;

    mod entropy;
    mod loss;
    mod standard;

    mod prelude {
        #[doc(inline)]
        pub use super::entropy::*;
        #[doc(inline)]
        pub use super::loss::*;
        #[doc(inline)]
        pub use super::standard::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::traits::*;
}
