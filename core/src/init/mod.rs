/*
    Appellation: init <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! This module works to provide the crate with various initialization methods suitable for
//! machine-learning models.
//!
//!
#![cfg(feature = "rand")]
#[doc(inline)]
pub use self::{distr::prelude::*, initialize::*, utils::*};

///
#[doc(inline)]
pub use crate::traits::init::*;

pub(crate) mod initialize;
pub(crate) mod utils;

pub mod distr {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod lecun;
    pub mod trunc;
    pub mod xavier;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::lecun::*;
        #[doc(inline)]
        pub use super::trunc::*;
        #[doc(inline)]
        pub use super::xavier::*;
    }
}

pub(crate) mod prelude {
    pub use super::UniformResult;
    pub use super::distr::prelude::*;
    pub use super::initialize::*;
    pub use super::utils::*;
}

pub type UniformResult<T = ()> = Result<T, rand_distr::uniform::Error>;
