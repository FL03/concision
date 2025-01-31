/*
    Appellation: init <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Initialization
//!
//! This module implements several initialization primitives for generating tensors using
//! various distributions and strategies. The module is designed to be used in conjuction with
//! the `rand` and `rand_distr` libraries. While `ndarray_rand` provides a `RandomExt` trait,
//! we provide an alternative [Initialize] trait which is designed to be more flexible and
//! better suited for machine-learning workloads.
#![cfg(feature = "rand")]

pub use self::distr::prelude::*;
pub use self::traits::*;
pub use self::utils::*;

pub(crate) mod traits;
pub(crate) mod utils;

pub mod distr {
    pub use self::prelude::*;

    pub mod lecun;
    pub mod trunc;
    pub mod xavier;

    pub(crate) mod prelude {
        pub use super::lecun::*;
        pub use super::trunc::*;
        pub use super::xavier::*;
    }
}

type UniformResult<T = ()> = Result<T, rand_distr::uniform::Error>;

#[doc(hidden)]
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;

pub(crate) mod prelude {
    pub use super::distr::prelude::*;
    pub use super::traits::{Initialize, InitializeExt};
    pub use super::utils::*;
}
