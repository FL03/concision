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

pub use self::prelude::*;

pub(crate) mod initialize;
pub(crate) mod utils;

pub mod gen {
    pub use self::prelude::*;

    pub mod lecun;

    pub(crate) mod prelude {
        pub use super::lecun::*;
    }
}

#[doc(no_inline)]
pub use ndarray_rand as ndrand;
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;

pub(crate) mod prelude {
    pub use super::gen::prelude::*;
    pub use super::initialize::{Initialize, InitializeExt};
    pub use super::utils::*;
}
