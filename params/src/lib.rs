/*
    Appellation: params <module>
    Contrib: @FL03
*/
//! Parameters for constructing neural network models. This module implements parameters using
//! the [ParamsBase] struct and its associated types. The [ParamsBase] struct provides:
//!
//! - An $`n`$ dimensional weight tensor
//! - An $`n-1`$ dimensional bias tensor
//!
//! The associated types follow suite with the [`ndarray`] crate, each of which defines a
//! different style of representation for the parameters.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(all(not(feature = "alloc"), not(feature = "std")))]
compiler_error! {
 "Either the \"alloc\" or \"std\" feature must be enabled for this crate."
}

#[doc(inline)]
pub use self::{error::*, params::ParamsBase, types::*};

#[cfg(feature = "init")]
extern crate concision_init as init;

/// Error handling for parameters
pub mod error;
/// The [`iter`] module implements various iterators for parameters
pub mod iter;

mod params;

mod impls {
    mod impl_params;
    #[allow(deprecated)]
    mod impl_params_deprecated;
    #[cfg(feature = "init")]
    mod impl_params_init;
    mod impl_params_iter;
    mod impl_params_ops;
    #[cfg(feature = "rand")]
    mod impl_params_rand;
    #[cfg(feature = "serde")]
    mod impl_params_serde;
}

mod types {
    //! Additional types supporting the params module
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    pub use crate::error::ParamsError;
    pub use crate::params::*;
    pub use crate::types::*;
}
