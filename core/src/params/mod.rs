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
#[doc(inline)]
pub use self::{error::ParamsError, params::ParamsBase, types::*};

/// this module provides the [`ParamsError`] type for handling various errors within the module
pub mod error;
/// this module implements various iterators for the [`ParamsBase`]
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
    //! this module defines various types and type aliases for the `params` module
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

pub(crate) mod prelude {
    pub use super::error::ParamsError;
    pub use super::params::ParamsBase;
    pub use super::types::*;
}
