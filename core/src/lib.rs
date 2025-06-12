/*
    Appellation: concision-core <library>
    Contrib: @FL03
*/
//! This library provides the core abstractions and utilities for the Concision framework.
//!
//! ## Features
//!
//! - [ParamsBase]: A structure for defining the parameters within a neural network.
//! - [Backward]: This trait denotes a single backward pass through a layer of a neural network.
//! - [Forward]: This trait denotes a single forward pass through a layer of a neural network.
//!
#![allow(clippy::module_inception)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use concision_utils as utils;

#[doc(inline)]
pub use self::{
    activate::prelude::*, error::*, loss::prelude::*, ops::prelude::*, params::prelude::*,
    traits::prelude::*, utils::prelude::*,
};

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
    #[macro_use]
    pub mod unary;
}

pub mod activate;
pub mod error;
pub mod init;
pub mod loss;
pub mod params;

pub mod ops {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod fill;
    pub mod pad;
    pub mod reshape;
    pub mod tensor;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::fill::*;
        #[doc(inline)]
        pub use super::pad::*;
        #[doc(inline)]
        pub use super::reshape::*;
        #[doc(inline)]
        pub use super::tensor::*;
    }
}
pub mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod apply;
    pub mod clip;
    pub mod codex;
    pub mod gradient;
    pub mod init;
    pub mod like;
    pub mod mask;
    pub mod norm;
    pub mod propagation;
    pub mod scalar;
    pub mod tensor;
    pub mod wnb;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::apply::*;
        #[doc(inline)]
        pub use super::clip::*;
        #[doc(inline)]
        pub use super::codex::*;
        #[doc(inline)]
        pub use super::gradient::*;
        #[doc(inline)]
        pub use super::init::*;
        #[doc(inline)]
        pub use super::like::*;
        #[doc(inline)]
        pub use super::mask::*;
        #[doc(inline)]
        pub use super::norm::*;
        #[doc(inline)]
        pub use super::propagation::*;
        #[doc(inline)]
        pub use super::scalar::*;
        #[doc(inline)]
        pub use super::tensor::*;
        #[doc(inline)]
        pub use super::wnb::*;
    }
}

pub mod prelude {
    #[doc(no_inline)]
    pub use crate::activate::prelude::*;
    #[doc(no_inline)]
    pub use crate::error::*;
    #[cfg(feature = "rand")]
    #[doc(no_inline)]
    pub use crate::init::prelude::*;
    #[doc(no_inline)]
    pub use crate::loss::prelude::*;
    #[doc(no_inline)]
    pub use crate::ops::prelude::*;
    #[doc(no_inline)]
    pub use crate::params::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::prelude::*;
    #[doc(no_inline)]
    pub use concision_utils::prelude::*;
}
