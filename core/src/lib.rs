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
#![crate_name = "concision_core"]
#![crate_type = "lib"]

#[doc(inline)]
pub use concision_math as math;

#[doc(inline)]
pub use self::{
    activate::prelude::*, error::*, ops::prelude::*, params::prelude::*, traits::prelude::*,
    utils::prelude::*,
};

#[allow(unused)]
#[macro_use]
pub(crate) mod macros;
#[allow(unused)]
#[macro_use]
pub(crate) mod seal;

pub mod activate;
pub mod data;
pub mod error;
pub mod init;
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

    pub mod clip;
    pub mod create;
    pub mod init;
    pub mod loss;
    pub mod mask;
    pub mod norm;
    pub mod predict;
    pub mod train;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::clip::*;
        #[doc(inline)]
        pub use super::create::*;
        #[doc(inline)]
        pub use super::init::*;
        #[doc(inline)]
        pub use super::loss::*;
        #[doc(inline)]
        pub use super::mask::*;
        #[doc(inline)]
        pub use super::norm::*;
        #[doc(inline)]
        pub use super::predict::*;
        #[doc(inline)]
        pub use super::train::*;
    }
}

pub mod types {
    // #[doc(inline)]
    // pub use self::features::*;

    // pub(crate) mod features;
}

pub mod utils {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod gradient;
    pub mod norm;
    pub mod patterns;
    pub mod tensor;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::gradient::*;
        #[doc(inline)]
        pub use super::patterns::*;
        #[doc(inline)]
        pub use super::tensor::*;
    }
}

pub mod prelude {
    #[doc(no_inline)]
    pub use crate::activate::prelude::*;
    #[doc(no_inline)]
    pub use crate::error::*;
    #[doc(no_inline)]
    pub use crate::ops::prelude::*;
    #[doc(no_inline)]
    pub use crate::params::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::prelude::*;
    #[doc(no_inline)]
    pub use crate::utils::prelude::*;
    #[doc(inline)]
    pub use concision_math::prelude::*;
}
