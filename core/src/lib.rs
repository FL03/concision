/*
    Appellation: concision-core <library>
    Contrib: @FL03
*/
//! Core abstractions and utilities for machine learning.
//!
//!
#![crate_name = "concision_core"]
#![crate_type = "lib"]

#[doc(inline)]
pub use concision_math as math;

#[doc(inline)]
pub use self::{
    activate::prelude::*, error::*, ops::prelude::*, params::prelude::*, traits::*,
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
    pub use self::{clip::*, create::*, init::*, loss::*, mask::*, norm::*, predict::*, train::*};

    pub(crate) mod clip;
    pub(crate) mod create;
    pub(crate) mod init;
    pub(crate) mod loss;
    pub(crate) mod mask;
    pub(crate) mod norm;
    pub(crate) mod predict;
    pub(crate) mod train;
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
        pub use super::gradient::*;
        pub use super::patterns::*;
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
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::utils::prelude::*;
    #[doc(inline)]
    pub use concision_math::prelude::*;
}
