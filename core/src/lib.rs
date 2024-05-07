/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_core"]

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate ndarray as nd;
#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;

pub use self::error::{Error, ErrorKind, PredictError};
pub use self::nn::{Backward, Forward, Module, Predict};
pub use self::{primitives::*, traits::prelude::*, types::prelude::*, utils::*};

#[cfg(feature = "rand")]
pub use self::rand::prelude::*;

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;
pub(crate) mod utils;

pub mod error;
pub mod func;
pub mod nn;
pub mod ops;
pub mod params;
#[cfg(feature = "rand")]
pub mod rand;
pub mod traits;
pub mod types;

pub mod prelude {

    pub use super::primitives::*;
    pub use super::utils::*;

    pub use super::error::prelude::*;
    pub use super::func::prelude::*;
    pub use super::nn::prelude::*;
    pub use super::ops::prelude::*;
    pub use super::params::prelude::*;
    #[cfg(feature = "rand")]
    pub use super::rand::prelude::*;
    pub use super::traits::prelude::*;
    pub use super::types::prelude::*;

    pub(crate) use super::primitives::rust::*;
}
