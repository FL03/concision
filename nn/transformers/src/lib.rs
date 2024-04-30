/*
   Appellation: transformers <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Transformers
//!
extern crate concision_core as concision;
extern crate concision_neural as neural;

pub use self::primitives::*;

#[allow(unused_imports)]
pub(crate) use self::base::*;

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod attention;
pub mod codec;
pub mod ffn;
pub mod ops;
pub mod transform;

pub(crate) use concision as core;

#[allow(unused_imports)]
pub(crate) mod base {

    pub(crate) use neural::params::masks::Mask;
    pub(crate) use neural::params::{Biased, Weighted};

    pub(crate) type BoxResult<T = ()> = core::result::Result<T, Box<dyn std::error::Error>>;
}

pub mod prelude {
    pub use crate::attention::params::*;
    pub use crate::codec::*;
    pub use crate::ffn::*;
    pub use crate::ops::*;
    pub use crate::transform::*;

    pub use crate::primitives::*;

    pub(crate) use crate::base::*;
}
