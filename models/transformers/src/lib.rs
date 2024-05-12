/*
   Appellation: concision-transformers <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Transformers
//!
//! ### Resources
//!
//! - [Attention is All You Need](https://arxiv.org/abs/1706.03762)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;
extern crate ndarray as nd;

pub use self::attention::AttentionHead;
pub use self::params::*;
pub use self::transformer::Transformer;

#[macro_use]
pub(crate) mod macros;
pub(crate) mod transformer;

pub mod attention;
pub mod codec;
pub mod params;

pub(crate) mod impls {
    pub mod impl_head;
    pub mod impl_linalg;
    pub mod impl_params;
}

pub mod prelude {
    pub use super::attention::prelude::*;
    pub use super::Transformer;
}
