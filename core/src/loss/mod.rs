/*
    Appellation: loss <module>
    Contrib: @FL03
*/
//! This module provides various loss functions used in machine learning.
//!
//! ## Features
//!
//! - [`entropy`]: entropic loss functions
//! - [`standard`]: basic loss functions like mse, mae, etc.
#[doc(inline)]
pub use self::prelude::*;

pub mod entropy;
pub mod standard;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::entropy::*;
    #[doc(inline)]
    pub use super::standard::*;
}

pub trait Loss {
    type Output;

    fn loss(&self) -> Self::Output;
}
