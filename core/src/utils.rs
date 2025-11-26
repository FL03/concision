/*
    Appellation: utils <module>
    Created At: 2025.11.26:13:20:12
    Contrib: @FL03
*/
//! Additional utilities for creating, manipulating, and managing tensors and models.
#[doc(inline)]
pub use self::prelude::*;

pub(crate) mod arith;
pub(crate) mod gradient;
pub(crate) mod mask;
pub(crate) mod norm;
pub(crate) mod patterns;
pub(crate) mod pad;
pub(crate) mod tensor;

pub(crate) mod prelude {
    pub use super::arith::*;
    pub use super::gradient::*;
    pub use super::mask::*;
    pub use super::norm::*;
    pub use super::pad::*;
    pub use super::patterns::*;
    pub use super::tensor::*;
}