/*
    Appellation: utils <module>
    Created At: 2025.11.26:13:20:12
    Contrib: @FL03
*/
//! Additional utilities for creating, manipulating, and managing tensors and models.
#[doc(inline)]
pub use self::prelude::*;

mod arith;
mod dropout;
mod gradient;
mod norm;
pub mod pad;
mod patterns;
mod tensor;

pub(crate) mod prelude {
    pub use super::arith::*;
    pub use super::dropout::*;
    pub use super::gradient::*;
    pub use super::norm::*;
    pub use super::pad::*;
    pub use super::patterns::*;
    pub use super::tensor::*;
}
