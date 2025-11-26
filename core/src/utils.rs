/*
    Appellation: utils <module>
    Created At: 2025.11.26:13:20:12
    Contrib: @FL03
*/
//! Additional utilities for creating, manipulating, and managing tensors and models.
#[doc(inline)]
pub use self::prelude::*;

mod arith;
mod gradient;
mod norm;
mod patterns;
mod tensor;

mod prelude {
    #[doc(inline)]
    pub use super::arith::*;
    #[doc(inline)]
    pub use super::gradient::*;
    #[doc(inline)]
    pub use super::norm::*;
    #[doc(inline)]
    pub use super::patterns::*;
    #[doc(inline)]
    pub use super::tensor::*;
}
