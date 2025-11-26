/*
    Appellation: ops <module>
    Created At: 2025.11.26:13:21:14
    Contrib: @FL03
*/
//! Common operations used to maintain, train, and manipulate neural networks.
#[doc(inline)]
pub use self::prelude::*;

pub mod mask;
pub mod pad;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::mask::*;
    #[doc(inline)]
    pub use super::pad::*;
}
