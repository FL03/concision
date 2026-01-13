/*
    Appellation: loader <module>
    Created At: 2026.01.13:14:05:18
    Contrib: @FL03
*/
//! this module provides loading mechanisms for datasets and models.
#[doc(inline)]
pub use self::dataloader::Dataloader;

pub mod dataloader;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::dataloader::*;
}

pub trait Compile {}

/// The [`Loader`] trait establishes a common interface for all implemented loading mechanisms
/// within the framework.
pub trait Loader {}
