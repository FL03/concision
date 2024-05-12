/*
    Appellation: math <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Mathematics
//!
//! This module focuses on implementing various mathematical objects and operations that are
//! critical to the development of machine learning algorithms.
pub use self::traits::*;

pub mod traits;

pub(crate) mod prelude {
    pub use super::traits::*;
}
