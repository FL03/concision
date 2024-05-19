/*
    Appellation: math <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Mathematics
//!
//! This module focuses on providing the mathematical foundation for the library.
//! Any defined operation is designed to extend the functionality of the basic primitives
//! as well as the `ndarray` crate. 
pub use self::traits::*;

pub mod traits;

pub(crate) mod prelude {
    pub use super::traits::*;
}
