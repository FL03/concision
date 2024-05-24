/*
    Appellation: stats <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Statistics
//!
pub use self::summary::*;

mod summary;

pub(crate) mod prelude {
    pub use super::summary::*;
}
