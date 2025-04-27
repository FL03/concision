/*
    Appellation: stats <module>
    Contrib: @FL03
*/
//! Statistical primitives and utilities commonly used in machine learning.
pub use self::summary::SummaryStatistics;

pub mod summary;

pub(crate) mod prelude {
    pub use super::summary::*;
}
