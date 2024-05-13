/*
    Appellation: norm <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Normalization
//!
//!
pub use self::layer::LayerNorm;

pub mod layer;

pub const EPSILON: f64 = 1e-5;

pub(crate) mod prelude {
    pub use super::layer::LayerNorm;
}
