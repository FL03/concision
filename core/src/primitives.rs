/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, types::*};

pub use ndarray::ShapeError;
// #[cfg(feature = "rand")]
pub use ndarray_rand::rand_distr::uniform::SampleUniform;

/// Collection of constants used throughout the system
mod constants {
    pub const DEFAULT_MODEL_SIZE: usize = 2048;
}

/// Collection of static references used throughout
mod statics {}

/// Collection of types used throughout the system
mod types {
    ///
    pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
    ///
    pub type BoxResult<T = ()> = std::result::Result<T, BoxError>;

    ///
    pub type ShapeResult<T = ()> = std::result::Result<T, ndarray::ShapeError>;
}
