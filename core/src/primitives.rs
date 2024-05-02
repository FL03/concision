/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::constants::*;

pub use ndarray::ShapeError;
// #[cfg(feature = "rand")]
pub use ndarray_rand::rand_distr::uniform::SampleUniform;

///
pub type ShapeResult<T = ()> = core::result::Result<T, ndarray::ShapeError>;

mod constants {
    pub const DEFAULT_MODEL_SIZE: usize = 2048;
}
