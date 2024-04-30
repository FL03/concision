/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub(crate) use self::base::*;
pub use self::constants::*;
#[cfg(feature = "std")]
pub use self::std_types::*;

pub use ndarray::ShapeError;
// #[cfg(feature = "rand")]
pub use ndarray_rand::rand_distr::uniform::SampleUniform;

pub type Result<T = ()> = core::result::Result<T, crate::error::Error>;
///
pub type ShapeResult<T = ()> = core::result::Result<T, ndarray::ShapeError>;

mod constants {
    pub const DEFAULT_MODEL_SIZE: usize = 2048;
}

pub(crate) mod base {
    #[cfg(not(feature = "std"))]
    use alloc::collections;
    #[cfg(feature = "std")]
    use std::collections;

    #[cfg(not(feature = "std"))]
    pub type Map<K, V> = collections::BTreeMap<K, V>;
    #[cfg(feature = "std")]
    pub type Map<K, V> = collections::HashMap<K, V>;
}

#[cfg(feature = "std")]
mod std_types {
    ///
    pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
    ///
    pub type BoxResult<T = ()> = core::result::Result<T, BoxError>;
}
