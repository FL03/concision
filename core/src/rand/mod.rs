/*
   Appellation: rand <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub use self::generate::*;
#[cfg(feature = "rand")]
pub use self::utils::*;

pub(crate) mod generate;
pub(crate) mod utils;

pub use ndarray_rand::{rand_distr, RandomExt, SamplingStrategy};
pub use rand;

pub(crate) mod prelude {
    pub use super::generate::*;
    #[cfg(feature = "rand")]
    pub use super::utils::*;
}
