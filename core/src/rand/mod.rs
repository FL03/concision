/*
   Appellation: rand <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

pub use self::prelude::*;

pub(crate) mod generate;
pub(crate) mod utils;

pub use ndarray_rand::{rand_distr, RandomExt, SamplingStrategy};
pub use rand;

pub(crate) mod prelude {
    pub use super::generate::*;
    pub use super::utils::*;
}
