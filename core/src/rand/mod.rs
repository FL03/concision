/*
   Appellation: rand <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

pub use self::prelude::*;

pub(crate) mod generate;
pub(crate) mod utils;

#[doc(no_inline)]
pub use ndarray_rand as ndrand;
#[doc(no_inline)]
pub use ndarray_rand::{RandomExt, SamplingStrategy};
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;

pub(crate) mod prelude {
    #[doc(no_inline)]
    pub use ndarray_rand::RandomExt;

    pub use super::generate::*;
    pub use super::utils::*;
}
