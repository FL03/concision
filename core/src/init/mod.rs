/*
   Appellation: rand <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

pub use self::prelude::*;

pub(crate) mod initialize;
pub(crate) mod utils;

#[doc(hidden)]
pub mod gen {
    pub mod lecun;
}

#[doc(no_inline)]
pub use ndarray_rand as ndrand;
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;

pub(crate) mod prelude {
    #[doc(hidden)]
    pub use super::initialize::{Initialize, InitializeExt};
    pub use super::utils::*;
}
