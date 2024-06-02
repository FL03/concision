/*
   Appellation: func <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Functional
pub use self::prelude::*;

#[macro_use]
pub mod activate;
pub mod loss;

pub(crate) mod prelude {
    pub use super::activate::prelude::*;
    pub use super::loss::prelude::*;
}

#[doc(hidden)]
pub trait Apply<T> {
    type Output;

    fn apply<U, F>(&self, f: F) -> Self::Output
    where
        F: Fn(T) -> U;
}
