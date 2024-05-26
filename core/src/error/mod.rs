/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

mod err;

pub mod kinds;

pub trait ErrorKind: Clone + ToString {}

macro_rules! impl_error_type {
    ($($ty:ty),* $(,)*) => {
        $(impl_error_type!(@impl $ty);)*
    };
    (@impl $ty:ty) => {
        impl ErrorKind for $ty {}

        impl_error_type!(@std $ty);
    };
    (@std $ty:ty) => {

        #[cfg(feature = "std")]
        impl std::error::Error for $ty {}
    };
}

impl_error_type!(
    kinds::Errors,
    kinds::PredictError,
    crate::nn::ModelError
);

pub(crate) mod prelude {
    pub use super::err::Error;
    pub use super::kinds::*;
}
