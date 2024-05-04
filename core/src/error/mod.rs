/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

mod err;

pub trait ErrKind {}

macro_rules! impl_error_type {
    ($($ty:ty),* $(,)*) => {
        $(impl_error_type!(@impl $ty);)*
    };
    (@impl $ty:ty) => {
        impl ErrKind for $ty {}

        impl_error_type!(@std $ty);
    };
    (@std $ty:ty) => {

        #[cfg(feature = "std")]
        impl std::error::Error for $ty {}
    };
}

impl_error_type!(
    err::Error,
    kinds::ExternalError,
    kinds::PredictError,
    crate::models::ModelError
);

pub mod kinds {
    pub use self::prelude::*;

    pub mod external;
    pub mod predict;

    pub(crate) mod prelude {
        pub use super::external::*;
        pub use super::predict::*;
    }
}

pub(crate) mod prelude {
    pub use super::err::*;
    pub use super::kinds::prelude::*;
}
