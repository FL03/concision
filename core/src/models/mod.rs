/*
    appellation: params <module>
    authors: @FL03
*/
//! this module provides the [`ModelParamsBase`] type and its associated aliases. The
//! implementation focuses on providing a generic container for the parameters of a neural
//! network.
#[doc(inline)]
pub use self::{model_params::*, traits::*, types::*};

pub mod model_params;

#[doc(hidden)]
pub mod ex {
    #[cfg(all(feature = "rand", feature = "std"))]
    pub mod sample;
}

mod impls {
    mod impl_model_params;
    mod impl_params_deep;
    mod impl_params_shallow;

    #[cfg(feature = "rand")]
    mod impl_model_params_rand;
    #[cfg(feature = "serde")]
    mod impl_model_params_serde;
}

mod types {
    #[doc(inline)]
    pub use self::aliases::*;

    mod aliases;
}

mod traits {
    #[doc(inline)]
    pub use self::{hidden::*, model::*};

    mod hidden;
    mod model;
}

pub(crate) mod prelude {
    pub use super::model_params::*;
    pub use super::traits::*;
    pub use super::types::*;
}
