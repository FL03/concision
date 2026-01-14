/*
    appellation: params <module>
    authors: @FL03
*/
//! this module works to provide a common interface for storing sets of parameters within a
//! given model. The [`ModelParamsBase`] implementation generically captures the behavior of
//! parameter storage, relying on the [`ParamsBase`](concision_params::ParamsBase) instance to represent
//! individual layers within the network.
#[doc(inline)]
pub use self::{layout::*, model_params::*, traits::*, types::*};

pub mod layout;
pub mod model_params;

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
    pub use self::aliases::*;

    mod aliases;
}

mod traits {
    pub use self::{format::*, hidden::*};

    mod format;
    mod hidden;
}

#[doc(hidden)]
pub(crate) mod prelude {
    pub use super::layout::*;
    pub use super::model_params::*;
    pub use super::traits::*;
    pub use super::types::*;
}
