/*
    appellation: params <module>
    authors: @FL03
*/
//! this module provides the [`ModelParamsBase`] type and its associated aliases. The
//! implementation focuses on providing a generic container for the parameters of a neural
//! network.
#[doc(inline)]
pub use self::{model_params::*, types::*};

mod model_params;

mod impls {
    mod impl_model_params;
    mod impl_params_deep;
    mod impl_params_shallow;

    #[cfg(feature = "init")]
    mod impl_model_params_rand;
    #[cfg(feature = "serde")]
    mod impl_model_params_serde;
}

mod types {
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::model_params::*;
    #[doc(inline)]
    pub use super::types::*;
}
