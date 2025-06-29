/*
    appellation: config <module>
    authors: @FL03
*/
#[doc(inline)]
pub use self::{model_config::StandardModelConfig, traits::*, types::*};

pub mod model_config;

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod config;

    mod prelude {
        #[doc(inline)]
        pub use super::config::*;
    }
}

mod types {
    //! this module defines various types in-support of the configuration model for the neural
    //! library of the concision framework.
    #[doc(inline)]
    pub use self::prelude::*;

    mod hyper_params;

    mod prelude {
        #[doc(inline)]
        pub use super::hyper_params::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::model_config::*;
    #[doc(inline)]
    pub use super::traits::*;
    #[doc(inline)]
    pub use super::types::*;
}
