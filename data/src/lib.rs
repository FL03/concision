/*
    Appellation: concision-data <library>
    Contrib: @FL03
*/
//! Datasets and data loaders for the Concision framework.
#![crate_name = "concision_data"]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use self::{dataset::DatasetBase, traits::prelude::*};

pub mod dataset;
#[cfg(feature = "loader")]
pub mod loader;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

pub mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod convert;
    pub mod records;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::convert::*;
        #[doc(inline)]
        pub use super::records::*;
    }
}

pub mod prelude {
    #[doc(no_inline)]
    pub use crate::dataset::*;
    #[cfg(feature = "loader")]
    #[doc(no_inline)]
    pub use crate::loader::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::prelude::*;
}
