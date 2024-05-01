/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![crate_name = "concision_core"]

#[cfg(not(feature = "std"))]
extern crate alloc;
extern crate ndarray as nd;

pub use self::{error::Error, primitives::*, traits::prelude::*, utils::*};

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;
pub(crate) mod utils;

pub mod error;
pub mod models;
pub mod ops;
pub mod params;
pub mod traits;

#[allow(unused_imports)]
pub(crate) mod rust {
    // pub(crate) use core::borrow;
    pub(crate) use core::*;

    #[cfg(not(feature = "std"))]
    pub(crate) use self::no_std::*;
    #[cfg(feature = "std")]
    pub(crate) use self::with_std::*;

    #[cfg(not(feature = "std"))]
    mod no_std {
        pub use alloc::borrow::Cow;
        pub use alloc::boxed::{self, Box};
        pub use alloc::collections::{self, BTreeMap, BTreeSet, BinaryHeap, VecDeque};
        pub use alloc::vec::{self, Vec};
    }
    #[cfg(feature = "std")]
    mod with_std {
        pub use std::borrow::Cow;
        pub use std::boxed::{self, Box};
        pub use std::collections::{self, BTreeMap, BTreeSet, BinaryHeap, VecDeque};
        pub(crate) use std::sync::Arc;
        pub use std::vec::{self, Vec};
    }
}

pub mod prelude {

    pub use crate::primitives::*;
    pub use crate::utils::*;

    pub use crate::error::prelude::*;
    pub use crate::models::prelude::*;
    pub use crate::ops::prelude::*;
    pub use crate::params::prelude::*;
    pub use crate::traits::prelude::*;
}
