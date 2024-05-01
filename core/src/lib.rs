/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_core"]

#[cfg(no_std)]
extern crate alloc;
extern crate ndarray as nd;
#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;

pub use self::{error::Error, primitives::*, traits::prelude::*, utils::*};

#[cfg(feature = "rand")]
pub use self::rand::prelude::*;

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;
pub(crate) mod utils;

pub mod error;
pub mod func;
pub mod models;
pub mod ops;
pub mod params;
// #[cfg(feature = "rand")]
pub mod rand;
pub mod traits;

#[allow(unused_imports)]
pub(crate) mod rust {
    pub(crate) use core::*;

    #[cfg(no_std)]
    pub(crate) use self::no_std::*;
    #[cfg(feature = "std")]
    pub(crate) use self::with_std::*;

    #[cfg(no_std)]
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

    #[cfg(no_std)]
    pub type Map<K, V> = collections::BTreeMap<K, V>;
    #[cfg(feature = "std")]
    pub type Map<K, V> = collections::HashMap<K, V>;
}

pub mod prelude {

    pub use super::primitives::*;
    pub use super::utils::*;

    pub use super::error::prelude::*;
    pub use super::func::prelude::*;
    pub use super::models::prelude::*;
    pub use super::ops::prelude::*;
    pub use super::params::prelude::*;
    #[cfg(feature = "rand")]
    pub use super::rand::prelude::*;
    pub use super::traits::prelude::*;

    pub(crate) use super::rust::*;
}
