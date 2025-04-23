/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! A collection of mathematical functions and utilities for signal processing, statistics, and more.

#![crate_name = "concision_neural"]
#![crate_type = "lib"]

#[allow(unused_imports)]
#[doc(inline)]
pub use self::{error::*, traits::*, types::*, utils::*};

#[allow(unused_macros)]
#[macro_use]
pub(crate) mod macros;

pub mod error;
pub mod layers;
pub mod model;
pub mod train;
pub mod utils;

pub mod traits {}

pub mod types {
    #[doc(inline)]
    pub use self::dropout::*;

    pub(crate) mod dropout;
}

#[allow(unused_imports)]
pub mod prelude {
    pub use crate::layers::prelude::*;
    pub use crate::train::prelude::*;
    pub use crate::traits::*;
    pub use crate::types::*;
    pub use crate::utils::*;
}
