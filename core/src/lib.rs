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

pub mod prelude {

    pub use crate::primitives::*;
    pub use crate::utils::*;

    pub use crate::error::prelude::*;
    pub use crate::models::prelude::*;
    pub use crate::ops::prelude::*;
    pub use crate::params::prelude::*;
    pub use crate::traits::prelude::*;
}
