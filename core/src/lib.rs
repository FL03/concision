/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![crate_name = "concision_core"]

#[cfg(not(feature = "std"))]
extern crate alloc;
extern crate ndarray as nd;

pub use self::{primitives::*, traits::prelude::*, utils::*};

pub(crate) mod primitives;

pub(crate) mod utils;

pub mod errors;
pub mod ops;
pub mod params;
pub mod time;
pub mod traits;

pub mod prelude {

    pub use crate::primitives::*;
    pub use crate::utils::*;

    pub use crate::errors::*;
    pub use crate::time::*;
    pub use crate::traits::prelude::*;
}
