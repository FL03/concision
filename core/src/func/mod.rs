/*
   Appellation: func <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Functional
pub use self::prelude::*;

#[macro_use]
pub mod activate;
pub mod loss;

pub(crate) mod prelude {
    pub use super::activate::prelude::*;
    pub use super::loss::prelude::*;
}
