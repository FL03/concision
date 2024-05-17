/*
   Appellation: func <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Functional
pub use self::prelude::*;

pub mod activate;
pub mod dropout;
pub mod loss;

pub(crate) mod prelude {
    pub use super::activate::prelude::*;
    #[cfg(feature = "rand")]
    pub use super::dropout::*;
    pub use super::loss::prelude::*;
}
