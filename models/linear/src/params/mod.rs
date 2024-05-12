/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::entry::{Entry, Param};
pub use self::mode::*;
pub use self::params::ParamsBase;

mod params;

pub mod entry;
pub mod mode;

#[doc(inline)]
pub use crate::primitives::params::*;

pub(crate) mod prelude {
    pub use super::mode::*;
}
