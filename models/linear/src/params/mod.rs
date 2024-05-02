/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{kinds::*, params::*};

mod kinds;
mod params;

mod impls {
    mod impl_params;
    #[cfg(feature = "rand")]
    mod impl_rand;
}

pub(crate) mod prelude {
    pub use super::kinds::*;
    pub use super::params::LinearParams;
}
