/*
    Appellation: activate <module>
    Created At: 2026.01.13:17:07:42
    Contrib: @FL03
*/
//! this module provides the [`Activate`] trait alongside additional primitives and utilities
//! for activating neurons within a neural network.
#[doc(inline)]
pub use self::{rho::*, traits::*, utils::*};

pub mod rho;

mod impls {
    mod impl_activate_linear;
    mod impl_activate_nonlinear;
    mod impl_activator;
}

mod traits {
    #[doc(inline)]
    pub use self::{activate::*, ops::*};

    mod activate;
    mod ops;
}

pub mod utils {
    #[doc(inline)]
    pub use self::funcs::*;

    mod funcs;
}

#[doc(hidden)]
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::utils::*;
}
