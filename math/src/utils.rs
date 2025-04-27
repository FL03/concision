/*
    Appellation: utils <module>
    Contrib: @FL03
*/
//! utilties supporting various mathematical routines for machine learning tasks.
#[doc(inline)]
pub use self::prelude::*;

pub mod arith;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::arith::*;
}
