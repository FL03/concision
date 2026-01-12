/*
    Appellation: iter <module>
    Created At: 2026.01.12:11:08:01
    Contrib: @FL03
*/
//! iterators for parameters within a neural network
#[doc(inline)]
pub use self::iter_params::*;
// modules
pub mod iter_params;
// prelude (local)
#[doc(hidden)]
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::iter_params::*;
}
