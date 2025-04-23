/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//!
#[doc(inline)]
pub use self::store::ModelParams;

pub mod layer;
pub mod store;

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::store::*;
}
