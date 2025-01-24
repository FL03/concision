/*
    Appellation: traits <module>
    Contrib: @FL03
*/
#[doc(inline)]
pub use self::prelude::*;

pub mod root;
pub mod unary;

pub(crate) mod prelude {
    pub use super::root::*;
    pub use super::unary::*;
}