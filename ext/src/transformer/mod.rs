/*
    Appellation: transformers <module>
    Created At: 2025.11.26:13:59:39
    Contrib: @FL03
*/
//! A custom transformer model implementation for the Concision framework.
#[doc(inline)]
pub use self::model::*;

mod model;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::model::*;
}
