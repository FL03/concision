/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::kinds::*;

pub(crate) mod kinds;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, scsys::VariantConstructors, thiserror::Error)]
pub enum Error {
    #[error("Model Error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Shape Error: {0}")]
    ShapeError(#[from] nd::ShapeError),
    #[error("Unknown Error: {0}")]
    Unknown(String),
}
