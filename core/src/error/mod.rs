/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

mod err;

pub mod kinds;

pub trait ErrorKind: Clone + ToString {}

impl_err!(kinds::Errors, kinds::PredictError, crate::nn::ModelError);

pub(crate) mod prelude {
    pub use super::err::Error;
    pub use super::kinds::*;
}
