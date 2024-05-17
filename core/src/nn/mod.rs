/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::ModelError, model::prelude::*};

pub mod error;
pub mod model;

pub(crate) mod prelude {
    pub use super::error::ModelError;
    pub use super::model::prelude::*;
}

#[cfg(any(feature = "alloc", feature = "std"))]
pub type ForwardDyn<T = nd::Array2<f64>, O = T> =
    crate::rust::Box<dyn crate::Forward<T, Output = O>>;

#[cfg(test)]
mod tests {}
