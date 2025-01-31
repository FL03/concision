/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[cfg(feature = "alloc")]
pub use self::types::*;
pub use self::{dropout::Dropout, model::prelude::*};

pub mod dropout;
pub mod mask;
pub mod model;
#[doc(hidden)]
pub mod optim;

mod traits;

pub(crate) mod prelude {
    pub use super::dropout::Dropout;
    pub use super::mask::Mask;
    pub use super::model::prelude::*;
    pub use super::optim::prelude::*;
}

#[cfg(feature = "alloc")]
mod types {
    use crate::rust::Box;
    use nd::prelude::Array2;

    pub type ForwardDyn<T = Array2<f64>, O = T> = Box<dyn crate::Forward<T, Output = O>>;

    pub type PredictDyn<T = Array2<f64>, O = T> = Box<dyn crate::Predict<T, Output = O>>;
}

#[cfg(test)]
mod tests {}
