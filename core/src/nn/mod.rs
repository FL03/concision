/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[cfg(any(feature = "alloc", feature = "std"))]
pub use self::types::*;
pub use self::{dropout::*, model::prelude::*};

pub mod dropout;
pub mod mask;
pub mod model;
#[doc(hidden)]
pub mod optim;

pub(crate) mod prelude {
    pub use super::dropout::*;
    pub use super::mask::prelude::*;
    pub use super::model::prelude::*;
    pub use super::optim::prelude::*;
}

#[cfg(any(feature = "alloc", feature = "std"))]
mod types {
    use crate::rust::Box;
    use nd::prelude::Array2;

    pub type ForwardDyn<T = Array2<f64>, O = T> = Box<dyn crate::Forward<T, Output = O>>;
}

#[cfg(test)]
mod tests {}
