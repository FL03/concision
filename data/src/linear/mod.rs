/*
   Appellation: linear <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Layers
pub use self::{features::*, layer::*, utils::*};

pub(crate) mod features;
pub(crate) mod layer;

pub(crate) mod utils {
    use ndarray::prelude::{Array1, Array2};
    use num::Float;

    pub fn linear_transformation<T: Float + 'static>(
        data: &Array2<T>,
        bias: &Array1<T>,
        weights: &Array2<T>,
    ) -> Array2<T> {
        data.dot(&weights.t()) + bias
    }
}
