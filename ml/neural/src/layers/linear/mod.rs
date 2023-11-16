/*
   Appellation: linear <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Layer
pub use self::{layer::*, regress::*, utils::*};

pub(crate) mod layer;
pub(crate) mod regress;

use ndarray::prelude::Array2;
use num::Float;

// pub trait Lin<T = f64> where T: Float {

//     fn forward(&self, data: &Array2<T>) -> Array2<T>;
// }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{Features, Forward};
    use ndarray::prelude::Array2;

    #[test]
    fn test_linear_layer() {
        let (samples, inputs, outputs) = (20, 2, 2);
        let features = Features::new(inputs, outputs);
        let data = Array2::<f64>::ones((20, features.inputs()));
        let layer = LinearLayer::from_features(features).init_weight();
        let pred = layer.forward(&data);
        assert_eq!(pred.dim(), (samples, outputs));
    }
}
