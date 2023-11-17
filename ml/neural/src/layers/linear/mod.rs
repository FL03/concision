/*
   Appellation: linear <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Layer
pub use self::{layer::*, regress::*, utils::*};

pub(crate) mod layer;
pub(crate) mod regress;

pub(crate) mod utils {
    use ndarray::prelude::{Array1, Array2, NdFloat};

    pub fn linear_transformation<T>(
        data: &Array2<T>,
        bias: &Array1<T>,
        weights: &Array2<T>,
    ) -> Array2<T>
    where
        T: NdFloat,
    {
        data.dot(&weights.t()) + bias
    }

    pub fn linear_node<T>(data: &Array2<T>, bias: &T, weights: &Array1<T>) -> Array1<T>
    where
        T: NdFloat,
    {
        data.dot(&weights.t()) + bias.clone()
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
