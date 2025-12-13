/*
    Appellation: nn <module>
    Created At: 2025.11.28:14:59:44
    Contrib: @FL03
*/
//! This module provides network specific implementations and traits supporting the development
//! of neural network models.
//!
#[doc(inline)]
pub use self::{layer::*, traits::*};

pub mod layer;

mod traits {
    #[doc(inline)]
    pub use self::{layer::*, model::*, neural_network::*};

    mod layer;
    mod model;
    mod neural_network;
}

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::traits::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use concision_params::Params;
    use ndarray::Array1;

    #[test]
    fn test_func_layer() {
        let params = Params::<f32>::from_elem((3, 2), 0.5);
        let layer = Layer::new(|x: Array1<f32>| x.mapv(|i| i.powi(2)), params);
        // initialize some inputs
        let inputs = Array1::<f32>::linspace(1.0, 2.0, 3);
        // verify the shape of the layer's parameters
        assert_eq!(layer.params().shape(), &[3, 2]);
        // compare the actual output against the expected output
        assert_eq!(layer.forward(&inputs), Array1::from_elem(2, 7.5625));
    }

    #[test]
    fn test_linear_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::linear(params);
        // verify the shape of the layer's parameters
        assert_eq!(layer.params().shape(), &[3, 2]);
        // initialize some inputs
        let inputs = Array1::<f32>::linspace(1.0, 2.0, 3);
        // compare the actual output against the expected output
        assert_eq!(layer.forward(&inputs), Array1::from_elem(2, 2.75));
    }

    #[test]
    fn test_relu_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::relu(params);
        // initialize some inputs
        let inputs = Array1::<f32>::linspace(1.0, 2.0, 3);
        // verify the shape of the layer's parameters
        assert_eq!(layer.params().shape(), &[3, 2]);
        // compare the actual output against the expected output
        assert_eq!(layer.forward(&inputs), Array1::from_elem(2, 2.75));
    }
    #[test]
    fn test_tanh_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::tanh(params);
        // initialize some inputs
        let inputs = Array1::<f32>::linspace(1.0, 2.0, 3);
        // verify the shape of the layer's parameters
        assert_eq!(layer.params().shape(), &[3, 2]);
        // compare the actual output against the expected output
        assert!(
            (layer.forward(&inputs) - Array1::from_elem(2, 0.99185973))
                .iter()
                .all(|i| i.abs() < 1e-6)
        );
    }
}
