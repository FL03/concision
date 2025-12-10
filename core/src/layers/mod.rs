/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//! This module provides the [`Layer`] implementation along with supporting traits and types.
//!
//! struct, a generic representation of a neural network
//! layer by associating
//!
#[doc(inline)]
pub use self::{layer::Layer, traits::*, types::*};

mod layer;

pub mod seq;

mod traits {
    #[doc(inline)]
    pub use self::layers::*;

    mod layers;
}

mod types {
    #[doc(inline)]
    pub use self::aliases::*;

    mod aliases;
}

pub(crate) mod prelude {
    pub use super::layer::Layer;
    pub use super::traits::*;
    pub use super::types::*;
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
        // initialize some inputs
        let inputs = Array1::<f32>::linspace(1.0, 2.0, 3);
        // verify the shape of the layer's parameters
        assert_eq!(layer.params().shape(), &[3, 2]);
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
