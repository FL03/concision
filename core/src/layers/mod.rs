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
pub use self::{layer::*, types::*};

mod layer;

pub mod seq;

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::types::*;
}

mod types {
    use super::Layer;
    use crate::activate::{HeavySide, HyperbolicTangent, Linear, ReLU, Sigmoid};
    use concision_params::{Params, ParamsBase};
    use ndarray::Ix2;

    pub type LayerParamsBase<F, S, D = Ix2, A = f32> = Layer<F, ParamsBase<S, D, A>>;

    pub type LayerParams<F, A = f32, D = Ix2> = Layer<F, Params<A, D>>;

    /// A type alias for a [`Layer`] configured with a [`Linear`] activation function.
    pub type LinearLayer<T> = Layer<Linear, T>;
    /// A type alias for a [`Layer`] configured with a [`Sigmoid`] activation function.
    pub type SigmoidLayer<T> = Layer<Sigmoid, T>;
    /// A type alias for a [`Layer`] configured with a [`HyperbolicTangent`] activation function.
    pub type TanhLayer<T> = Layer<HyperbolicTangent, T>;
    /// A type alias for a [`Layer`] configured with a [`ReLU`] activation function.
    pub type ReluLayer<T> = Layer<ReLU, T>;
    /// A type alias for a [`Layer`] configured with a [`HeavySide`] activation function.
    /// This activation function is also known as the step function.
    pub type HeavysideLayer<T> = Layer<HeavySide, T>;

    #[cfg(feature = "alloc")]
    /// A dynamic instance of the layer using a boxed activator.
    pub type LayerDyn<'a, T> =
        Layer<alloc::boxed::Box<dyn crate::Activator<T, Output = T> + 'a>, T>;
    #[cfg(feature = "alloc")]
    /// A dynamic, functional alias of the [`Layer`] implementation leveraging boxed closures.
    pub type FnLayer<'a, T> = Layer<alloc::boxed::Box<dyn Fn(T) -> T + 'a>, T>;
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
