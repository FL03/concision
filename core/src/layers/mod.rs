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

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::{activator::*, layers::*};

    mod activator;
    mod layers;
    mod store;
}

pub(crate) mod types {
    #[doc(inline)]
    pub use self::aliases::*;

    mod aliases;
}

pub(crate) mod prelude {
    pub use super::layer::*;
    pub use super::types::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use concision_params::Params;
    use ndarray::{Array1, array};

    #[test]
    fn test_linear_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::linear(params);

        assert_eq!(layer.params().shape(), &[3, 2]);

        let inputs = Array1::linspace(1.0_f32, 2.0_f32, 3);
        println!("{:?}", inputs);
        assert_eq!(layer.forward(&inputs), array![2.75, 2.75]);
    }

    #[test]
    fn test_relu_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::relu(params);

        assert_eq!(layer.params().shape(), &[3, 2]);

        let inputs = Array1::linspace(1.0_f32, 2.0_f32, 3);
        assert_eq!(layer.forward(&inputs), array![2.75, 2.75]);
    }
    #[cfg(feature = "approx")]
    #[test]
    fn test_tanh_layer() {
        let params = Params::from_elem((3, 2), 0.5_f32);
        let layer = Layer::tanh(params);

        assert_eq!(layer.params().shape(), &[3, 2]);

        let inputs = Array1::linspace(1.0_f32, 2.0_f32, 3);
        approx::assert_abs_diff_eq!(
            layer.forward(&inputs),
            Array1::from_elem(2, 0.99185973),
            epsilon = 1e-6
        );
    }
}
