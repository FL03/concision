/*
    appellation: types <module>
    authors: @FL03
*/
#[cfg(feature = "alloc")]
pub use self::impl_alloc::*;

use crate::activate::{HeavySide, HyperbolicTangent, Linear, ReLU, Sigmoid};
use crate::layers::Layer;
use concision_params::{Params, ParamsBase};
use ndarray::Ix2;

pub type LayerParamsBase<F, S, D = Ix2, A = f32> = Layer<F, ParamsBase<S, D, A>>;

pub type LayerParams<F, A = f32, D = Ix2> = Layer<F, Params<A, D>>;

/// A type alias for a [`Layer`] configured with a [`Linear`] activation function.
pub type LinearLayer<T> = Layer<Linear, T>;
/// A type alias for a [`Layer`] configured with a [`Sigmoid`] activation function.
pub type SigmoidLayer<T> = Layer<Sigmoid, T>;
/// A type alias for a [`Layer`] configured with a [`Tanh`] activation function.
pub type TanhLayer<T> = Layer<HyperbolicTangent, T>;
/// A type alias for a [`Layer`] configured with a [`ReLU`] activation function.
pub type ReluLayer<T> = Layer<ReLU, T>;
/// A type alias for a [`Layer`] configured with a [`HeavySide`] activation function.
/// This activation function is also known as the step function.
pub type HeavysideLayer<T> = Layer<HeavySide, T>;

#[cfg(feature = "alloc")]
mod impl_alloc {
    use crate::activate::Activator;
    use crate::layers::Layer;
    use alloc::boxed::Box;

    /// A type alias for a [`Layer`] configured with a dynamic [`Activator`].
    pub type LayerDyn<A, T> = Layer<Box<dyn Activator<A, Output = A> + 'static>, T>;
}
