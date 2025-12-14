/*
    Appellation: layer <module>
    Contrib: @FL03
*/
mod impl_layer;
mod impl_layer_deprecated;
mod impl_layer_repr;

#[doc(inline)]
pub use self::types::*;

/// The [`Layer`] implementation works to provide a generic interface for layers within a
/// neural network by associating an activation function `F` with a set of parameters `P`.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Layer<F, P> {
    /// the activation function of the layer
    pub rho: F,
    /// the parameters of the layer; often weights and biases
    pub params: P,
}

mod types {
    use super::Layer;
    use crate::activate::{HeavySide, Linear, ReLU, Sigmoid, TanhActivator};
    #[cfg(feature = "alloc")]
    use alloc::boxed::Box;
    use concision_params::{Params, ParamsBase};

    /// A type alias for a layer configured to use the [`ParamsBase`] instance
    pub type LayerParamsBase<F, S, D = ndarray::Ix2, A = f32> = Layer<F, ParamsBase<S, D, A>>;
    /// A type alias for an owned [`Layer`] configured to use the standard [`Params`] instance
    pub type LayerParams<F, A = f32, D = ndarray::Ix2> = Layer<F, Params<A, D>>;
    /// A type alias for a layer using a linear activation function.
    pub type LinearLayer<T> = Layer<Linear, T>;
    /// A type alias for a [`Layer`] using a sigmoid activation function.
    pub type SigmoidLayer<T> = Layer<Sigmoid, T>;
    /// An alias for a [`Layer`] that uses the hyperbolic tangent function.
    pub type TanhLayer<T> = Layer<TanhActivator, T>;
    /// A [`Layer`] type using the ReLU activation function.
    pub type ReluLayer<T> = Layer<ReLU, T>;
    /// A [`Layer`] type using the heavyside activation function.
    pub type HeavySideLayer<T> = Layer<HeavySide, T>;

    #[cfg(feature = "alloc")]
    /// A dynamic instance of the layer using a boxed activator.
    pub type LayerDyn<'a, T> = Layer<Box<dyn crate::Activator<T, Output = T> + 'a>, T>;
    #[cfg(feature = "alloc")]
    /// A dynamic, functional alias of the [`Layer`] implementation leveraging boxed closures.
    pub type FnLayer<'a, T> = Layer<Box<dyn Fn(T) -> T + 'a>, T>;
}
