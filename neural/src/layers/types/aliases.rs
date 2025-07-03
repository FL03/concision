/*
    appellation: aliases <module>
    authors: @FL03
*/
use crate::layers::{Activator, LayerBase, Linear, ReLU, Sigmoid, Tanh};
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[cfg(feature = "alloc")]
/// A type alias for a [`LayerBase`] configured with a dynamic [`Activator`].
pub type LayerDyn<A, S, D> = LayerBase<Box<dyn Activator<A, Output = A> + 'static>, S, D>;

/// A type alias for a [`LayerBase`] configured with a [`Linear`] activation function.
pub type LinearLayer<S, D> = LayerBase<Linear, S, D>;
/// A type alias for a [`LayerBase`] configured with a [`Sigmoid`] activation function.
pub type SigmoidLayer<S, D> = LayerBase<Sigmoid, S, D>;
/// A type alias for a [`LayerBase`] configured with a [`Tanh`] activation function.
pub type TanhLayer<S, D> = LayerBase<Tanh, S, D>;
/// A type alias for a [`LayerBase`] configured with a [`ReLU`] activation function.
pub type ReluLayer<S, D> = LayerBase<ReLU, S, D>;
