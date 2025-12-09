/*
    appellation: aliases <module>
    authors: @FL03
*/
#[cfg(feature = "alloc")]
use crate::layers::Activator;
use crate::layers::{LayerBase, Linear, ReLU, Sigmoid, Tanh};
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[cfg(feature = "alloc")]
/// A type alias for a [`LayerBase`] configured with a dynamic [`Activator`].
pub type LayerDyn<A, T> = LayerBase<Box<dyn Activator<A, Output = A> + 'static>, T>;

/// A type alias for a [`LayerBase`] configured with a [`Linear`] activation function.
pub type LinearLayer<T> = LayerBase<Linear, T>;
/// A type alias for a [`LayerBase`] configured with a [`Sigmoid`] activation function.
pub type SigmoidLayer<T> = LayerBase<Sigmoid, T>;
/// A type alias for a [`LayerBase`] configured with a [`Tanh`] activation function.
pub type TanhLayer<T> = LayerBase<Tanh, T>;
/// A type alias for a [`LayerBase`] configured with a [`ReLU`] activation function.
pub type ReluLayer<T> = LayerBase<ReLU, T>;
