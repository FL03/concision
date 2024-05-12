/*
    Appellation: mlp <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Multi-Layer Perceptron (MLP)
//!
//! A multi-layer perceptron (MLP) is a class of feed-forward artificial neural networks (FFN).
//!
//!
#[doc(inline)]
pub use self::perceptron::Perceptron;

pub mod perceptron;

pub trait MultiLayerPerceptron {
    type Input;
    type Hidden;
    type Output;
}

pub trait Neuron<T, F> {
}

pub trait Rho<T> {
    type Output;

    fn activate(&self, args: T) -> Self::Output;
}
