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
pub use self::perceptron::*;

pub(crate) mod perceptron;

pub(crate) mod prelude {
    pub use super::perceptron::Perceptron;
}

use concision::Forward;

pub trait Neuron: Forward<Self::Elem> {
    type Elem;
    type Rho;
}
