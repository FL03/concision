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

pub trait MultiLayerPerceptron {
    type Input;
    type Hidden;
    type Output;
}

pub trait Container {
    type Elem;
}

pub trait Params {
    type Data: Container<Elem = Self::Elem>;
    type Dim;
    type Elem;
}

pub trait Neuron<F, A, D> {
    type Rho;
}
