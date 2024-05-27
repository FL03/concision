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
pub use self::{model::*, perceptron::*};

pub(crate) mod model;
pub(crate) mod perceptron;

pub(crate) mod prelude {
    pub use super::perceptron::Perceptron;
}

use concision::prelude::{Activate, Module, Predict};

pub trait Neuron<T>:
    Predict<T, Output = <Self::Rho as Activate<<Self::Module as Predict<T>>::Output>>::Output>
{
    type Module: Module + Predict<T>;
    type Rho: Activate<<Self::Module as Predict<T>>::Output>;
}

pub trait Layer: Predict<Self::Elem> {
    type Elem;
    type Module: Module;
}
