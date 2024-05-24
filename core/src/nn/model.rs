/*
    Appellation: model <traits>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::module::*;

pub mod config;
pub mod module;
#[doc(hidden)]
pub mod repo;

pub(crate) mod prelude {
    pub use super::config::*;
    pub use super::module::*;
    pub use super::Model;
}

use crate::traits::Forward;

pub trait Model: Module
where
    Self: Forward<Self::Data>,
{
    type Ctx;
    type Data;

    fn children(&self) -> Vec<ModuleDyn<Self::Ctx, Self::Params>>;

    fn context(&self) -> Self::Ctx;
}

/// This trait describes any neural networks or models that
/// adhears to the deep netural network architecture.
/// This design considers a single input and output layer, while
/// allowing for any number of hidden layers to be persisted.
///
/// The `HIDDEN` constant is used to specify the number of hidden layers
/// and is used to compute the total number of layers (HIDDEN + 2)
pub trait DeepNeuralNetwork<S, T>: Forward<S, Output = T> {
    const HIDDEN: Option<usize> = None;

    type Input: Forward<S, Output = T>;
    type Hidden: Forward<T, Output = T>; // The type of `hidden` layers; all hidden layers implement the same activation function
    type Out: Forward<T, Output = T>;

    fn input(&self) -> &Self::Input;

    fn hidden(&self) -> &[Self::Hidden];

    fn output(&self) -> &Self::Out;

    fn nlayers(&self) -> usize {
        self.nhidden() + 2
    }

    fn nhidden(&self) -> usize {
        Self::HIDDEN.unwrap_or_else(|| self.hidden().len())
    }
}
