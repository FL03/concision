/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Neuron, Perceptron};

pub struct Mlp<A, I, H, O>
where
    I: Neuron<A>,
    H: Neuron<A>,
    O: Neuron<A>,
{
    input: Perceptron<I::Module, I::Rho>,
    hidden: Vec<Perceptron<H::Module, H::Rho>>,
    output: Perceptron<O::Module, O::Rho>,
}

impl<A, I, H, O> Mlp<A, I, H, O>
where
    I: Neuron<A>,
    H: Neuron<A>,
    O: Neuron<A>,
{
    pub const fn input(&self) -> &Perceptron<I::Module, I::Rho> {
        &self.input
    }

    pub fn hidden(&self) -> &[Perceptron<H::Module, H::Rho>] {
        &self.hidden
    }

    pub const fn output(&self) -> &Perceptron<O::Module, O::Rho> {
        &self.output
    }
}
