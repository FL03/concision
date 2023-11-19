/*
    Appellation: shallow <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, LinearActivation};
use crate::prelude::{Forward, Layer, Parameterized};

use ndarray::prelude::{Array2, Ix2, NdFloat};
use num::Float;

pub struct ShallowNetwork<T = f64, I = LinearActivation, O = LinearActivation>
where
    T: Float,
    I: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub input: Layer<T, I>,
    pub output: Layer<T, O>,
}

impl<T, I, O> ShallowNetwork<T, I, O>
where
    T: Float,
    I: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub fn new(input: Layer<T, I>, output: Layer<T, O>) -> Self {
        Self { input, output }
    }

    pub fn input(&self) -> &Layer<T, I> {
        &self.input
    }

    pub fn output(&self) -> &Layer<T, O> {
        &self.output
    }

    pub fn validate_dims(&self) -> bool {
        self.input.features().outputs() == self.output.features().inputs()
    }
}

impl<T, I, O> Forward<Array2<T>> for ShallowNetwork<T, I, O>
where
    T: NdFloat,
    I: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.output.forward(&self.input.forward(args))
    }
}
