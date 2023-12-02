/*
    Appellation: shallow <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, Linear, ReLU, Softmax};
use crate::prelude::{Features, Forward, Layer, LayerShape, Parameterized};

use ndarray::prelude::{Array2, NdFloat};
use num::Float;

pub struct ShallowNetwork<T = f64, I = Linear, H = ReLU, O = Softmax>
where
    T: Float,
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
{
    pub input: Layer<T, I>,
    pub hidden: Layer<T, H>,
    pub output: Layer<T, O>,
}

impl<T, I, H, O> ShallowNetwork<T, I, H, O>
where
    T: Float,
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
{
    pub fn new(input: Layer<T, I>, hidden: Layer<T, H>, output: Layer<T, O>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }

    pub fn hidden(&self) -> &Layer<T, H> {
        &self.hidden
    }

    pub fn input(&self) -> &Layer<T, I> {
        &self.input
    }

    pub fn output(&self) -> &Layer<T, O> {
        &self.output
    }

    pub fn validate_dims(&self) -> bool {
        self.input.features().outputs() == self.hidden.features().inputs()
            && self.hidden.features().outputs() == self.output.features().inputs()
    }
}

impl<T, I, H, O> ShallowNetwork<T, I, H, O>
where
    T: Float,
    I: Activate<T> + Default,
    H: Activate<T> + Default,
    O: Activate<T> + Default,
{
    pub fn create(inputs: usize, outputs: usize) -> Self {
        let s1 = LayerShape::new(inputs, outputs);
        let s2 = LayerShape::new(outputs, outputs);

        let input = Layer::<T, I>::from(s1);
        let hidden = Layer::<T, H>::new(s2);
        let output = Layer::<T, O>::new(s2);

        Self::new(input, hidden, output)
    }
}

impl<T, I, H, O> Forward<Array2<T>> for ShallowNetwork<T, I, H, O>
where
    T: NdFloat,
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.output
            .forward(&self.hidden.forward(&self.input.forward(args)))
    }
}
