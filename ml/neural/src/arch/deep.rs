/*
    Appellation: network <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, Linear};
use crate::models::Stack;
use crate::prelude::{Forward, Layer, Parameterized};

use ndarray::prelude::{Array2, Ix2, NdFloat};
use num::Float;

pub struct DeepNetwork<T = f64, I = Linear, H = Linear, O = Linear>
where
    T: Float,
    I: Activate<T, Ix2>,
    H: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub input: Layer<T, I>,
    pub hidden: Stack<T, H>,
    pub output: Layer<T, O>,
}

impl<T, I, H, O> DeepNetwork<T, I, H, O>
where
    T: Float,
    I: Activate<T, Ix2>,
    H: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub fn new(input: Layer<T, I>, hidden: Stack<T, H>, output: Layer<T, O>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }
}
impl<T, I, H, O> DeepNetwork<T, I, H, O>
where
    T: Float,
    I: Activate<T, Ix2> + Clone,
    H: Activate<T, Ix2> + Clone,
    O: Activate<T, Ix2> + Clone,
{
    pub fn validate_dims(&self) -> bool {
        self.hidden.validate_shapes()
            && self.input.features().outputs() == self.hidden.first().unwrap().features().inputs()
            && self.output.features().inputs() == self.hidden.last().unwrap().features().outputs()
    }
}

impl<T, I, H, O> Forward<Array2<T>> for DeepNetwork<T, I, H, O>
where
    T: NdFloat,
    I: Activate<T, Ix2> + Clone,
    H: Activate<T, Ix2> + Clone,
    O: Activate<T, Ix2> + Clone,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let mut out = self.input.forward(args);
        for layer in self.hidden.clone().into_iter() {
            out = layer.forward(&out);
        }
        self.output.forward(&out)
    }
}
