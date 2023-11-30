/*
    Appellation: mlp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Multi-Layer Perceptron
//!

use crate::func::activate::{Activate, Linear};
use crate::layers::{Layer, LayerShape, Stack};
use crate::prelude::{Features, Forward, Parameterized};

use ndarray::prelude::{Array2, Ix2, NdFloat};
use ndarray::IntoDimension;
use num::Float;

pub struct MLP<T = f64, I = Linear, H = Linear, O = Linear>
where
    T: Float,
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
{
    pub input: Layer<T, I>,
    pub hidden: Stack<T, H>,
    pub output: Layer<T, O>,
}

impl<T, I, H, O> MLP<T, I, H, O>
where
    T: Float,
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
{
    pub fn new(input: Layer<T, I>, hidden: Stack<T, H>, output: Layer<T, O>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }

    pub fn input(&self) -> &Layer<T, I> {
        &self.input
    }

    pub fn hidden(&self) -> &Stack<T, H> {
        &self.hidden
    }

    pub fn output(&self) -> &Layer<T, O> {
        &self.output
    }
}
impl<T, I, H, O> MLP<T, I, H, O>
where
    T: Float,
    I: Activate<T> + Default,
    H: Activate<T> + Default,
    O: Activate<T> + Default,
{
    // pub fn create<Sh: IntoDimension<Dim = Ix2>>(inputs: Sh, hidden: impl IntoIterator<Item = Sh>, outputs: Sh) -> Self {
    //     let input = LayerShape::from_dimension(inputs);
    //     let hidden = Stack::new().build_layers(hidden);
    //     let output = LayerShape::from_dimension(outputs);

    //     Self::new(Layer::from(input), hidden, Layer::new(output))
    // }
}

impl<T, I, H, O> MLP<T, I, H, O>
where
    T: Float,
    I: Activate<T> + Clone,
    H: Activate<T> + Clone,
    O: Activate<T> + Clone,
{
    pub fn validate_dims(&self) -> bool {
        self.hidden.validate_shapes()
            && self.input.features().outputs() == self.hidden.first().unwrap().features().inputs()
            && self.output.features().inputs() == self.hidden.last().unwrap().features().outputs()
    }
}

impl<T, I, H, O> Forward<Array2<T>> for MLP<T, I, H, O>
where
    T: NdFloat,
    I: Activate<T> + Clone,
    H: Activate<T> + Clone,
    O: Activate<T> + Clone,
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
