/*
    Appellation: network <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, LinearActivation};
use crate::prelude::{Forward, Layer, Parameterized};

use ndarray::prelude::{Array2, Ix2, NdFloat};
use num::Float;

pub struct DeepNetwork<T = f64, I = LinearActivation, H = LinearActivation, O = LinearActivation>
where
    T: Float,
    I: Activate<T, Ix2>,
    H: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub input: Layer<T, I>,
    pub hidden: Vec<Layer<T, H>>,
    pub output: Layer<T, O>,
}

impl<T, I, H, O> DeepNetwork<T, I, H, O>
where
    T: Float,
    I: Activate<T, Ix2>,
    H: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    pub fn new(input: Layer<T, I>, hidden: Vec<Layer<T, H>>, output: Layer<T, O>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }

    pub fn validate_dims(&self) -> bool {
        let mut dim = true;
        for (i, layer) in self.hidden.iter().enumerate() {
            if i == 0 {
                dim = self.input.features().outputs() == self.hidden[i].features().inputs()
            } else if i == self.hidden.len() - 1 {
                dim = dim && layer.features().outputs() == self.output.features().inputs();
            } else {
                dim = dim && layer.features().outputs() == self.hidden[i + 1].features().inputs();
            }
        }
        dim
    }
}

impl<T, I, H, O> Forward<Array2<T>> for DeepNetwork<T, I, H, O>
where
    T: NdFloat,
    I: Activate<T, Ix2>,
    H: Activate<T, Ix2>,
    O: Activate<T, Ix2>,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let mut out = self.input.forward(args);
        for layer in &self.hidden {
            out = layer.forward(&out);
        }
        self.output.forward(&out)
    }
}
