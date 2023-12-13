/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use crate::func::activate::Activator;
use crate::prelude::{Features, Forward, Layer, Parameterized};

use ndarray::prelude::{Array2, NdFloat};
use num::Float;

pub struct FFN<T = f64>
where
    T: Float,
{
    layers: Vec<Layer<T, Activator<T>>>,
}

impl<T> FFN<T>
where
    T: Float,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer<T, Activator<T>>) {
        self.layers.push(layer);
    }

    pub fn validate_dims(&self) -> bool {
        let depth = self.layers.len();
        let mut dim = true;
        for (i, layer) in self.layers[..(depth - 1)].into_iter().enumerate() {
            dim = dim && layer.features().outputs() == self.layers[i + 1].features().inputs();
        }
        dim
    }
}

impl<T> Forward<Array2<T>> for FFN<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let mut out = args.clone();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }
}
