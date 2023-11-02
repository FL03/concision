/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, LayerType};
use crate::neurons::activate::Activator;
use ndarray::prelude::{Array1, Array2};
use num::Float;

pub trait L<T: Float> {
    //
    fn process(&self, args: &Array2<T>, rho: impl Activator<T>) -> Array2<T>
    where
        T: 'static,
    {
        let z = args.dot(self.weights()) + self.bias();
        z.mapv(|x| rho.activate(x))
    }

    fn bias(&self) -> &Array1<T>;

    fn weights(&self) -> &Array2<T>;
}

pub trait Linear<T: Float> {
    fn linear(&self, data: &Array2<T>) -> Array2<T>
    where
        T: 'static;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layer {
    bias: Option<Array1<f64>>,
    features: Features,
    layer: LayerType,
    weights: Array2<f64>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, bias: bool, layer: LayerType) -> Self {
        let features = Features::new(inputs, outputs);
        let bias = if bias {
            Some(Array1::ones(outputs))
        } else {
            None
        };
        let weights = Array2::ones((features.inputs(), features.outputs()));

        Self {
            bias,
            features,
            layer,
            weights,
        }
    }

    pub fn bias(&self) -> &Option<Array1<f64>> {
        &self.bias
    }

    pub fn layer(&self) -> &LayerType {
        &self.layer
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn set_layer(&mut self, layer: LayerType) {
        self.layer = layer;
    }
}
