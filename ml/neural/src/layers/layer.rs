/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, LayerType};
use crate::bias::Bias;
use crate::neurons::activate::Activator;
use crate::prop::Forward;

use ndarray::prelude::{Array1, Array2};
use ndarray_rand::rand_distr::uniform::SampleUniform;
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
pub struct Layer<T: Float = f64> {
    bias: Bias<T>,
    features: Features,
    layer: LayerType,
    weights: Array2<T>,
}

impl<T: Float> Layer<T> {
    pub fn new(inputs: usize, outputs: usize, bias: bool, layer: LayerType) -> Self where T: SampleUniform {
        let features = Features::new(inputs, outputs);
        let bias = if bias {
            Bias::biased(outputs)
        } else {
            Bias::default()
        };
        let weights = Array2::ones((features.inputs(), features.outputs()));

        Self {
            bias,
            features,
            layer,
            weights,
        }
    }

    pub fn bias(&self) -> &Bias<T> {
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

    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }
}

impl<T: Float + 'static> Forward<Array2<T>> for Layer<T> {
    type Output = Array2<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        data.dot(&self.weights().t()) + self.bias()
    }
}