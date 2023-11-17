/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, ModelParams,};
use crate::prelude::{Biased, Parameterized, Weighted};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;

pub struct Model<T = f64> {
    pub features: Features,
    children: Vec<Model<T>>,
    params: ModelParams<T>
}

impl<T> Model<T>
where
    T: Float,
{
    pub fn new(features: Features) -> Self {
        Self {
            features,
            children: Vec::new(),
            params: ModelParams::new(features),
        }
    }

    pub fn features(&self) -> &Features {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut Features {
        &mut self.features
    }
}

impl<T> Model<T>
where
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        args.dot(&self.weights().t()) + self.bias()
    }
}

impl<T> Model<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T> Parameterized<T> for Model<T>
where
    T: Float,
{
    type Features = Features;
    type Params = ModelParams<T>;

    fn features(&self) -> &Features {
        &self.features
    }

    fn features_mut(&mut self) -> &mut Features {
        &mut self.features
    }
    
    fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}
