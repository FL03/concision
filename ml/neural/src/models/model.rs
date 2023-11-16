/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, ModelParams, Parameterized};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;

pub struct Model<T = f64> {
    pub features: Features,
    params: ModelParams<T>,
}

impl<T> Model<T>
where
    T: Float,
{
    pub fn new(features: Features) -> Self {
        Self {
            features,
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
        args.dot(&self.params().weights().t()) + self.params().bias()
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
    fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}
